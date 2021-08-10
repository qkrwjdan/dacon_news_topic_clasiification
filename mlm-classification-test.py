import os
import random
import logging
from IPython.display import display, HTML
from tqdm import tqdm, tqdm_notebook, tnrange

import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForMaskedLM, AutoModelForPreTraining, TrainingArguments, Trainer

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score

print(torch.cuda.device_count())

device = torch.device("cuda:0")

def seed_everything(seed: int = 17):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
seed_everything(17)

model_checkpoint = "klue/bert-base"
RANDOM_SEED = 17

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
batch_size_list = [4]
learning_rate_list = [ 1e-5, 2e-5, 5e-5, 7e-5, 1e-4]

max_val_accuracy = 0.0

for batch_size in batch_size_list:
    for lr in learning_rate_list:

        dataset = pd.read_csv("data/train_data.csv",index_col=False)
        dataset_augmented = pd.read_csv("data/train_data_m2m_translation.csv",index_col=False)
        test = pd.read_csv("data/test_data.csv",index_col=False)

        dataset_augmented["topic_idx"] = dataset["topic_idx"]

        dataset_train, dataset_val = train_test_split(dataset,test_size = 0.1,random_state = RANDOM_SEED)

        train_dataset_augmented_title = dataset_augmented["title"][dataset_train["index"]]
        train_dataset_augmented_topic_idx = dataset_augmented["topic_idx"][dataset_train["index"]]
        train_dataset_augmented = pd.DataFrame({'title' : train_dataset_augmented_title.tolist(), "topic_idx" : train_dataset_augmented_topic_idx.tolist()})

        dataset_train = pd.concat([dataset_train,train_dataset_augmented])

        topic_token_dict = {0:4038,1:3674,2:3647,3:3697,4:3665,5:4559,6:3713}
        token_topic_dict = {4038 : 0, 3674 : 1, 3647 : 2, 3697 : 3, 3665 : 4, 4559 : 5, 3713 : 6}
        topic_dict = {0: "과학", 1:"경제", 2:"사회", 3:"문화", 4:"세계", 5:"스포츠", 6 : "정치"}

        tmp = []

        for title in dataset_train["title"]:
            sentence = title + ".[SEP] 이 문장은 [MASK]"
            tmp.append(sentence)
        dataset_train["title"] = tmp

        tmp = []

        for title in dataset_val["title"]:
            sentence = title + ".[SEP] 이 문장은 [MASK]"
            tmp.append(sentence)
        dataset_val["title"] = tmp
            
        tmp = []
        for title in test["title"]:
            sentence = title + ".[SEP] 이 문장은 [MASK]"
            tmp.append(sentence)

        test["title"] = tmp

        def bert_tokenize(dataset,sent_key,label_key,tokenizer):
            if label_key is None :
                labels = [np.int64(0) for i in dataset[sent_key]]
            else :
                labels = [np.int64(i) for i in dataset[label_key]]
            
            sentences = tokenizer(dataset[sent_key].tolist(),truncation=True,padding=True)

            input_ids = sentences.input_ids
            token_type_ids = sentences.token_type_ids
            attention_mask = sentences.attention_mask
            masked_token_idx = []
            
            for input_id in input_ids:
                masked_token_idx.append(input_id.index(4))
            
            return list([input_ids, token_type_ids, attention_mask, labels, masked_token_idx])

        train_inputs = bert_tokenize(dataset_train,"title","topic_idx",tokenizer)
        validation_inputs = bert_tokenize(dataset_val,"title","topic_idx",tokenizer)
        test_inputs = bert_tokenize(test,"title",None,tokenizer)

        for i in range(len(train_inputs)):
            train_inputs[i] = torch.tensor(train_inputs[i])
            
        for i in range(len(validation_inputs)):
            validation_inputs[i] = torch.tensor(validation_inputs[i])
            
        for i in range(len(test_inputs)):
            test_inputs[i] = torch.tensor(test_inputs[i])

        train_data = TensorDataset(*train_inputs)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)

        validation_data = TensorDataset(*validation_inputs)
        validation_sampler = RandomSampler(validation_data)
        validation_dataloader = DataLoader(validation_data,sampler=validation_sampler,batch_size=batch_size)

        test_data = TensorDataset(*test_inputs)
        test_dataloader = DataLoader(test_data,batch_size=batch_size)

        model = AutoModelForPreTraining.from_pretrained(model_checkpoint)
        model.to(device)


        adam_epsilon = 1e-8

        # Number of training epochs (authors recommend between 2 and 4)
        epochs = 3

        num_warmup_steps = 0

        warmup_ratio = 0.1
        num_training_steps = len(train_dataloader)*epochs
        warmup_step = int(num_training_steps * warmup_ratio)

        ### In Transformers, optimizer and schedules are splitted and instantiated like this:
        optimizer = AdamW(model.parameters(), lr=lr,eps=adam_epsilon)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)  # PyTorch scheduler

        train_loss_set = []

        criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        criterion_cls = torch.nn.CrossEntropyLoss()

        model.zero_grad()

        for _ in tnrange(1,epochs+1,desc='Epoch'):
            print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
            batch_loss = 0
            
            # train
            model.train()
            for step, batch in enumerate(tqdm_notebook(train_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_token_type_ids, b_input_mask, b_labels, b_masked_token_idx = batch
                
                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
                
                # calculate loss
                logits_cls, logits_lm = outputs[1], outputs[0]
                
                # topic -> token 
                # Ex.6 -> 3713
                mask_label = [topic_token_dict[lb] for lb in b_labels.to('cpu').numpy() ]
                
                labels_lms = []
                SEQUENCE_LENGTH = 93
                
                # label 만들기
                for idx, label in zip(b_masked_token_idx.to('cpu').numpy(),mask_label):
                    labels_lm = np.full(SEQUENCE_LENGTH, dtype=np.int, fill_value=-1)
                    labels_lm[idx] = label
                    labels_lms.append(labels_lm)
                label_lms_pt = torch.tensor(labels_lms,dtype=torch.int64).to(device)
                
                # lm loss 계산
                loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), label_lms_pt.view(-1))
                
                # cls loss 계산
                labels_cls = [1 for _ in range(len(b_input_ids))]
                labels_cls = torch.tensor(labels_cls).to(device)
                loss_cls = criterion_cls(logits_cls, labels_cls)
                
                loss = loss_cls + loss_lm
                loss.backward()        
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()

                batch_loss += loss.item()
                
            avg_train_loss = batch_loss / len(train_dataloader)
            train_loss_set.append(avg_train_loss)
            print(F'\n\tAverage Training loss: {avg_train_loss}')
            
            # eval
            model.eval()
            
            predict_li = []
            label_li = []
            
            for step, batch in enumerate(tqdm_notebook(validation_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_token_type_ids, b_input_mask, b_labels, b_masked_token_idx = batch

                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)

                logits_cls, logits_lm = outputs[1], outputs[0]
                logits_lm_np = logits_lm.to('cpu').detach().numpy()
                pred = np.argmax(logits_lm_np,axis=2)

                masked_token_idx_np = b_masked_token_idx.to('cpu').numpy()
                
                labels_np = b_labels.to('cpu').numpy()

                for i in range(len(pred)):
                    l = token_topic_dict[pred[i][masked_token_idx_np[i]]]
                    predict_li.append(l)

                for l in labels_np:
                    label_li.append(l)

                mask_label = [ topic_token_dict[lb] for lb in labels_np ]
                
                labels_lms = []
                SEQUENCE_LENGTH = 35

                for idx, label in zip(masked_token_idx_np,mask_label):
                    
                    labels_lm = np.full(SEQUENCE_LENGTH, dtype=np.int, fill_value=-1)
                    labels_lm[idx] = label
                    
                    labels_lms.append(labels_lm)

                labels_lms_pt = torch.tensor(labels_lms,dtype=torch.int64).to(device)
                loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lms_pt.view(-1))

                labels_cls = [1 for _ in range(len(b_input_ids))]
                labels_cls = torch.tensor(labels_cls).to(device)
                loss_cls = criterion_cls(logits_cls, labels_cls)

                loss = loss_cls + loss_lm
                
                batch_loss += loss.item()

            val_accuracy = accuracy_score(predict_li,label_li)
            print("\n\tAccuracy : {}".format(val_accuracy))
            avg_validation_loss = batch_loss / len(validation_dataloader)
            print(F'\n\tAverage validation loss: {avg_validation_loss}')

            if val_accuracy > max_val_accuracy :
                print("="*22 + "save model" + "="*22)
                print("path : mlm_classification_{}_{}_{}".format(batch_size,lr,val_accuracy))
                torch.save(model, "mlm_classification_{}_{}_{}".format(batch_size,lr,val_accuracy))
