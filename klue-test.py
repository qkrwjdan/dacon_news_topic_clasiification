import random
import logging
from IPython.display import display, HTML

import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    print(torch.cuda.device_count())
else:
    print("no cuda!")

batch_size = 32
task = "nli"
RANDOM_SEED = 17
num_labels = 7

model_info = [["klue/roberta-large", "models/klue-roberta-large.pth", "results/klue-roberta-large.csv"],
    ["klue/roberta-base","models/klue-roberta-base.pth", "results/klue-roberta-base.csv"],
    ["klue/bert-base","models/klue-bert-base.pth", "results/klue-bert-base.csv"],
    ["klue/roberta-small","models/klue-roberta-small.pth", "results/klue-roberta-small.csv"]]

for model_checkpoint, model_path, result_path in model_info: 

    print("model checkpoint :",model_checkpoint)
    print("model path :",model_path)
    print("result path :",result_path)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    dataset = pd.read_csv("data/train_data.csv",index_col=False)
    test = pd.read_csv("data/test_data.csv",index_col=False)

    dataset_train, dataset_val = train_test_split(dataset,test_size = 0.2,random_state = RANDOM_SEED)

    def preprocess_function(examples):
        return tokenizer(examples,truncation=True,return_token_type_ids=False)

    class BERTDataset(Dataset):
        def __init__(self, dataset, sent_key, label_key, bert_tokenizer):
            
            self.sentences = [ bert_tokenizer(i,truncation=True,return_token_type_ids=False) for i in dataset[sent_key] ]
            
            if not label_key == None:
                self.mode = "train"
            else:
                self.mode = "test"
                
            if self.mode == "train":
                self.labels = [np.int64(i) for i in dataset[label_key]]
            else:
                self.labels = [np.int64(0) for i in dataset[sent_key]]

        def __getitem__(self, i):
            if self.mode == "train":
                self.sentences[i]["label"] = self.labels[i]
                return self.sentences[i]
            else:
                return self.sentences[i]

        def __len__(self):
            return (len(self.labels))

    data_train = BERTDataset(dataset_train, "title", "topic_idx", tokenizer)
    data_val = BERTDataset(dataset_val, "title", "topic_idx", tokenizer)
    data_test = BERTDataset(test, "title", None, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    metric = load_metric("glue", "qnli")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    metric_name = "accuracy"

    args = TrainingArguments(
        "test-nli",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=data_train,
        eval_dataset=data_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

    pred = trainer.predict(data_test)
    pred = pred[0]
    pred = np.argmax(pred,1)

    submission = pd.read_csv('data/sample_submission.csv')
    submission['topic_idx'] = pred
    submission.to_csv(result_path,index=False)