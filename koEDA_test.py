import pandas as pd

from koeda import EasyDataAugmentation

EDA = EasyDataAugmentation(
    morpheme_analyzer=None, alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, prob_rd=0.2
)

dataset = pd.read_csv("data/train_data.csv")

def augment_data(dataset_df,EDA,repetition_num):

    augmented_list = []
    label_list = []

    for text, label in zip(dataset_df["title"],dataset_df["topic_idx"]):
        augmenteds = EDA(data=text, p=None, repetition=repetition_num)

        for aug in augmenteds:
            augmented_list.append(aug)
            label_list.append(label)

    new_df = pd.DataFrame({
        'title' : augmented_list,
        'topic_idx' : label_list
    })

    return new_df

aug_df = augment_data(dataset,EDA,4)
print(aug_df.head())


        
