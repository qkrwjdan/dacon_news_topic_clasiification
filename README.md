# **뉴스 토픽 분류 AI 경진대회**

## 월간 데이콘 17 | 자연어 | 분류 | KLUE | Accuracy

---

[https://dacon.io/competitions/official/235747/overview/description](https://dacon.io/competitions/official/235747/overview/description)

## 개요

- ### Why?

    두번째로 데이콘 대회에 도전하면서 코드들을 정리해 놓으면 편하겠구나 라고 생각하여 코드를 정리했습니다. 

- ### Grade

    public : 17th, private : 17th. 상위 7%

- ### Data

    Download : [https://dacon.io/competitions/official/235747/data](https://dacon.io/competitions/official/235747/data)

    data폴더에 저장하였습니다.

- ### Requirements

    ```python
    torch==1.7.1
    transformers==4.8.2
    optuna==2.8.0
    ray[tune]==1.5.0

    datasets
    pandas
    sklearn

    koeda
    ...
    ```

- ### Code
1. **Klue/bert-base fine-tuning**

    hugginface에서 Klue/bert-base 모델을 받아 해당 데이터셋에 맞게 fine-tuning하는 과정입니다. 

2. **Klue/bert-base fine-tuning with Trainer API**

    Trainer API를 사용하여 더욱 쉽게 모델을 fine-tuning할 수 있습니다.

3. **Klue/bert-base hyperparameter search with Trainer API**

    Trainer API에서는 Hyperparameter를 찾는 기능을 제공합니다. 최적의 성능을 보이는 Hyper parameter를 찾을 수 있습니다. 

4. **Load checkpoints** ❌

    학습된 모델을 불러와 다시 사용할 수 있습니다. 

5. **lue/bert-base MLM Classification**

    BERT의 Pretrain에 사용된 방법인 MLM을 사용하여 Classification을 수행합니다. 

6. **Klue/bert-base fine-tuning using augmented data**

    Augmented data를 사용하여  모델을 학습합니다.

7. **Data augmentation - 1**

    함수를 직접 정의하여 RD, RS augmentation을 수행합니다.

8. **Data augmentation - 2**

    Koeda 라이브러리를 사용하여 쉽게 한국어 데이터를 augmentation할 수 있습니다. 

9. **Cross Validation**

    교차검증학습을 수행합니다. 

10. **ensemble (Soft, Hard)** ❌

    여러 모델을 합치는 ensemble을 사용합니다.
