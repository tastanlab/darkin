# DARKIN
Hello! This is the official repository for the DARKIN sataset which we present in our paper "DARKIN: A zero-shot classification benchmark and an evaluation of protein language models". 

DARKIN is dataset created for the Zero-Shot Learning setup, which you could create different versions of in a reproducible manner by playing around with the available parameters. This document will walk you through on how you could generate the DARKIN dataset in you local envionment and other details on the DARKIN dataset. For more detail please refer to our article. This repo will go over these topics:
1. How to generate a DARKIN split in your local environment
 - How to generate the dataset statistics
3. Details of the implementation (skippable if you are only interested in using the dataset)
 - Example dataset statictics of the main DARKIN split we used in our paper

## 1. Generate a DARKIN split

### 1.1. Installation & Setting up the Environment

As the first step you have to download this repository to your local environment. You can either download it az a zip file or just clone the repository like this:

```
git clone git@github.com:aysesunar/darkin.git
```

Now you have to create a conda environment to be able to run the code. Create the conda environment like this:

```
conda create --name darkin python=3.11.3
```

then activate this conda environment:

```
conda activate darkin
```

install pip if it is not installed:

```
conda install pip
```

now install the required packages. You could either use the requirements.txt file like this:

```
pip install -r requirements.txt
```

or you could directly install the required packges like this:

```
pip install pandas
pip install numpy
pip install matplotlib
```

now you should be all set to run the code!

### 1.2. Running the Code

Now you are ready to run the code and create the DARKIN dataset in your local environment. To run the code, you have to run the create_darkin_split.py file like this:

```
python create_darkin_split.py
```

(I will add some informatoin on the strategy that is used in DARKIN before getting to the parameters.)

| Parameter | Description |
|:---------|:---------|
| RANDOM_SEED | The random seed which is set at the beggining fo the script, to ensure the same split on different runs of the script. This variable could also be used to create different splits to see the performance on different splits of the data. In our paper we have used random seeds 0, 42, 87 and 12345. We have used random seed 12345 as our default split. |
| KINASE_SIMILARITY_PERCENT | The identity similarity score percentage of the kinase domains that will be taken into consideration when splitting the dataset. (Kinase domains which have similarity equal to or above this percentage will be placed inside the same set, so they will all be added to either the train, validation or test set) |
| KINASE_COUNT_TEST_THRESHOLD | This is the number of phosphosite-kinase association threshold for kinases to be able to enter the test dataset. Kinases which have fewer phosphosite-kinase associations than this threshold will not be considered to be placed into the test set. |
| STRATIFY_PERCENTAGE_FOR_UNSEEN_TEST_KINASE | The percentage of the dataset size that should be entered into the test set as unseen data. |
| TEST_GZSL_PERCENTAGE | The percentage of how much of a seen kinase’s data will be placed into the test dataset. After a kinase is decided to be a seen kinase, this much percentage of it’s data will be placed into test, and the rest will be inside train. |
| KINASE_COUNT_VALIDATION_THRESHOLD | This is the number of phosphosite-kinase association threshold for kinases to be able to enter the validation dataset. Kinases which have fewer phosphosite-kinase associations than this threshold will not be considered to be placed into the test set. |
| STRATIFY_PERCENTAGE_FOR_UNSEEN_VALIDATION_KINASE | The percentage of the dataset size that should be entered into the validation set as unseen data. |
| VALIDATION_GZSL_PERCENTAGE | The percentage of how much of a seen kinase’s data will be placed into the validation dataset (here seen means in the context of train-validation split). After a kinase is decided to be a seen kinase, this much percentage of it’s data will be placed into validation, and the rest will be inside validation. |
| INCLUDE_VALIDATION | Whether to perform the train-validation split as well. If selected True, then the script will first perform train-test split, and then will divide train into train-validation. |
| TAKE_SEQUENCE_SIMILARITY_INTO_CONSIDERATION | This parameter defines whether to take kinase domain sequence similarity into consideration when splitting the datasets. If selected True, kinases who have sequence similarity equal to or above the KINASE_SIMILARITY_RATE will be placed into the same dataset.  |
| DIVIDE_WRT_GROUP | Defines whether to stratify the kinases with respect to the kinase groups. If set to False, the dataset will be split without taking the kinase group information into account, thus datasets might have imbalanced kinase groups. |


