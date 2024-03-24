# DARKIN
Hello! This is the official repository for the DARKIN dataset which we present in our paper "DARKIN: A zero-shot classification benchmark and an evaluation of protein language models". 

DARKIN is dataset created for the Zero-Shot Learning setup, which you could create different versions of in a reproducible manner by playing around with the available parameters. This document will walk you through on how you could generate the DARKIN dataset in you local envionment and other details on the DARKIN dataset. For more detail please refer to our article. This repo will go over these topics:
1. DARKIN and its Implementation Strategy
2. How to generate a DARKIN split in your local environment
(In progress) 3. Using the Dataset Statistics script to get insight of the created DARKIN split
(In progress) 4. Example dataset statictics of the main DARKIN split we used in our paper

## 1. DARKIN and its Implementation Strategy

DARKIN is a dataset consisting of phosphorylation data, mainly it contains rows of phosphosites and the kinases which phosphorylate these specific phosphosites. So the phosphosites are the inputs, and the kinases are the labels. Also since a phosphosite could be phosphorylated by several kinases, it is not unusual to see multiple kinases next to a single phosphosite in the dataset. DARKIN is created for the Zero-Sot learning setup, thus the kinases present in the train, validation and test sets are all disjoint. Here is a sample snapshot of a portion of the dataset:

| SUB_ACC_ID | SUB_MOD_RSD | SITE_+/-7_AA | KINASE_ACC_IDS |
|:---------|:---------|:---------|:---------|
|P68101|S52|MILLSELsRRRIRSI|Q9BQI3|
|P83268|S51|RILLsELsR______|Q9BQI3|
|P05198|S52|MILLsELsRRRIRsI|P28482,Q7KZI7,Q9BQI3,Q9NZJ5|
|P05198|S49|IEGMILLsELsRRRI|Q9BQI3,Q9NZJ5|

There are several startegies utilized when creating the dataset splits:

- **Number of phosphosites per kinase**: To ensure robust evaluation in the test and validation sets, we set a minimum threshold for the number of phosphosite-kinase pair associations a kinase should have in order to be defined as a test or validation kinase. This is to make sure that the scores obtained for a specific kinase class in test or validation do not rely on very few data, since results obtained on very few data points could be misleading. 
- **Stratification based on kinase groups**: Data points are stratified into train, validation and test sets based on kinase groups. This is to ensure that every kinase group is represented in each set whenever feasible.  
- **Sequence similarity of kinases**: To prevent optimistic performance estimates, kinases with sequence similarity above a paramterized threshold  are grouped and assigned to the same sets (train, validation, or test).

Here is the high level illustration of the steps of our dataset splitting strategy implementation:

<p align="center">
  <img src="/images/high_level_dataset_split_script_algorithm_flow-ZSL-GitHub.drawio.png" alt="High Level Steps of the Dataset Splitting Script" width="20%"/>
</p>

## 2. How to generate a DARKIN split in your local environment

### 2.1. Installation & Setting up the Environment

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

### 2.2. Running the Code

Now you are ready to run the code and create the DARKIN dataset in your local environment. To run the code, you have to run the create_darkin_split.py file like this:

```
python create_darkin_split.py
```

There are several parameters which you could play aorund with, according to your specific interests (Please refer to section 1 is these parameters are confusing to you):

| Parameter | Description |
|:---------|:---------|
| <sub>RANDOM_SEED</sub> | <sub>The random seed which is set at the beggining fo the script, to ensure the same split on different runs of the script. This variable could also be used to create different splits to see the performance on different splits of the data. In our paper we have used random seeds 0, 42, 87 and 12345. We have used random seed 12345 as our default split.</sub> |
| <sub>KINASE_SIMILARITY_PERCENT</sub> | <sub>The identity similarity score percentage of the kinase domains that will be taken into consideration when splitting the dataset. (Kinase domains which have similarity equal to or above this percentage will be placed inside the same set, so they will all be added to either the train, validation or test set)</sub> |
| <sub>KINASE_COUNT_TEST_THRESHOLD</sub> | <sub>This is the number of phosphosite-kinase association threshold for kinases to be able to enter the test dataset. Kinases which have fewer phosphosite-kinase associations than this threshold will not be considered to be placed into the test set.</sub> |
| <sub>STRATIFY_PERCENTAGE_FOR_UNSEEN_TEST_KINASE</sub>| <sub>The percentage of the dataset size that should be entered into the test set as unseen data.</sub> |
| <sub>TEST_GZSL_PERCENTAGE</sub> | <sub>The percentage of how much of a seen kinase’s data will be placed into the test dataset. After a kinase is decided to be a seen kinase, this much percentage of it’s data will be placed into test, and the rest will be inside train.</sub> |
| <sub>KINASE_COUNT_VALIDATION_THRESHOLD</sub> | <sub>This is the number of phosphosite-kinase association threshold for kinases to be able to enter the validation dataset. Kinases which have fewer phosphosite-kinase associations than this threshold will not be considered to be placed into the test set.</sub> |
| <sub>STRATIFY_PERCENTAGE_FOR_UNSEEN_VALIDATION_KINASE</sub> | <sub>The percentage of the dataset size that should be entered into the validation set as unseen data.</sub> |
| <sub>VALIDATION_GZSL_PERCENTAGE</sub> | <sub>The percentage of how much of a seen kinase’s data will be placed into the validation dataset (here seen means in the context of train-validation split). After a kinase is decided to be a seen kinase, this much percentage of it’s data will be placed into validation, and the rest will be inside validation.</sub> |
| <sub>INCLUDE_VALIDATION | <sub>Whether to perform the train-validation split as well. If selected True, then the script will first perform train-test split, and then will divide train into train-validation.</sub> |
| <sub>TAKE_SEQUENCE_SIMILARITY_INTO_CONSIDERATION</sub> | <sub>This parameter defines whether to take kinase domain sequence similarity into consideration when splitting the datasets. If selected True, kinases who have sequence similarity equal to or above the KINASE_SIMILARITY_RATE will be placed into the same dataset. </sub> |
| <sub>DIVIDE_WRT_GROUP</sub> | <sub>Defines whether to stratify the kinases with respect to the kinase groups. If set to False, the dataset will be split without taking the kinase group information into account, thus datasets might have imbalanced kinase groups.</sub> |

These parameters could directly modified in the create_darkin_split.py file directly. Or it the parameters could be set when wunning create_darkin_split.py like this:

```
python create_darkin_split.py --RANDOM_SEED 12
```

## 3. Dataset Statistics

In order to provide better insight in the dataset split that has been generated, the file dataset_statistics.py is implemented. Several different dataset statistucs could be found here. The plots that could be generate are listed as follows:

1. **Kinase Count**: This plot shows the number of kinases in each set (train, validation adn test).
2. **Phosphosite Count**: This plot shows the number of phosphosites in each set (train, validation adn test). 
3. **Phosphosite-Kinase Count**: This plot shows the number of phosphosite-kinase association data points in each set (train, validation adn test).
4. **Phosphosite-Kinase Count Histogram of Kinases**: This histogram shows how many phosphosite-kinase count is associated with how many kinases in the train, validation and test sets. 
5. **Multilabel Phosphosites vs Single Kiase Phosphosites**: A single phosphosite could be phosphorylated by several kinases. A phosphosite which is associated with several kinases is named as multilabel phosphosite/site, and a phosphosite which is phosphorylated by a single kinase is named as single kinase phosphosite. This plot illustrates the number of multilabel and single kinase phosphosites in the train, validation and test set.
6. **Novel Site vs Common Site in Test**: A phosphosite could be phosphorylated by several kinases, thus a phosphosite could appear in different sets. We call sites which only appear in a single set such as train, validation or test set as novel sites. Likewise we call phosphosites which appear in several sets as common sites between those sets. In this plot we report the number of novel sites in the test set, and the number of common sites with the train and validation sets.
7. **Novel Site vs Common Site in Each Set**: In this plot we show the number of novel and common sites for each set (First row). Furthermore we also show the number of phosphosite-kinase data point association corresponding to these kinases (Second row).
8. **Phosphosite Kinase Association Histogram of Kinases**: In this plot, we show the number of phosphosites associated with the number of kinases specified in the x label. 
9. **Kinase Group Distribution**: This plot illustrates the number of kinases from each group for that specific set (First row). Furthermore it also illustrates the number of phosphosite-kinase association data points these kinases correspond to in that specific set (second row).
**Several more could be added...**:

### 3.1. How to Generate these Dataset Statistics

After generating the DARKIN split, the dataset statistics could be generated by running the dataset_statistics.py file (The function call lines at the bottom of the script should be made uncommented). Another option to generate the dataset statistics is by using the dataset_statistics.ipynb notebook file. Here are the steps to follow in order to run this notebook if you are running on a remote server:

First install jupyter notebook:

```
pip install jupyter
```

Run this line in your remote server to open jupyter notebook:

```
jupyter notebook --no-browser --port=8888
```

Then open SSH tunnel from your local PC like this:
```
ssh -v -N -L 8888:localhost:8888 your_username@your_server_ip
```

Then you could access jupyter notebbok from http://localhost:8888/ on your browser. On your first login it might ask you for credentials, in this case follow the steps mentioned in the remote server side (a token id will be provided like this: http://localhost:8888/?token=token_id). Now you are ready to run the provided dataset_statistics.ipynb file. 

### 3.2. Dataset Statistics of our Default DARKIN Split

<p align="left">
  <img src="/dataset_statistics/kinase_count_in_each_set.png" alt="High Level Steps of the Dataset Splitting Script" width="150px"/>
  <img src="/dataset_statistics/phosphosite_count_in_each_set.png" alt="High Level Steps of the Dataset Splitting Script" width="150px"/>
  <img src="/dataset_statistics/phosphosite_kinase_pair_distribution.png" alt="High Level Steps of the Dataset Splitting Script" width="150px"/>
</p>

<p align="left">
  <img src="/dataset_statistics/Kinase_Occurrenes_Histogram.png" alt="High Level Steps of the Dataset Splitting Script" width="300px"/>
</p>

<p align="left">
  <img src="/dataset_statistics/Multilabel_vs_NonMultilabel_Rows_Distribution.png" alt="High Level Steps of the Dataset Splitting Script" width="300px"/>
  <img src="/dataset_statistics/Novel_Sites_vs_Common_Sites_in_test.png" alt="High Level Steps of the Dataset Splitting Script" width="300px"/>
</p>

<p align="left">
  <img src="/dataset_statistics/Novel_Sites_vs_Common_Sites.png" alt="High Level Steps of the Dataset Splitting Script" width="300px"/>
  <img src="/dataset_statistics/Multilabel_Rows_Histogram.png" alt="High Level Steps of the Dataset Splitting Script" width="400px"/>
  <img src="/dataset_statistics/Kinase_Group_Distribution_and_Site_Count.png" alt="High Level Steps of the Dataset Splitting Script" width="500px"/>
</p>









