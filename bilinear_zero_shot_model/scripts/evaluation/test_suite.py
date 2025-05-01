import sys
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Get the directory of the current file (test_suite.py)
CURR_DIRECTORY = os.path.dirname(__file__)

# Get the parent directory of `current_file_directory` (which is `scripts/evaluation`),
# then get the parent of that directory (which should be the root of your project),
# which contains the `scripts` directory.
ROOT_DIRECTORY = os.path.dirname(os.path.dirname(CURR_DIRECTORY))
SIMILARITY_FILE = f"dataset/new_dataset/kinase_pairwise_identity_similarity_scores.csv"

# Add the project root directory to sys.path
if ROOT_DIRECTORY not in sys.path:
    sys.path.append(ROOT_DIRECTORY)

import json
import pandas as pd
from matplotlib import pyplot as plt
import csv
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
from scripts.utils.arguments import set_saved_checkpoint_filename
# from scripts.utils.arguments import load_config
from scripts.utils.arguments import load_config
import torch
from datetime import datetime
import copy
import matplotlib.cm as cm

'''
Here I will write the test suite for DKZ

Possible test ideas:
+ 1. (Not test just informative) Show how many kinases there are inside the train/validation (Group Distribution Plot)
+ 2. Kinase Distribution Plot: show how many kinases in train validation or test
+ 3.5. Precision and Recall curves for each kinase
- 3. Precision Recall curve for overall data (class based)!!! All pr curves dont have the same thresholds so i couldnt do this yet
- 3. Precision Recall curve group based!!! same, PR curves dont have same thresholds so couldnt do this
+ 5. Group-wise aupr scores (We can directly get the aupr score average of the kinases in that specific group)
+ 7. Box plot of the group-wise kinase scores. I mean its gonna be the kinases class based scores but it is gonna be group wise boxes.
+ 9. Show the performance of the kinases in the scatter plot (For example the more errored places should be red, the better places could be black)
+ 11. Kinazlarin başari histogrami
+ 12. Testte olan kinazlarin heterogenesity? Ne kadar cok datasi var bunlarin
13. W'yu da ekle??
'''

def read_file(file_path):
    return pd.read_csv(f'{ROOT_DIRECTORY}/{file_path}')


def map_family_group_info(config):
    df = read_file(config['kinase']['dataset']['train'])

    uniprots_to_family = {}
    uniprots_to_group = {}
    for index, row in df.iterrows():
        kinase = row["Kinase"]
        family = row["Family"] if not pd.isna(row["Family"]) else "missing family"
        group = row["Group"] if not pd.isna(row["Group"]) else "missing group"
        uniprots_to_family[kinase] = family
        uniprots_to_group[kinase] = group
    return uniprots_to_family, uniprots_to_group

def get_kinase_group_distributions(config, all_groups, uniprots_to_groups):
    df_train = read_file(config['phosphosite']['dataset']['train'])
    df_valid = read_file(config['phosphosite']['dataset']['validation'])
    df_test = read_file(config['phosphosite']['dataset']['test'])

    # So even though we will either run train (train vs valid) or test (train vs test)
    # I think it is still useful to plot all three datasets for whichever case.

    group_to_kinase_set_train = {group: set() for group in all_groups}
    group_to_data_count_train = {group: 0 for group in all_groups}
    group_to_kinase_set_valid = {group: set() for group in all_groups}
    group_to_data_count_valid = {group: 0 for group in all_groups}
    group_to_kinase_set_test = {group: set() for group in all_groups}
    group_to_data_count_test = {group: 0 for group in all_groups}

    missing = set()
    for index, row in df_train.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            group = uniprots_to_groups[kinase]
            if group == "missing":
                missing.add(kinase)
            group_to_kinase_set_train[group].add(kinase)
            group_to_data_count_train[group] += 1
    
    print(f'kinase with missing group : {missing}')

    for index, row in df_valid.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        groups_of_kinases = set()
        for kinase in kinases:
            group = uniprots_to_groups[kinase]
            group_to_kinase_set_valid[group].add(kinase)
            groups_of_kinases.add(group)

        # So here, normally i was going to count this site for every kinase inside this row, however
        # Gokberk hoca said that we shouldn't count this row multiple times. since it is going to be
        # passed once. However if I am going to count this data once, then which group am I going to
        # add this to? Since there could be multiple kinases, each belonging to a different group. So
        # what I did is I put all of the groups of the kinases inside a set, and for each existing group
        # I counted this test row.
        for group in groups_of_kinases:
            group_to_data_count_valid[group] += 1

    for index, row in df_test.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        groups_of_kinases = set()
        for kinase in kinases:
            group = uniprots_to_groups[kinase]
            group_to_kinase_set_test[group].add(kinase)
            groups_of_kinases.add(group)

        # So here, normally i was going to count this site for every kinase inside this row, however
        # Gokberk hoca said that we shouldn't count this row multiple times. since it is going to be
        # passed once. However if I am going to count this data once, then which group am I going to
        # add this to? Since there could be multiple kinases, each belonging to a different group. So
        # what I did is I put all of the groups of the kinases inside a set, and for each existing group
        # I counted this test row.
        for group in groups_of_kinases:
            group_to_data_count_test[group] += 1

    return group_to_kinase_set_train, group_to_data_count_train, group_to_kinase_set_valid, group_to_data_count_valid, group_to_kinase_set_test, group_to_data_count_test

def get_kinase_distibution(config, save_filepath):
    all_train_kinases, all_valid_kinases, all_test_kinases = get_kinase_counts_in_datasets_new_data(config)

    sets = ["Train set", "Validation set", "Test set"]
    values = [len(all_train_kinases), len(all_valid_kinases), len(all_test_kinases)]

    fig, ax = plt.subplots()  # Create figure and axis objects
    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    plt.bar(sets, values, color=grayish_color, linewidth=0.5)
    plt.xlabel('Datasets', fontweight='bold')
    plt.ylabel('Size', fontweight='bold')
    plt.title('Dataset Size Comparisons')
    ax.set_axisbelow(True)
    ax.grid(axis='y', zorder=0)

    # Remove the line boundaries
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', which='both', length=0)
    for i, count in enumerate(values):
        plt.text(i, count + 0.5, str(count), ha='center')

    fig.savefig(f"{save_filepath}/kinase_distribution.png")


def get_kinase_counts_in_datasets_new_data(config):
    df_train = read_file(config['phosphosite']['dataset']['train'])
    df_valid = read_file(config['phosphosite']['dataset']['validation'])
    df_test = read_file(config['phosphosite']['dataset']['test'])

    all_train_kinases = set()
    for index, row in df_train.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        all_train_kinases = all_train_kinases.union(kinases)
    all_valid_kinases = set()
    for index, row in df_valid.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        all_valid_kinases = all_valid_kinases.union(kinases)
    all_test_kinases = set()
    for index, row in df_test.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        all_test_kinases = all_test_kinases.union(kinases)
    return all_train_kinases, all_valid_kinases, all_test_kinases

def get_kinase_counts_in_datasets_old_data(args):
    train_file = args.TRAIN_DATA
    valid_file = args.VAL_DATA
    test_file = args.TEST_DATA

    all_train_kinases = set()
    with open(train_file, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            kinase = columns[3]
            all_train_kinases.add(kinase)

    all_valid_kinases = set()
    with open(valid_file, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            kinases = set(columns[3].split(','))
            all_valid_kinases = all_valid_kinases.union(kinases)

    all_test_kinases = set()
    with open(test_file, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            kinases = set(columns[3].split(','))
            all_test_kinases = all_test_kinases.union(kinases)

    return all_train_kinases, all_valid_kinases, all_test_kinases

# def create_folder(config):
#     model_name = config['phosphosite']['model']['model_type']
#     # embedding_mode = config['phosphosite']['model']['embedding_mode']
#     checkpoint_filename = set_saved_checkpoint_filename(config)
# 
#     save_filepath = os.path.join(ROOT_DIRECTORY,
#                                 'test_suite_results',
#                                  model_name,
#                                  # embedding_mode,
#                                  checkpoint_filename["logging"]["local"]["checkpoint_file_name"])
# 
#     count = 0
#     while os.path.exists(os.path.join(f'{save_filepath}', f'{count}')):
#         count += 1
# 
#     save_filepath = os.path.join(f'{save_filepath}', f'{count}')
#     os.makedirs(save_filepath)
# 
#     save_filepath = os.path.join(f'{save_filepath}')
#     return save_filepath

def create_folder(config):
    model_name = config['phosphosite']['model']['model_type']
    # embedding_mode = config['phosphosite']['model']['embedding_mode']
    config = set_saved_checkpoint_filename(config)
    checkpoint_filename = config["logging"]["local"]["checkpoint_file_name"]

    save_filepath = os.path.join(ROOT_DIRECTORY,
                                'test_suite_results',
                                 model_name,
                                 # embedding_mode,
                                 checkpoint_filename)

    # count = 0
    # while os.path.exists(os.path.join(f'{save_filepath}', f'{count}')):
    #     count += 1

    job_id = os.getenv('SLURM_JOB_ID', 'Unknown Job ID')

    save_filepath = os.path.join(f'{save_filepath}', f'{job_id}')
    os.makedirs(save_filepath, exist_ok=True)

    save_filepath = os.path.join(f'{save_filepath}')
    return save_filepath

def write_arguments_to_txt_file(config, save_filepath):
    output_file = f'{save_filepath}/config.txt'
    
    config_dict = config if isinstance(config, dict) else vars(config)
    config_dict['current_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    job_id = os.getenv('SLURM_JOB_ID', 'Unknown Job ID')  # Adjust this if your system uses a different variable
    config_dict['job_id'] = job_id

    with open(output_file, 'w') as file:
        file.write(json.dumps(config_dict, indent=4, sort_keys=True))

# def calculate_kinase_scores(args, test_probabilities, test_data_true_labels, all_test_kinase_uniprotIDs):
def calculate_kinase_scores(args, metric):
    # test_probabilities.shape[0]: the number of rows in test
    # test_probabilities.shape[1]: the number of kinases in test (the number of kinases the predictions are made for)
    class_aupr_scores, aupr_per_group, class_precision_scores, class_reall_scores = metric.calculate_aupr_per_kinase_class_and_group()
    # kinase_info_dict = {kinase : dict() for kinase in all_test_kinase_uniprotIDs}
    # for class_index in range(test_probabilities.shape[1]):
    # for class_index in range(len(test_probabilities[0])):
    kinase_info_dict = dict()
    for kinase_uniprotID, _ in class_aupr_scores.items():
        # Store precision and recall values in the dictionary
        kinase_info_dict[kinase_uniprotID] = dict()
        kinase_info_dict[kinase_uniprotID]["aupr"] = class_aupr_scores[kinase_uniprotID]
        kinase_info_dict[kinase_uniprotID]["precision"] = class_precision_scores[kinase_uniprotID]
        kinase_info_dict[kinase_uniprotID]["recall"] = class_reall_scores[kinase_uniprotID]

    # Now, kinase_scores contains precision, recall and aupr values for each kinase
    return kinase_info_dict

def plot_precision_recall_curves(config, save_filepath, kinase_info_dict):
    # test_probabilities.shape[0]: the number of rows in test
    # test_probabilities.shape[1]: the number of kinases in test (the number of kinases the predictions are made for)
    for kinase_uniprotID, _ in kinase_info_dict.items():
        # Store precision and recall values in the dictionary
        precision = kinase_info_dict[kinase_uniprotID]["precision"]
        recall = kinase_info_dict[kinase_uniprotID]["recall"]

        precision_recall_curve_plots_path = os.path.join(f'{save_filepath}', 'precision_recall_curves')
        if not os.path.exists(precision_recall_curve_plots_path):
            os.makedirs(precision_recall_curve_plots_path)

        grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
        plt.figure()
        plt.step(recall, precision, where='post', color=grayish_color)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for Kinase {kinase_uniprotID}')
        plt.savefig(f'{precision_recall_curve_plots_path}/{kinase_uniprotID}.png')
        plt.close()

def plot_group_based_box_plots(config, save_filepath, kinase_info_dict):
    uniprots_to_family, uniprots_to_groups = map_family_group_info(config)
    kinase_names = []
    aupr_scores = []
    group_scores = {}
    for k, kinase_dict in kinase_info_dict.items():
        auprc = kinase_dict["aupr"] 
        kinase_names.append(k)
        aupr_scores.append([auprc])

        group = uniprots_to_groups[k]
        if group not in group_scores:
            group_scores[group] = []
        group_scores[group].append(auprc)

    print(group_scores)

    # Create a box plot
    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    plt.figure(figsize=(10, 6))
    boxprops = dict(color=grayish_color)
    plt.boxplot(aupr_scores, labels=kinase_names, boxprops=boxprops)
    plt.title('AUPR Scores for Different Kinases')
    plt.xlabel('Kinases')
    plt.ylabel('AUPR Score')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{save_filepath}/box_plot_individual_kinases.png')

    # Create a list of labels for each group
    group_labels = list(group_scores.keys())
    group_labels.sort()

    # Create a list of AUPR score lists for each group
    group_data = [group_scores[group] for group in group_labels]    

    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    plt.figure(figsize=(10, 6))
    boxprops = dict(color=grayish_color)
    plt.boxplot(group_data, labels=group_labels, boxprops=boxprops)
    plt.title('Box Plot of Group Based AUPR')
    plt.xlabel('Kinases')
    plt.ylabel('AUPR Score')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{save_filepath}/group_based_box_plot.png')

    return kinase_info_dict

def plot_kinase_aupr_score_histogram(config, save_filepath, kinase_info_dict):
    kinase_names, kinase_aupr_scores = [], []
    for k, v in kinase_info_dict.items():
        kinase_names.append(k)
        kinase_aupr_scores.append(float(v["aupr"])*100) # I am multiplying by 100 since the values are in decimal format
    
    # Define custom bin edges
    bin_edges = np.arange(0, 101, 5)

    # Create a histogram
    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    plt.figure(figsize=(10, 6))
    plt.hist(kinase_aupr_scores, bins=bin_edges, color=grayish_color, zorder=3)
    plt.title('Histogram of AUPR Scores for Kinases')
    plt.xlabel('AUPR Score Ranges')
    plt.ylabel('Number of Kinases')

    # Set the zorder of the grid lines to 0 to make them appear behind the bars
    plt.grid(axis='y', linestyle='-', zorder=0)  # Set linestyle to '-' for solid lines and set zorder for the grid lines
    plt.grid(axis='x', linestyle='None')

    # Adjust border visibility
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', which='both', left=False)
    ax.grid(zorder=0)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{save_filepath}/Kinase_AUPR_Score_Histogram.png')

def get_kinase_count_in_test_data(config):
    test_file = config['phosphosite']['dataset']['test'] if config['mode'] == "test" else config['phosphosite']['dataset']['validation']
    df_test = read_file(test_file)
    kinase_to_count = dict()
    for index, row in df_test.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if kinase not in kinase_to_count:
                kinase_to_count[kinase] = 0
            kinase_to_count[kinase] += 1
    return kinase_to_count

# So in terms of looking at the heterogeneity, this will work only for ZSL, since for GZSL,
# the kinases may also appear in the train dataset
def plot_test_kinase_heterogenity(config, save_filepath):
    # count the number of rows these kinase exist inside the test dataset. 
    kinase_to_count = dict()
    kinase_to_count = get_kinase_count_in_test_data(config)

    # Find the maximum count in kinase_to_count
    max_count = max(kinase_to_count.values())
    bin_edges = np.arange(0, max_count + 2, 5) 

    # Define custom bin edges
    # bin_edges = np.arange(0, 201, 5)

    # Create a histogram
    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    plt.figure(figsize=(10, 6))
    plt.hist(kinase_to_count.values(), bins=bin_edges, color=grayish_color, zorder=3)
    plt.title('Histogram of Site counts of Kinases (To measure Heterogenity)')
    plt.xlabel('Site Counts of Kinases')
    plt.ylabel('Number of Kinases')

    # Set the zorder of the grid lines to 0 to make them appear behind the bars
    plt.grid(axis='y', linestyle='-', zorder=0)  # Set linestyle to '-' for solid lines and set zorder for the grid lines
    plt.grid(axis='x', linestyle='None')

    # Adjust border visibility
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', which='both', left=False)
    ax.grid(zorder=0)

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{save_filepath}/Kinase_Site_Count_Heterogenity.png')

# So in terms of looking at the heterogeneity, this will work only for ZSL, since for GZSL,
# the kinases may also appear in the train dataset
def plot_group_heterogenity(config, save_filepath):
    # count the number of rows these kinase exist inside the test dataset. 
    kinase_to_count = dict()
    kinase_to_count = get_kinase_count_in_test_data(config)

    uniprots_to_family, uniprots_to_groups = map_family_group_info(config)
    group_to_hetrogenity = dict()
    for k, v in kinase_to_count.items():
        kinase_group = uniprots_to_groups[k]
        if kinase_group not in group_to_hetrogenity:
            group_to_hetrogenity[kinase_group] = []
        group_to_hetrogenity[kinase_group].append(v)

    # Define custom bin edges
    max_count = max(kinase_to_count.values())
    bin_edges = np.arange(0, max_count + 2, 5) 
    # bin_edges = np.arange(0, 201, 5)

    group_based_heterogenity_folder = f'{save_filepath}/Group_Based_Heterogenity'
    if not os.path.exists(group_based_heterogenity_folder):
        os.makedirs(group_based_heterogenity_folder)

    for group in group_to_hetrogenity:
        grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
        plt.figure(figsize=(10, 6))
        plt.hist(group_to_hetrogenity[group], bins=bin_edges, color=grayish_color, zorder=3)
        plt.title('Histogram of Site counts of Kinases (To measure Heterogenity)')
        plt.xlabel('Site Counts of Kinases')
        plt.ylabel('Number of Kinases')

        # Set the zorder of the grid lines to 0 to make them appear behind the bars
        plt.grid(axis='y', linestyle='-', zorder=0)  # Set linestyle to '-' for solid lines and set zorder for the grid lines
        plt.grid(axis='x', linestyle='None')

        # Adjust border visibility
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False)
        ax.grid(zorder=0)

        # Show the plot
        plt.tight_layout()
        plt.savefig(f'{group_based_heterogenity_folder}/{group}_hetrogenity.png')

def write_down_kinase_scores(args, save_filepath, kinase_info_dict):
    uniprots_to_family, uniprots_to_groups = map_family_group_info(args)
    aupr_scores = []
    group_scores = {}
    
    for kinase, kinase_score_dict in kinase_info_dict.items():
        kinase_group = uniprots_to_groups[kinase]
        aupr_scores.append(kinase_score_dict["aupr"])
        
        if kinase_group not in group_scores:
            group_scores[kinase_group] = {}
        group_scores[kinase_group][kinase] = kinase_score_dict["aupr"]
    
    kinase_aupr_score_csv = f'{save_filepath}/AUPR_Scores_Report.csv'
    with open(kinase_aupr_score_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write overall AUPR score
        writer.writerow(["Overall Macro aupr score:", sum(aupr_scores) / len(aupr_scores)])
        writer.writerow([])  # Empty row for separation
        
        # Write class-based AUPR scores
        for group, kinase_scores in group_scores.items():
            writer.writerow([f"{group} Aupr Score", sum(kinase_scores.values()) / len(kinase_scores)])
            
            # Sort kinases within the group by AUPR scores
            sorted_kinases = sorted(kinase_scores.keys(), key=lambda kinase: kinase_scores[kinase], reverse=True)
            
            # Write kinase names and AUPR scores within the group in separate columns
            for kinase in sorted_kinases:
                writer.writerow([kinase, kinase_scores[kinase]])
            writer.writerow([])  # Empty row for separation

def plot_kinase_group_distributions(args, save_filepath):
    # here first plot the kinase counts in train/valid/test
    # then plot the total data count for all of these kinases
    uniprots_to_family, uniprots_to_groups = map_family_group_info(args)
    all_groups = list(set(uniprots_to_groups.values()))
    '''if "missing group" in all_groups:
        all_groups.remove("missing group")
        '''
    all_groups.sort()

    group_to_kinase_set_train, group_to_data_count_train, group_to_kinase_set_valid, group_to_data_count_valid, group_to_kinase_set_test, group_to_data_count_test = get_kinase_group_distributions(
            args, all_groups, uniprots_to_groups)

    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    font_size = 7

    x_values_kinase_count_in_group_train = [len(v) for k, v in group_to_kinase_set_train.items()]
    x_values_data_count_in_group_train = [v for k, v in group_to_data_count_train.items()]

    x_values_kinase_count_in_group_valid = [len(v) for k, v in group_to_kinase_set_valid.items()]
    x_values_data_count_in_group_valid = [v for k, v in group_to_data_count_valid.items()]

    x_values_kinase_count_in_group_test = [len(v) for k, v in group_to_kinase_set_test.items()]
    x_values_data_count_in_group_test = [v for k, v in group_to_data_count_test.items()]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10),
                             gridspec_kw={
                                 'width_ratios': [1, 1, 1],
                                 'height_ratios': [0.1, 10, 10],
                                 'wspace': 0.4,
                                 'hspace': 1})

    plt.rcParams['font.size'] = font_size
    pad = 3  # in points
    cols = ["Train set", "Validation set", "Test set"]
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    column = 0
    axes[0, column].set_axis_off()
    text = f'Total unique kinases : {sum(x_values_kinase_count_in_group_train)}\n' \
           f'Total row counts : {sum(x_values_data_count_in_group_train)}'
    axes[0, column].text(0.5, 0.5, text, ha='center', va='top')

    axes[1, column].bar(all_groups, x_values_kinase_count_in_group_train, color=grayish_color)
    axes[1, column].set_xlabel('Kinase Groups', fontsize=font_size, fontweight='bold')
    axes[1, column].set_ylabel('Unique Kinase Count', fontsize=font_size, fontweight='bold')
    axes[1, column].tick_params(axis='x', rotation=45, labelsize=font_size)
    for i, count in enumerate(x_values_kinase_count_in_group_train):
        axes[1, column].text(i, count + 0.1, str(count), ha='center')
    axes[1, column].set_axisbelow(True)
    axes[1, column].grid(axis='y', zorder=0)
    axes[1, column].spines['top'].set_visible(False)
    axes[1, column].spines['right'].set_visible(False)
    axes[1, column].spines['left'].set_visible(False)
    axes[1, column].tick_params(axis='both', which='both', length=0)

    axes[2, column].bar(all_groups, x_values_data_count_in_group_train, color=grayish_color)
    axes[2, column].set_xlabel('Kinase Groups', fontsize=font_size, fontweight='bold')
    axes[2, column].set_ylabel('Total data(row) count', fontsize=font_size, fontweight='bold')
    axes[2, column].tick_params(axis='x', rotation=45, labelsize=font_size)
    for i, count in enumerate(x_values_data_count_in_group_train):
        axes[2, column].text(i, count + 0.5, str(count), ha='center')
    axes[2, column].set_axisbelow(True)
    axes[2, column].grid(axis='y', zorder=0)
    axes[2, column].spines['top'].set_visible(False)
    axes[2, column].spines['right'].set_visible(False)
    axes[2, column].spines['left'].set_visible(False)
    axes[2, column].tick_params(axis='both', which='both', length=0)

    column = 1
    axes[0, column].set_axis_off()
    text = f'Total unique kinases : {sum(x_values_kinase_count_in_group_valid)}\n' \
           f'Total row counts : {sum(x_values_data_count_in_group_valid)}'
    axes[0, column].text(0.5, 0.5, text, ha='center', va='top')

    axes[1, column].bar(all_groups, x_values_kinase_count_in_group_valid, color=grayish_color)
    axes[1, column].set_xlabel('Kinase Groups', fontsize=font_size, fontweight='bold')
    axes[1, column].set_ylabel('Unique Kinase Count', fontsize=font_size, fontweight='bold')
    axes[1, column].tick_params(axis='x', rotation=45, labelsize=font_size)
    for i, count in enumerate(x_values_kinase_count_in_group_valid):
        axes[1, column].text(i, count + 0.5, str(count), ha='center')
    axes[1, column].set_axisbelow(True)
    axes[1, column].grid(axis='y', zorder=0)
    axes[1, column].spines['top'].set_visible(False)
    axes[1, column].spines['right'].set_visible(False)
    axes[1, column].spines['left'].set_visible(False)
    axes[1, column].tick_params(axis='both', which='both', length=0)

    axes[2, column].bar(all_groups, x_values_data_count_in_group_valid, color=grayish_color)
    axes[2, column].set_xlabel('Kinase Groups', fontsize=font_size, fontweight='bold')
    axes[2, column].set_ylabel('Total data(row) count', fontsize=font_size, fontweight='bold')
    axes[2, column].tick_params(axis='x', rotation=45, labelsize=font_size)
    for i, count in enumerate(x_values_data_count_in_group_valid):
        axes[2, column].text(i, count + 0.5, str(count), ha='center')
    axes[2, column].set_axisbelow(True)
    axes[2, column].grid(axis='y', zorder=0)
    axes[2, column].spines['top'].set_visible(False)
    axes[2, column].spines['right'].set_visible(False)
    axes[2, column].spines['left'].set_visible(False)
    axes[2, column].tick_params(axis='both', which='both', length=0)

    column = 2
    axes[0, column].set_axis_off()
    text = f'Total unique kinases : {sum(x_values_kinase_count_in_group_test)}\n' \
           f'Total row counts : {sum(x_values_data_count_in_group_test)}'
    axes[0, column].text(0.5, 0.5, text, ha='center', va='top')

    axes[1, column].bar(all_groups, x_values_kinase_count_in_group_test, color=grayish_color)
    axes[1, column].set_xlabel('Kinase Groups', fontsize=font_size, fontweight='bold')
    axes[1, column].set_ylabel('Unique Kinase Count', fontsize=font_size, fontweight='bold')
    axes[1, column].tick_params(axis='x', rotation=45, labelsize=font_size)
    for i, count in enumerate(x_values_kinase_count_in_group_test):
        axes[1, column].text(i, count + 0.5, str(count), ha='center')
    axes[1, column].set_axisbelow(True)
    axes[1, column].grid(axis='y', zorder=0)
    axes[1, column].spines['top'].set_visible(False)
    axes[1, column].spines['right'].set_visible(False)
    axes[1, column].spines['left'].set_visible(False)
    axes[1, column].tick_params(axis='both', which='both', length=0)

    axes[2, column].bar(all_groups, x_values_data_count_in_group_test, color=grayish_color)
    axes[2, column].set_xlabel('Kinase Groups', fontsize=font_size, fontweight='bold')
    axes[2, column].set_ylabel('Total data(row) count', fontsize=font_size, fontweight='bold')
    axes[2, column].tick_params(axis='x', rotation=45, labelsize=font_size)
    for i, count in enumerate(x_values_data_count_in_group_test):
        axes[2, column].text(i, count + 0.5, str(count), ha='center')
    axes[2, column].set_axisbelow(True)
    axes[2, column].grid(axis='y', zorder=0)
    axes[2, column].spines['top'].set_visible(False)
    axes[2, column].spines['right'].set_visible(False)
    axes[2, column].spines['left'].set_visible(False)
    axes[2, column].tick_params(axis='both', which='both', length=0)

    fig.savefig(f"{save_filepath}/kinase_groups_distribution.png")

def plot_group_aupr_scores(config, save_filepath, kinase_info_dict):
    group_scores = calculate_group_based_aupr_scores(config, kinase_info_dict)

    all_groups = list(set(group_scores.keys()))
    all_groups.sort()
    groups_based_aupr_scores = []
    for group in all_groups:
        groups_based_aupr_scores.append(np.mean(list(group_scores[group].values())))

    fig, ax = plt.subplots()  # Create figure and axis objects
    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    plt.bar(all_groups, groups_based_aupr_scores, color=grayish_color, linewidth=0.5)
    plt.xlabel('Groups', fontweight='bold')
    plt.ylabel('AUPR Scores', fontweight='bold')
    plt.title('Kinase Group AP Scores')
    ax.set_axisbelow(True)
    ax.grid(axis='y', zorder=0)

    # Remove the line boundaries
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', which='both', length=0)
    for i, count in enumerate(groups_based_aupr_scores):
        plt.text(i, count + 0.001, f'{count:.4f}', ha='center')

    fig.savefig(f"{save_filepath}/Group_Based_AUPR_Scores_Bar_Plot.png")

def calculate_group_based_aupr_scores(args, kinase_info_dict):
    uniprots_to_family, uniprots_to_groups = map_family_group_info(args)
    group_scores = {}
    for kinase, kinase_score_dict in kinase_info_dict.items():
        kinase_group = uniprots_to_groups[kinase]
        if kinase_group not in group_scores:
            group_scores[kinase_group] = {}
        group_scores[kinase_group][kinase] = kinase_score_dict["aupr"]
    return group_scores

def plot_scatter_plot_with_kinase_aupr_scores_interactive(config, save_filepath, kinase_info_dict):
    uniprots_to_family, uniprots_to_groups = map_family_group_info(config)

    filename = config['kinase_similarity_file']
    df = pd.read_csv(f'{filename}', index_col=0)
    similarity_matrix = df.values

    tsne = TSNE(n_components=2, random_state=0)
    reduced = tsne.fit_transform(similarity_matrix)
    reduced_df = pd.DataFrame(reduced, index=df.index, columns=['x', 'y'])

    # Create a list to store colors
    colors = []

    kinase_aupr_scores = {kinase : info_dict["aupr"] for kinase, info_dict in kinase_info_dict.items()}

    # Loop through the kinases and assign colors based on AUPR scores
    default_color = 'darkgray'
    cmap = plt.get_cmap('coolwarm')  # You can choose a different colormap if you prefer
    for kinase in reduced_df.index:
        score = kinase_aupr_scores.get(kinase, None)
        if score is not None:
            normalized_score = (score - min(kinase_aupr_scores.values())) / (max(kinase_aupr_scores.values()) - min(kinase_aupr_scores.values()))
            colors.append(cmap(normalized_score))
        else:
            colors.append(default_color)

    # Create the scatter plot with the specified colors
    # plt.scatter(reduced_df['x'], reduced_df['y'], color=colors)

    # Create the plot layout
    plt.title('Global Alignment - Sequence Alignment Identity Score')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')

    reduced_df['Kinase'] = reduced_df.index
    reduced_df['AUPR'] = reduced_df['Kinase'].map(kinase_aupr_scores).fillna(np.nan)  # Use np.nan for missing values
    reduced_df['Group'] = reduced_df['Kinase'].map(uniprots_to_groups).fillna('NA')

    # Add a column for the size of each dot
    reduced_df['Size'] = reduced_df['AUPR'].apply(lambda x: 70 if not pd.isna(x) else 20)  # Handle np.nan

    # Create the custom color scale
    color_scale = [(0, 'blue'), (1, 'brown')]

    fig = px.scatter(reduced_df, x='x', y='y', color='AUPR',
                     size='Size',
                     hover_data={'Kinase': True, 'AUPR': True, 'Group': True, 'x': False, 'y': False, 'Size': False},
                     color_continuous_scale=color_scale,
                     range_color=[min(kinase_aupr_scores.values()), max(kinase_aupr_scores.values())])

    fig.update_layout(title='Global Alignment - Sequence Alignment Identity Score',
                      xaxis_title='t-SNE dimension 1',
                      yaxis_title='t-SNE dimension 2',
                      coloraxis_showscale=True)

    fig.update_layout(
        plot_bgcolor='lightgray',  # Change 'white' to whatever color you want for the plot background
        paper_bgcolor='lightgray',  # Change 'white' to whatever color you want for the paper background
        xaxis=dict(gridcolor='gray'),  # Change 'lightgray' to your desired x-axis grid color
        yaxis=dict(gridcolor='gray')  # Change 'lightgray' to your desired y-axis grid color
    )

    # fig.show()
    fig.write_html(f'{save_filepath}/kinase_aupr_score_scatter_plot.html')

def plot_aupr_score_to_heterogenity_analysis(config, save_filepath, kinase_to_info_dict):
    kinase_to_count = get_kinase_count_in_test_data(config)

    # Assuming kinase_to_aupr and kinase_to_count are dictionaries with kinase names as keys
    kinases, kinase_to_aupr_sync, kinase_to_data_count_sync = [], [], []
    for k, v in kinase_to_info_dict.items():
        kinases.append(k)
        kinase_to_aupr_sync.append(v['aupr'])
        kinase_count = kinase_to_count[k] if k in kinase_to_count else 0
        kinase_to_data_count_sync.append(kinase_count)

    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    plt.figure(figsize=(10, 6))
    plt.scatter(kinase_to_data_count_sync, kinase_to_aupr_sync, alpha=0.6, color=grayish_color)
    plt.title("Relation between Kinase AUPR Scores and Data Count")
    plt.xlabel("Data Count")
    plt.ylabel("AUPR Score")
    plt.savefig(f'{save_filepath}/Kinase_AUPR_Score_vs_Data_Count.png')


def plot_train_and_valid_loss(args, save_filepath, train_loss, valid_loss, model_no):
    grayish_color = (0.35, 0.35, 0.35)
    darker_grayish_color = (0.25, 0.25, 0.25)
    
    plt.figure(figsize=(10,6))
    plt.plot(train_loss, label='Training Loss', color='blue')  
    plt.plot(valid_loss, label='Validation Loss', color='brown')  
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Display legend
    plt.legend()

    # Show plot
    plt.savefig(f'{save_filepath}/train_and_validation_loss__{args["logging"]["local"]["checkpoint_file_name"]}.png')

def plot_group_and_family_score(args, save_filepath, group_or_family, train_group, validation_group, model_no):
    grayish_color = (0.35, 0.35, 0.35)
    darker_grayish_color = (0.25, 0.25, 0.25)
    
    plt.figure(figsize=(10,6))
    plt.plot(train_group, label=f'Training {group_or_family}', color='blue')  
    plt.plot(validation_group, label=f'Validation {group_or_family}', color='brown')  
    plt.title(f'Training and Validation {group_or_family}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{group_or_family} score')

    # Display legend
    plt.legend()

    # Show plot
    plt.savefig(f'{save_filepath}/train_and_validation_{group_or_family}_score__{args["logging"]["local"]["checkpoint_file_name"]}.png')


def plot_train_and_valid_losses(config, save_filepath, all_train_losses, all_valid_losses):
    if all_train_losses != None and all_valid_losses != None:
        for i in range(len(all_train_losses)):
            plot_train_and_valid_loss(config, save_filepath, all_train_losses[i], all_valid_losses[i], i)

def plot_group_and_family_scores(config, save_filepath, group_or_family, train_scores, validation_scores):
    if train_scores != None and validation_scores != None:
        for i in range(len(validation_scores)):
            # plot_group_and_family_score(args, save_filepath, group_or_family, train_group, validation_group, model_no)
            plot_group_and_family_score(config, save_filepath, group_or_family, train_scores[i], validation_scores[i], i)


def plot_train_data_size_vs_group_based_aupr(config, save_filepath, kinase_info_dict):
    uniprots_to_family, uniprots_to_groups = map_family_group_info(config)
    group_scores = calculate_group_based_aupr_scores(config, kinase_info_dict)
    all_groups_in_test = list(set(group_scores.keys()))
    all_groups_in_test.sort()

    all_groups = list(set(uniprots_to_groups.values()))
    if "missing" in all_groups:
        all_groups.remove("missing")
    all_groups.sort()

    group_to_kinase_set_train, group_to_data_count_train, group_to_kinase_set_valid, group_to_data_count_valid, group_to_kinase_set_test, group_to_data_count_test = get_kinase_group_distributions(
            config, all_groups, uniprots_to_groups)
        
    group_train_data_count = []
    group_based_test_aupr_scores = []
    for group in all_groups_in_test:
        group_train_data_count.append(group_to_data_count_train[group])
        group_based_test_aupr_scores.append(np.mean(list(group_scores[group].values())))

    # Creating a DataFrame for easier plotting with Seaborn
    df = pd.DataFrame({
        'TrainDataCount': group_train_data_count,
        'TestAUPRScore': group_based_test_aupr_scores,
        'Group': all_groups_in_test
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='TrainDataCount', y='TestAUPRScore', hue='Group', palette="deep", s=100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Relation between Kinase AUPR Scores and Data Count in Train")
    plt.xlabel("Group Based Data Count in Train")
    plt.ylabel("Group Based AUPR Score in Test")
    plt.tight_layout()

    plt.savefig(f'{save_filepath}/Kinase_Train_Data_Count_vs_Group_Based_AUPR_Scores_in_test.png')

def plot_train_kinase_count_vs_group_based_aupr(config, save_filepath, kinase_info_dict):
    uniprots_to_family, uniprots_to_groups = map_family_group_info(config)
    group_scores = calculate_group_based_aupr_scores(config, kinase_info_dict)
    all_groups_in_test = list(set(group_scores.keys()))
    all_groups_in_test.sort()

    all_groups = list(set(uniprots_to_groups.values()))
    if "missing" in all_groups:
        all_groups.remove("missing")
    all_groups.sort()

    group_to_kinase_set_train, group_to_data_count_train, group_to_kinase_set_valid, group_to_data_count_valid, group_to_kinase_set_test, group_to_data_count_test = get_kinase_group_distributions(
            config, all_groups, uniprots_to_groups)
        
    group_kinase_count_in_train = []
    group_based_test_aupr_scores = []
    for group in all_groups_in_test:
        group_kinase_count_in_train.append(len(group_to_kinase_set_train[group]))
        group_based_test_aupr_scores.append(np.mean(list(group_scores[group].values())))

    df = pd.DataFrame({
        'TrainKinaseCount': group_kinase_count_in_train,
        'TestAUPRScore': group_based_test_aupr_scores,
        'Group': all_groups_in_test
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='TrainKinaseCount', y='TestAUPRScore', hue='Group', palette="deep", s=100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Relation between Kinase AUPR Scores and Kinase Count in Train")
    plt.xlabel("Group Based Kinase Count in Train")
    plt.ylabel("Group Based AUPR Score in Test")
    plt.tight_layout()

    plt.savefig(f'{save_filepath}/Kinase_Train_Kinase_Count_vs_Group_Based_AUPR_Scores_in_test.png')

def plot_kinase_portion_per_group_vs_group_based_aupr(config, save_filepath, kinase_info_dict):
    uniprots_to_family, uniprots_to_groups = map_family_group_info(config)
    group_scores = calculate_group_based_aupr_scores(config, kinase_info_dict)
    all_groups_in_test = list(set(group_scores.keys()))
    all_groups_in_test.sort()

    all_groups = list(set(uniprots_to_groups.values()))
    if "missing" in all_groups:
        all_groups.remove("missing")
    all_groups.sort()

    group_to_kinase_set_train, group_to_data_count_train, group_to_kinase_set_valid, group_to_data_count_valid, group_to_kinase_set_test, group_to_data_count_test = get_kinase_group_distributions(
            config, all_groups, uniprots_to_groups)
        
    group_kinase_portion_per_group = [] # This is like total_data_in_group/total_kinase_in_group, so the number of data points per kinase in that group
    group_based_test_aupr_scores = []
    for group in all_groups_in_test:
        data_count_per_kinase = group_to_data_count_train[group]/len(group_to_kinase_set_train[group]) if group_to_data_count_train[group] != 0 else 0
        group_kinase_portion_per_group.append(data_count_per_kinase)
        group_based_test_aupr_scores.append(np.mean(list(group_scores[group].values())))

    df = pd.DataFrame({
        'TrainDataPortionPerKinaseCount': group_kinase_portion_per_group,
        'TestAUPRScore': group_based_test_aupr_scores,
        'Group': all_groups_in_test
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='TrainDataPortionPerKinaseCount', y='TestAUPRScore', hue='Group', palette="deep", s=100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Relation between Kinase AUPR Scores and Group_Count/Kinase_Count_in_Group")
    plt.xlabel("Group Based avg data count per kinase in Train")
    plt.ylabel("Group Based AUPR Score in Test")
    plt.tight_layout()

    plt.savefig(f'{save_filepath}/Kinase_Train_avg_data_per_kinase_vs_Group_Based_AUPR_Scores_in_test.png')

def find_project_root():
    marker = 'main.py'
    root = os.getcwd()
    while root != os.path.dirname(root):  # Check until the root of the filesystem is reached
        if marker in os.listdir(root):
            return root
        root = os.path.dirname(root)
    return None

def get_similar_train_kinase_data_count_of_test_kinases_from_same_group(config, group=True):
    similarity_df = pd.read_csv(SIMILARITY_FILE, index_col=0)

    uniprots_to_families, uniprots_to_groups = map_family_group_info(config)

    if group:
        uniprots_to_whatever = uniprots_to_groups
    else:
        uniprots_to_whatever = uniprots_to_families

    train_df = pd.read_csv(config['phosphosite']['dataset']['train'], index_col=0)
    test_df = pd.read_csv(config['phosphosite']['dataset']['test'], index_col=0)

    all_groups = set(uniprots_to_whatever.values())

    # Count occurrences of each kinase in the training dataset
    train_kinase_counts = dict()
    group_to_train_data_count = {group : 0 for group in all_groups}
    for index, row in train_df.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = list(set(kinases.split(',')))
        kinases.sort()
        for kinase in kinases:
            if kinase not in train_kinase_counts:
                train_kinase_counts[kinase] = 0
            train_kinase_counts[kinase] += 1

            train_group = uniprots_to_whatever[kinase]
            group_to_train_data_count[train_group] += 1


    all_test_kinases = set()
    for index, row in test_df.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = list(set(kinases.split(',')))
        kinases.sort()
        for kinase in kinases:
            all_test_kinases.add(kinase)

    # Specify a similarity threshold
    count_train_kinases = dict()

    for test_kinase in all_test_kinases:
        test_kinase_group = uniprots_to_whatever[test_kinase]
        count_train_kinases[test_kinase] = group_to_train_data_count[test_kinase_group]

    return count_train_kinases, uniprots_to_whatever

def find_test_kinase_AP_score_to_train_count_scatter_plot_family_subb(config, ax, model_type, kinase_info_dict, similarity_threshold, group):
    count_train_kinases, uniprots_to_groups = get_similar_train_kinase_data_count_of_test_kinases_from_same_group(config, group)

    test_kinase_AP_scores = [v for k, v in kinase_info_dict.items()]
    test_kinases_train_count = [count_train_kinases[k] for k in kinase_info_dict.keys()]
    groups = [uniprots_to_groups[k] for k in kinase_info_dict.keys()]

    # Define unique groups and colors
    unique_groups = sorted(set(groups))
    color_map = {group: cm.tab20(i / len(unique_groups)) for i, group in enumerate(unique_groups)}

    min_train_count = min([count_train_kinases[k] for k in kinase_info_dict.keys()])
    max_train_count = max([count_train_kinases[k] for k in kinase_info_dict.keys()])
    min_ap_score = min([v for k, v in kinase_info_dict.items()])
    max_ap_score = max([v for k, v in kinase_info_dict.items()])

    # Plot each group with a different color
    for group in unique_groups:
        mask = [g == group for g in groups]
        ax.scatter(
            [tc for tc, m in zip(test_kinases_train_count, mask) if m],
            [ap for ap, m in zip(test_kinase_AP_scores, mask) if m],
            s=200,
            color=color_map[group],
            edgecolors='#F0F0F0',
            linewidths=0.8,
            label=group,
            zorder=3
        )

    '''
    model_info = {
        "default": default_model,
        "QFSL": qfsl_model,
        "upsampling": upsampled_model,
        "upsampling_with_PL": pseudolabeled_model
    }
    '''
    if model_type in {"Up-sampling", "Up-sampling with Pseudo-labeling"}:
        ax.set_xlabel('Related Train Kinase Count', fontname='Arial')
    if model_type in {"DeepKinZero", "Up-sampling"}:
        ax.set_ylabel('AP Score', fontname='Arial')
    ax.set_title(f'Model: {model_type}', fontname='Arial')

    return min_train_count, max_train_count, min_ap_score, max_ap_score

def combined_family_sub_figures(config, default_model, qfsl_model, upsampled_model, pseudolabeled_model, similarity_threshold, save_filepath, group=True):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 18  # You can adjust the size as needed

    fig, axs = plt.subplots(2, 2, figsize=(20, 15))# , facecolor='#F0F0F0')  # Increase the overall figure size
    axs = axs.flatten()

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 18  # You can adjust the size as needed

    # Define your models and information dictionaries
    model_info = {
        "DeepKinZero": default_model,
        "QFSM": qfsl_model,
        "Up-sampling": upsampled_model,
        "Up-sampling with Pseudo-labeling": pseudolabeled_model
    }

    similarity_threshold = 50

    global_min_train_count = float('inf')
    global_max_train_count = float('-inf')
    global_min_ap_score = float('inf')
    global_max_ap_score = float('-inf')

    # Plot each model in its subplot
    for ax, (model_type, kinase_info_dict) in zip(axs, model_info.items()):
        min_train_count, max_train_count, min_ap_score, max_ap_score = find_test_kinase_AP_score_to_train_count_scatter_plot_family_subb(config, ax, model_type, kinase_info_dict,
                                                                     similarity_threshold, group)
        # ax.set_facecolor('#E0E0E0')
        global_min_train_count = min(global_min_train_count, min_train_count)
        global_max_train_count = max(global_max_train_count, max_train_count)
        global_min_ap_score = min(global_min_ap_score, min_ap_score)
        global_max_ap_score = max(global_max_ap_score, max_ap_score)

    # Adjust subplot parameters to give more space for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)  # Adjust the right margin to create space for the legend

    # Handle legend
    handles, labels = axs[0].get_legend_handles_labels()  # You can use any axes to collect the legend handles
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize='medium', title=f"{'Families' if not group else 'Groups'}",
               title_fontsize='medium') #, facecolor='#E0E0E0')

    for ax in axs:
        ax.set_xlim(global_min_train_count-100, global_max_train_count + 100)
        ax.set_ylim(global_min_ap_score-0.1, global_max_ap_score + 0.05)

    # Add a global title above all subplots
    type = f"{'Family' if not group else 'Group'}"
    plt.subplots_adjust(top=0.92)
    fig.suptitle(f'Test AP Score vs. Related Train Kinases ({type})', fontsize=30, fontname='Arial')
    # plt.tight_layout(pad=3)  # Adjust padding as needed

    # Save the figure
    type = f"{'Families' if not group else 'Groups'}"
    plt.savefig(f'{save_filepath}/{type}_combined_model_comparison.png', bbox_inches='tight')
    plt.savefig(f'{save_filepath}/{type}_combined_model_comparison.pdf', bbox_inches='tight')
    # plt.show()

'''
This is the scatter plot of the test kinase AP scores vs the related train kinase count of these test kinases (according to kinase group or famil conection )
This could be used to compare the results of 4 different models
'''
def run_comaprison_satter_plot_of_4_diff_models(config):
    save_filepath = create_folder(config)
    
    # replace these with your model's results:
    default_model = {"Q96QT4":0.07984563899277804,"O00418":0.017651059567465054,"Q9UHD2":0.1560326539641929,"O00444":0.069354894044134,"Q9HC98":0.06859788281660395,"P45983":0.49554053159727013,"P19784":0.4281025119024467,"Q00526":0.2578056130818315,"P49759":0.2500394046948316,"Q9Y463":0.2090752219949207,"P45984":0.15515476579215337,"Q13627":0.13024406360317853,"O15264":0.12582182668825617,"P53779":0.0987704065084088,"P53778":0.08607329856439255,"Q13164":0.05940250692874611,"Q9H0K1":0.4299373846125153,"O43293":0.25104634849113666,"Q14012":0.0960786107380747,"Q15831":0.053426653558617435,"Q14680":0.023706585308365596,"P23443":0.537372407145668,"P05129":0.48037464947986475,"Q13464":0.18242201902484437,"P34947":0.15108824188614384,"P25098":0.09019502535732726,"O75116":0.02672368693679928,"P35626":0.026314484050580584,"P00533":0.47826418988831193,"P43405":0.40696767213362206,"P07947":0.33615527221894453,"P51813":0.0785131968929738,"P04049":0.4015961950946227,"Q38SD2":0.1570793878264212,"Q9UEW8":0.5067189724304876,"Q99683":0.2965909845045732,"Q99558":0.04236921357564154,"Q99986":0.23500167785232154}
    qfsl_model = {"Q96QT4":0.10228556417797892,"O00418":0.03371317545201777,"O00444":0.18888346649327858,"Q9UHD2":0.1793036609369948,"Q9HC98":0.06436092048655914,"P45983":0.504991430244866,"P19784":0.32719164815465845,"P49759":0.32563237874350637,"Q00526":0.21280717841416993,"P45984":0.1647299565247588,"Q9Y463":0.16280244247021627,"Q13627":0.12568888962644498,"O15264":0.1140418066328958,"P53778":0.09439116638548392,"P53779":0.05692171639938094,"Q13164":0.04972357931889956,"Q9H0K1":0.44470186804816614,"O43293":0.2674991856302113,"Q14012":0.10109727283228885,"Q15831":0.056676850671255295,"Q14680":0.030044421796532165,"P05129":0.5146548957522239,"P23443":0.5100108349437537,"P34947":0.2025278652208076,"Q13464":0.1915663004266262,"P25098":0.09479249811861899,"P35626":0.03069678037614989,"O75116":0.024667339781042505,"P00533":0.4892784327705676,"P43405":0.4463493187218172,"P07947":0.3432906931311481,"P51813":0.0856831995639594,"P04049":0.3490584987746047,"Q38SD2":0.1688978490188689,"Q9UEW8":0.49268443909125803,"Q99683":0.3477021069022677,"Q99558":0.033654949531522364,"Q99986":0.23890815255332662}
    upsampled_model = {"Q96QT4":0.06115552128668146,"O00418":0.01874918003562352,"Q9UHD2":0.169004642744628,"O00444":0.07653100824885889,"Q9HC98":0.06441581875007842,"P45983":0.4988034592425378,"P19784":0.4524112150028903,"P49759":0.2918791676816952,"Q00526":0.2764217171739557,"Q9Y463":0.26194963322148485,"P45984":0.17916544406846854,"Q13627":0.14916044473813264,"P53778":0.07243645583890329,"O15264":0.07000453086994018,"Q13164":0.06636533239042597,"P53779":0.051377870499687055,"Q9H0K1":0.3526933332995924,"O43293":0.31594426620790494,"Q14012":0.09868241051460402,"Q15831":0.05675729716494858,"Q14680":0.022699888692749953,"P23443":0.5632026967415588,"P05129":0.4681038025919746,"P34947":0.2249906010301738,"Q13464":0.175561979443003,"P25098":0.0845839838507702,"P35626":0.03898904473943889,"O75116":0.023302257105397448,"P00533":0.47726476611125657,"P43405":0.38095400976638266,"P07947":0.29252101252032514,"P51813":0.10966942073408724,"P04049":0.44824107412488595,"Q38SD2":0.11665876867057531,"Q9UEW8":0.5873162664419084,"Q99683":0.28614028240863254,"Q99558":0.06107311392231765,"Q99986":0.24859989784354716}
    pseudolabeled_model = {"Q96QT4":0.07287558429584758,"O00418":0.014293833090576467,"Q9UHD2":0.19289287346584724,"O00444":0.09689786868978993,"Q9HC98":0.06291555533557862,"P45983":0.5293281856427772,"P19784":0.4354623374736,"Q00526":0.29014768091437837,"P49759":0.2734014390943708,"Q13627":0.17142743534574598,"P45984":0.16984826192866073,"Q9Y463":0.16522359016030405,"O15264":0.11029144509549602,"P53778":0.0984856327821202,"Q13164":0.05925321946845872,"P53779":0.05807198509640714,"Q9H0K1":0.4468941785928238,"O43293":0.30142431839681316,"Q14012":0.09151158191447899,"Q15831":0.05463302044123546,"Q14680":0.025153254429905007,"P23443":0.5501521363004321,"P05129":0.4634324288900466,"P34947":0.18224298644496845,"Q13464":0.17548992122563392,"P25098":0.07773159418392026,"P35626":0.034731247641650094,"O75116":0.027216803371473156,"P00533":0.4611035033073262,"P43405":0.37462406510173224,"P07947":0.31008955473257077,"P51813":0.10002987857660098,"P04049":0.4646933012270275,"Q38SD2":0.16541820502481142,"Q9UEW8":0.658594331569089,"Q99683":0.37524267499751923,"Q99558":0.0705623344773426,"Q99986":0.22986740165035974}

    combined_family_sub_figures(config, default_model, qfsl_model, upsampled_model, pseudolabeled_model, 50, save_filepath, group=True)
    combined_family_sub_figures(config, default_model, qfsl_model, upsampled_model, pseudolabeled_model, 50, save_filepath, group=False)

def find_test_kinase_AP_score_to_train_count_scatter_plot_single_model_result(config, kinase_info_dict, similarity_threshold, dataset_statistics_directory, plot_group):
    count_train_kinases, uniprots_to_groups = get_similar_train_kinase_data_count_of_test_kinases_from_same_group(config, plot_group)

    test_kinase_AP_scores = [v["aupr"] for k, v in kinase_info_dict.items()]
    test_kinases_train_count = [count_train_kinases[k] for k in kinase_info_dict.keys()]
    groups = [uniprots_to_groups[k] for k in kinase_info_dict.keys()]

    # Define unique groups and colors
    unique_groups = sorted(set(groups))
    color_map = {group: cm.tab20(i / len(unique_groups)) for i, group in enumerate(unique_groups)}

    fig, ax = plt.subplots()

    # Plot each group with a different color
    for group in unique_groups:
        mask = [g == group for g in groups]
        ax.scatter(
            [tc for tc, m in zip(test_kinases_train_count, mask) if m],
            [ap for ap, m in zip(test_kinase_AP_scores, mask) if m],
            s=100,
            color=color_map[group],
            edgecolors='#F0F0F0',
            linewidths=0.8,
            label=group,
            zorder=3
        )

    # Add a global title above all subplots
    print(plot_group)
    plot_type = 'Family' if not plot_group else 'Group'
    plt.subplots_adjust(top=0.92)
    fig.suptitle(f'Test AP Score vs. Related Train Kinases ({plot_type})', fontsize=18, fontname='Arial')

    # Save the figure
    plot_type_plural = 'Families' if not plot_group else 'Groups'
    plt.savefig(f'{dataset_statistics_directory}/{plot_type_plural}_test_AP_score_vs_related_train_count.png', bbox_inches='tight')
    plt.savefig(f'{dataset_statistics_directory}/{plot_type_plural}_test_AP_score_vs_related_train_count.pdf', bbox_inches='tight')

def plot_wrong_prediction_count(config, dummy_metric, dataset_statistics_directory):
    kinase_incorrect_counts = calculate_incorrect_predictions(dummy_metric, k_values=[1, 3, 5])
    
    fig, ax = plt.subplots()

    # Define the kinases and their counts at different k values
    kinases = sorted(kinase_incorrect_counts[1].keys())
    num_kinases = len(kinases)
    bar_width = 0.25  # width of each bar

    # Calculate the position of bars
    index = np.arange(num_kinases) * (len(kinase_incorrect_counts) + 1) * bar_width

    # Adding each group of bars
    for i, k in enumerate(sorted(kinase_incorrect_counts.keys())):
        counts = [kinase_incorrect_counts[k][kinase] for kinase in kinases]
        ax.bar(index + i * bar_width, counts, bar_width, label=f'@{k} incorrect')

    # Customizing the plot
    ax.set_xlabel('Kinase')
    ax.set_ylabel('Number of Incorrect Predictions')
    ax.set_title('Incorrect Predictions at Different Top-k Levels')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(kinases, rotation='vertical')
    ax.legend()

    # Show the plot
    # plt.show()
    plt.savefig(f'{dataset_statistics_directory}/Kinase_Incorrect_Precitions_at_Different_Levels.png', bbox_inches='tight')
    plt.savefig(f'{dataset_statistics_directory}/Kinase_Incorrect_Precitions_at_Different_Levels.pdf', bbox_inches='tight')
    

def calculate_incorrect_predictions(dummy_metric, k_values=[1, 3, 5]):
    """
    Calculates the number of incorrect predictions for each kinase at different top-k levels, 
    excluding cases where any true label is already correctly predicted.
    """
    num_samples = dummy_metric.labels.size(0)
    kinase_incorrect_counts = {k: {kin_class_name: 0 for kin_class_name in dummy_metric.label_mapping.values()} for k in k_values}

    for i in range(num_samples):
        label_indices = dummy_metric.labels[i].nonzero().view(-1)  # True labels for the current sample

        for k in k_values:
            topk_indices = torch.argsort(dummy_metric.probabilities[i], descending=True)[:k]
            correct_prediction_made = any(idx in label_indices for idx in topk_indices)
            
            if not correct_prediction_made:
                # Find incorrectly predicted kinases that are not in the true label set
                incorrect_predictions = [idx for idx in topk_indices if idx not in label_indices]
                
                # Increment count for each incorrectly predicted kinase
                for idx in incorrect_predictions:
                    kinase_name = dummy_metric.label_mapping[idx.item()]
                    kinase_incorrect_counts[k][kinase_name] += 1

    return kinase_incorrect_counts

def export_kinase_similarity_to_excel(config, dataset_statistics_directory):
    similarity_df = pd.read_csv("dataset/new_dataset/kinase_pairwise_identity_similarity_scores.csv", index_col=0)
    
    train_df = pd.read_csv(config['phosphosite']['dataset']['train'], index_col=0)
    test_df = pd.read_csv(config['phosphosite']['dataset']['test'], index_col=0)

    train_kinase_counts = train_df["KINASE_ACC_IDS"].str.split(',').explode().value_counts()

    all_test_kinases = set(test_df["KINASE_ACC_IDS"].str.split(',').explode().unique())

    wb = Workbook()
    ws = wb.active
    ws.title = "Kinase Similarities"

    for test_kinase in all_test_kinases:
        if test_kinase not in similarity_df.columns:
            continue

        similarities = similarity_df[test_kinase].sort_values(ascending=False)

        similarities = similarities.drop(test_kinase, errors='ignore')
        similarities = similarities[similarities.index.isin(train_kinase_counts.index)]

        train_kinases = similarities.index.tolist()
        similarity_percents = similarities.values.tolist()
        train_counts = [train_kinase_counts.get(kinase, 0) for kinase in train_kinases]

        ws.append([f"Test Kinase: {test_kinase}"])
        ws.append(['Train Kinase'] + train_kinases)
        ws.append(['Similarity %'] + similarity_percents)
        ws.append(['Count in Train Data'] + train_counts)

    wb.save(f'{dataset_statistics_directory}/Test_Kinases_Most_Similar_Train_Kinases_and_their_Counts_in_train.xlsx')


# def run_helper(config, test_probabilities, test_data_true_labels, all_test_kinase_uniprotIDs, all_train_losses, all_valid_losses):
def run_helper(config, dummy_metric, all_train_losses, all_valid_losses):
    '''
    Boyle cagirmisim onceden:
    test_suite.run(args, Valprobabilities, val_dataset.labels, val_candidate_uniprotid, AllTrainLosses, AllValidLosses, AllValidAuprScore)
    test_suite.run(args, ensembled_test_probabilities, test_data_true_labels, all_test_kinase_uniprotIDs, None, None, None)
    '''
    
    save_filepath = create_folder(config)
    print(f'test suite saved in {save_filepath}')
    write_arguments_to_txt_file(config, save_filepath)
    plot_kinase_group_distributions(config, save_filepath)
    get_kinase_distibution(config, save_filepath)

    kinase_info_dict = calculate_kinase_scores(config, dummy_metric)
    write_down_kinase_scores(config, save_filepath, kinase_info_dict)
    plot_group_aupr_scores(config, save_filepath, kinase_info_dict)
    plot_precision_recall_curves(config, save_filepath, kinase_info_dict)
    plot_group_based_box_plots(config, save_filepath, kinase_info_dict)
    plot_kinase_aupr_score_histogram(config, save_filepath, kinase_info_dict)
    plot_test_kinase_heterogenity(config, save_filepath)
    plot_group_heterogenity(config, save_filepath)
    plot_scatter_plot_with_kinase_aupr_scores_interactive(config, save_filepath, kinase_info_dict)
    plot_aupr_score_to_heterogenity_analysis(config, save_filepath, kinase_info_dict)
    plot_train_and_valid_losses(config, save_filepath, all_train_losses, all_valid_losses)
    plot_train_data_size_vs_group_based_aupr(config, save_filepath, kinase_info_dict)
    plot_train_kinase_count_vs_group_based_aupr(config, save_filepath, kinase_info_dict)
    plot_kinase_portion_per_group_vs_group_based_aupr(config, save_filepath, kinase_info_dict)
    plot_wrong_prediction_count(config, dummy_metric, save_filepath)
    export_kinase_similarity_to_excel(config, save_filepath)

    find_test_kinase_AP_score_to_train_count_scatter_plot_single_model_result(config, kinase_info_dict, 50, save_filepath, plot_group=True)
    find_test_kinase_AP_score_to_train_count_scatter_plot_single_model_result(config, kinase_info_dict, 50, save_filepath, plot_group=False)
    # run_comaprison_satter_plot_of_4_diff_models(config)

def get_ensemble_probabilities(probabilities):
    ensemble_probabilities = torch.stack(probabilities).mean(dim=0)
    return ensemble_probabilities

def run(config, results, all_train_loasses, all_val_loasses):

    # IMPORTANT NOTE! : This wouldn't work for the train results because the train dataset is shuffled!!!
    # So the model's predictions will not be for the same samples in the same prder, thus they cannot
    # be ensembled.
    all_probabilities = []
    all_logits = []
    for result in results:
        metric = result["metrics"]
        all_probabilities.append(metric.get_probabilities())
        all_logits.append(metric.get_unique_logits())
    # The selected metric is not important at this point since we only need the label_mapping etc, which is the same for all  
    ensemble_probabilities = get_ensemble_probabilities(all_probabilities)
    ensemble_logits = get_ensemble_probabilities(all_logits)
    dummy_metric = copy.deepcopy(results[0]["metrics"])
    dummy_metric.set_probabilities(ensemble_probabilities)
    dummy_metric.set_unique_logits(ensemble_logits)
    ensembled_macro_aupr = dummy_metric.calculate_aupr()
    ensembled_macro_aupr_with_logits = dummy_metric.calculate_aupr(use_logits=True)

    run_helper(config, dummy_metric, all_train_loasses, all_val_loasses)

'''test_probabilities = torch.tensor([[5.44519135e-05,2.70887148e-02,2.16445464e-04,1.67453800e-05,
2.05929155e-05,4.82846360e-07,3.90963869e-05,4.35719412e-04,
1.55012003e-05,9.68210399e-01],
[4.59658302e-04,1.23565137e-01,3.66785494e-03,9.88461892e-04,
2.56443251e-04,6.07009897e-05,7.00493576e-03,3.61040495e-02,
9.07231297e-04,4.10507798e-01],
[8.00930371e-04,2.00624764e-02,5.21210767e-03,1.61027703e-02,
7.01565109e-03,2.21661234e-04,1.04684262e-02,2.23539516e-01,
4.01592627e-03,5.79026751e-02],
[1.36615417e-04,1.82900706e-03,4.17564093e-04,6.33311065e-05,
3.18997013e-06,9.98690259e-03,7.96641416e-06,2.86090417e-05,
3.43944788e-01,7.44750723e-03],
[5.29103127e-05,6.33066101e-03,4.27846052e-02,2.86929280e-05,
1.45467318e-04,5.40537849e-06,1.96556750e-04,1.46705471e-03,
2.68425702e-05,3.98748601e-03],
[8.32173217e-04,7.34224590e-03,1.69975497e-03,1.11558028e-02,
3.60451709e-03,3.63505933e-05,6.76335476e-04,8.43918696e-02,
2.29802215e-04,7.96125233e-02],
[7.27011415e-04,3.16066593e-02,8.62972985e-04,9.85774696e-02,
4.47139051e-03,7.50659956e-06,2.58784066e-03,4.82417643e-02,
5.37396176e-04,2.27044418e-01],
[2.48690601e-04,9.50214360e-03,4.28550702e-05,9.87685144e-06,
3.07079148e-07,1.14010619e-02,7.84070198e-06,8.40122448e-06,
7.94779718e-01,2.86024064e-02],
[3.26804555e-04,1.29054738e-02,1.05431955e-03,4.46339101e-02,
3.80057120e-03,6.11509540e-06,2.75316555e-03,4.01336223e-01,
7.56816662e-05,1.00787371e-01],
[7.20589960e-05,1.73036521e-03,1.09582856e-04,1.81666246e-06,
7.37404378e-07,2.10624915e-02,1.53324527e-06,1.73878361e-05,
3.09694171e-01,1.59370247e-02]])

true_labels = torch.tensor([9, 23, 10, 5, 21, 10, 10, 14, 10, 14])

uniprotIDs = ['O00418', 'O00444', 'O15264', 'O43293', 'O75116', 'P00533', 'P04049', 'P05129', 'P07947', 'P19784']

config = load_config(f'{ROOT_DIRECTORY}/configs/pseudo_labeling/protvec_all_features_lstm_12345.yaml')
config['mode'] = 'train'
config['kinase_similarity_file'] = '/truba/scratch/esunar/DeepKinZero/DeepKinZero/dataset/new_dataset/kinase_pairwise_identity_similarity_scores.csv'
all_train_losses = [[10, 5, 1, 0.5, 0.3, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
all_valid_losses = [[12, 10, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
run(config, test_probabilities, true_labels, uniprotIDs, all_train_losses, all_valid_losses)'''