import argparse
import csv
import os
import pickle
import random
import numpy as np
import pandas as pd
import copy

from matplotlib import pyplot as plt

CURATED_KINASE_DATASET_FILE = "data_files/Curated_557_kinase_dataset.csv"
CLEAN_KINASE_SUBSTRATE_DATASET = "data_files/Clean_Kinase_Substrate_Dataset.csv"
KINASE_SIMILARITY_SCORE_FILE = "data_files/kinase_similarity_matrix.csv"
MULTI_CLASS_KINASE_SUBSTRATE_DATASET_FILE = "data_files/Formatted_Kinase_Substrate_Dataset.csv"
RANDOM_SEED = 0

def set_random_seed(random_seed):
    global RANDOM_SEED
    RANDOM_SEED = random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

def create_folder_if_non_existent(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_all_kinases_from_kinase_dataset():
    file = CURATED_KINASE_DATASET_FILE
    df = pd.read_csv(file)
    UNIPROT_ID_COLUMN_NAME = "Entry"
    uniprotIDs = list(df[UNIPROT_ID_COLUMN_NAME].unique())
    uniprotIDs.sort()
    return uniprotIDs

def map_kinase_family_group_info():
    file = CURATED_KINASE_DATASET_FILE
    df = pd.read_csv(file)

    uniprots_to_family = {}
    uniprots_to_group = {}
    for index, row in df.iterrows():
        family = row["Family"] if not pd.isna(row["Family"]) else "missing"
        group = row["Group"] if not pd.isna(row["Group"]) else "missing"

        uniprotID = row["Entry"]

        uniprots_to_family[uniprotID] = family
        uniprots_to_group[uniprotID] = group

    return uniprots_to_family, uniprots_to_group

def get_kinase_occurrence_in_the_dataset():
    all_kinases = get_all_kinases_from_kinase_dataset()
    kinase_to_occurrence = {uniprot: 0 for uniprot in all_kinases}

    file = CLEAN_KINASE_SUBSTRATE_DATASET
    df = pd.read_csv(file)

    for index, row in df.iterrows():
        uniprotID = row["KIN_ACC_ID"]  # since we deleted all isoforms and the fusion kinases the remaining accession ids are equal to the uniprotIDs
        if uniprotID in kinase_to_occurrence:
            kinase_to_occurrence[uniprotID] += 1

    return kinase_to_occurrence

def get_kinase_pair_to_similarity_score():
    similarity = {}

    with open(KINASE_SIMILARITY_SCORE_FILE, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row

        for row in reader:
            for i in range(1, len(row)):
                column_kinase = header[i]
                row_kinase = row[0]
                if column_kinase != row_kinase:
                    key = (header[i], row[0])
                    sorted_key = tuple(sorted(key))  # so that (a,b) = (b,a)
                    value = float(row[i])
                    if sorted_key not in similarity:
                        similarity[sorted_key] = value

    return similarity

def find_kinases_with_high_sequence_similarity(kinase_similarity_percent):
    similarity_scores_map = get_kinase_pair_to_similarity_score()
    kinases_with_high_sim_scores = dict()
    count = 0
    isadded = False
    for kinase_pair, sim_score in similarity_scores_map.items():
        if sim_score >= kinase_similarity_percent:
            for key, value in kinases_with_high_sim_scores.items():
                if kinase_pair[0] in value or kinase_pair[1] in value:
                    kinases_with_high_sim_scores[key].add(kinase_pair[0])
                    kinases_with_high_sim_scores[key].add(kinase_pair[1])
                    isadded = True
                    break
            if not isadded:
                kinases_with_high_sim_scores[count] = {kinase_pair[0], kinase_pair[1]}
                count += 1
            isadded = False

    return kinases_with_high_sim_scores

def get_train_rows(train_test, setup):
    train_rows = dict()
    df_train = pd.read_csv(f'datasets/random_seed_{RANDOM_SEED}/{setup}/train_data_random_seed_{RANDOM_SEED}.csv')
    for index, row in df_train.iterrows():
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        train_rows[site_identifier] = row
    if train_test:
        df_valid = pd.read_csv(f'datasets/random_seed_{RANDOM_SEED}/{setup}/{setup}_validation_data_random_seed_{RANDOM_SEED}.csv')
        for index, row in df_valid.iterrows():
            site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
            if site_identifier in train_rows:
                row_in_train = train_rows[site_identifier]
                kinases_in_train = row_in_train["KINASE_ACC_IDS"]
                kinases_in_train = set(kinases_in_train.split(','))

                kinases_in_valid = row["KINASE_ACC_IDS"]
                kinases_in_valid = set(kinases_in_valid.split(','))

                kinases_in_train = list(kinases_in_train.union(kinases_in_valid))
                kinases_in_train.sort()
                row_in_train['KINASE_ACC_IDS'] = ",".join(kinases_in_train)
            else:
                train_rows[site_identifier] = row

    train_rows = list(train_rows.values())
    return train_rows

def get_train_kinases(train_test, setup):
    # if it is train_test, this means we need to combine test and validation
    df_train = pd.read_csv(f'datasets/random_seed_{RANDOM_SEED}/{setup}/train_kinases_random_seed_{RANDOM_SEED}.csv')
    train_kinases = set(df_train["Kinase"].unique())
    if train_test:
        df_valid = pd.read_csv(f'datasets/random_seed_{RANDOM_SEED}/{setup}/{setup}_validation_kinases_random_seed_{RANDOM_SEED}.csv')
        valid_kinases = set(df_valid["Kinase"].unique())
        train_kinases = train_kinases.union(valid_kinases)
    return train_kinases

def get_test_kinases(setup):
    df_test = pd.read_csv(f'datasets/random_seed_{RANDOM_SEED}/{setup}/{setup}_test_kinases_random_seed_{RANDOM_SEED}.csv')
    test_kinases = set(df_test["Kinase"].unique())
    return test_kinases

def get_validation_kinases(setup):
    df_valid = pd.read_csv(f'datasets/random_seed_{RANDOM_SEED}/{setup}/{setup}_validation_kinases_random_seed_{RANDOM_SEED}.csv')
    valid_kinases = set(df_valid["Kinase"].unique())
    return valid_kinases

def get_test_kinases_according_to_split(train_test, setup):
    test_kinase = get_test_kinases(setup) if train_test else get_validation_kinases(setup)
    return test_kinase

def get_test_rows(setup):
    df_test = pd.read_csv(f'datasets/random_seed_{RANDOM_SEED}/{setup}/{setup}_test_data_random_seed_{RANDOM_SEED}.csv')
    return [row for _, row in df_test.iterrows()]

def get_validation_rows(setup):
    df_valid = pd.read_csv(f'datasets/random_seed_{RANDOM_SEED}/{setup}/{setup}_validation_data_random_seed_{RANDOM_SEED}.csv')
    return [row for _, row in df_valid.iterrows()]

def get_test_rows_according_to_split(train_test, setup):
    test_rows = get_test_rows(setup) if train_test else get_validation_rows(setup)
    return test_rows

def get_kinase_to_sites_which_its_been_trained_on(train_test_split, setup):
    all_train_rows = get_train_rows(train_test_split, setup)
    kinase_to_sites = dict()
    for row in all_train_rows:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if kinase not in kinase_to_sites:
                kinase_to_sites[kinase] = set()
            kinase_to_sites[kinase].add(site_identifier)
    return kinase_to_sites

def are_there_contradicting_datasets(kinase_in):
    similar_kinases = list(set(kinase_in.keys()))
    similar_kinases.sort()

    first_kinase_datasets = kinase_in[similar_kinases[0]]

    # So logically if the first kinase is in the same dataset with the 2nd, and the 3rd, then the 2nd and 3rd should also be in the same dataset.
    # similar to transitivity?
    for kinase in similar_kinases[1:]:
        other_kinase_datasets = kinase_in[kinase]
        if ("test" in first_kinase_datasets and "train" in other_kinase_datasets) or ("train" in first_kinase_datasets and "test" in other_kinase_datasets):
            return False

    return True

#####################################################################################
#######################   TEST CASE IMPLEMENTATIONS   ###############################
#####################################################################################
def site_count_check_after_train_test_split(all_data_rows, masked_rows_test, sites_in_test, sites_in_train,
                                            uniprot_to_occurrence):
    # The total count of the sites in train + test has to equal to the total site count from the original dataset
    # However the masked rows are added twice, so we have to remove the number of masked rows.
    # masked rows --> these rows contain unseen test kinases, so the version where test kinases are removed are added
    # to train, and the original row itself with the unseen test kinases is added to test. (So its added twice)
    if len(sites_in_train) + len(sites_in_test) - len(masked_rows_test) != len(all_data_rows):
        print(f'FAILED TEST : Number of sites after train_test split doesn\'t match with the site number of the original data. Total sites should be : {len(all_data_rows)}, in the final we\'ve got : {len(sites_in_train) + len(sites_in_test) - len(masked_rows_test)}')
    else:
        print(f'PASSED TEST : Number of sites after train_test split holds with the site number of the original data')

# test_kinase_to_count_new, train_kinase_to_count_new, kinase_to_occurrence
def data_count_check_after_train_test_split(test_kinase_to_count_new, train_kinase_to_count_new):
    kinase_to_occurrence = get_kinase_occurrence_in_the_dataset()
    all_kinases = get_all_kinases_from_kinase_dataset()
    calculated_kinase_to_count = {kinase: 0 for kinase in all_kinases}
    for kinase, count in train_kinase_to_count_new.items():
        calculated_kinase_to_count[kinase] += train_kinase_to_count_new[kinase]
    for kinase, count in test_kinase_to_count_new.items():
        calculated_kinase_to_count[kinase] += test_kinase_to_count_new[kinase]
    errored_kinases = set()
    for kinase, count in calculated_kinase_to_count.items():
        if calculated_kinase_to_count[kinase] != kinase_to_occurrence[kinase]:
            errored_kinases.add(kinase)
            print(f'THERE IS A MISMATCH : {kinase} : should have : {kinase_to_occurrence[kinase]} but has {calculated_kinase_to_count[kinase]}')
    if len(errored_kinases) == 0:
        print(f'PASSED TEST : All kinases have the correct number of data')
    else:
        print(f'FAILED TEST : All kinases dont have the correct number of data')

def check_total_row_counts_from_dictionaries(train_kinase_to_count, valid_kinase_to_count, test_kinase_to_count, kinase_to_occurrence):
    errored_kinases = set()
    for kinase in kinase_to_occurrence:
        kinase_tot_count = 0
        train_count, valid_count, test_count = 0, 0, 0
        if kinase in train_kinase_to_count:
            kinase_tot_count += train_kinase_to_count[kinase]
            train_count = train_kinase_to_count[kinase]
        if kinase in valid_kinase_to_count:
            kinase_tot_count += valid_kinase_to_count[kinase]
            valid_count = valid_kinase_to_count[kinase]
        if kinase in test_kinase_to_count:
            kinase_tot_count += test_kinase_to_count[kinase]
            test_count = test_kinase_to_count[kinase]
        if kinase_to_occurrence[kinase] != kinase_tot_count:
            print(f'FAILED TEST : After train, validation and test splits, not all kinases have the corrct number of their original data count. ({kinase}) Actual count : {kinase_to_occurrence[kinase]} ?= {kinase_tot_count}')
            '''print(f'train count : {train_count}')
            print(f'valid count : {valid_count}')
            print(f'test count : {test_count}')'''
            errored_kinases.add(kinase)
            break
    if len(errored_kinases) == 0:
        print(f'PASSED TEST : After train, validation and test splits, all kinases have the corrct number of their original data count.')

def all_seen_sites_should_have_unseen_test_kinase(sites_in_test, sites_in_train, unseen_test_kinases):
    # By seen sites we mean sites which have also appeared in train. And now is appearing in test lets say.
    seen_sites = dict()
    for row in sites_in_train:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = list(set(kinases.split(',')))
        kinases.sort()
        seen_sites[site_identifier] = kinases
    error = False
    for row in sites_in_test:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        if site_identifier in seen_sites:
            if len(kinases.intersection(unseen_test_kinases)) == 0:
                error = True
    if error:
        print(f'ERROR - THERE IS A SITE WHICH EXISTS IN BOTH TRAIN AND TEST AND HAS NO ADDITIONAL UNSEEN KINASE IN TEST')
    else:
        print("PASSED TEST : test sites which also exist in train have an additional unseen kinase")

def check_duplciate_rows(sites_in_train, sites_in_test, train_test):
    train_site_to_kinase = dict()
    for row in sites_in_train:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = list(set(kinases.split(',')))
        kinases.sort()
        train_site_to_kinase[site_identifier] = kinases

    duplicate_rows = 0
    for row in sites_in_test:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = list(set(kinases.split(',')))
        kinases.sort()
        if site_identifier in train_site_to_kinase:
            if train_site_to_kinase[site_identifier] == kinases:
                print("ERORR : DUPLICATE ROW!")
                duplicate_rows += 1

    if duplicate_rows == 0:
        print(f'PASSED TEST : No duplicate rows in train and {"test" if train_test else "validation"}')
    else:
        print(f'FAILED TEST : In total {duplicate_rows} many duplicate rows between train and {"test" if train_test else "validation"}')

def check_whether_unseen_test_data_exists_in_train(train_test_split, setup):
    all_train_rows = get_train_rows(train_test_split, setup)
    all_train_kinases = get_train_kinases(train_test_split, setup)
    all_test_kinases = get_test_kinases_according_to_split(train_test_split, setup)
    unseen_test_kinases = all_test_kinases - all_train_kinases
    # another option --> all_train_rows = get_train_rows_from_subfolder(train_test_split, setup)
    error = False
    for row in all_train_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        if len(kinases.intersection(unseen_test_kinases)) > 0:
            print(kinases.intersection(unseen_test_kinases))
            print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: Unseen Kinase appears in {"train" if train_test_split else "validation"} rows')
            error = True
            break
    if not error:
        print(f'PASSED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: Unseen Kinase does not appear in {"train" if train_test_split else "validation"} rows')

def check_for_duplicate_rows(train_test_split, setup):
    train_rows = get_train_rows(train_test_split, setup)
    test_rows = get_test_rows_according_to_split(train_test_split, setup)

    train_site_to_kinases = dict()
    for row in train_rows:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        train_site_to_kinases[site_identifier] = kinases

    error = False
    for row in test_rows:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        if site_identifier in train_site_to_kinases:
            if train_site_to_kinases[site_identifier] == kinases:
                error = True
                break

    if error:
        print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: There are duplicate rows between train and {"test" if train_test_split else "validation"}')
    else:
        print(f'PASSED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: No duplicate rows between train and {"test" if train_test_split else "validation"}')

def check_site_to_label_correctness(train_test_split, setup):
    all_kinases = get_all_kinases_from_kinase_dataset()
    original_dataset = CLEAN_KINASE_SUBSTRATE_DATASET
    df_original = pd.read_csv(original_dataset)
    kinase_to_site = {kinase: set() for kinase in all_kinases}
    for index, row in df_original.iterrows():
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinase = row["KIN_ACC_ID"]
        kinase_to_site[kinase].add(site_identifier)

    train_rows = get_train_rows(train_test_split, setup)
    for row in train_rows:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if site_identifier not in kinase_to_site[kinase]:
                print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: There is a label mismatch in Train {kinase} with site {site_identifier}')
                return

    test_rows = get_test_rows_according_to_split(train_test_split, setup)
    for row in test_rows:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if site_identifier not in kinase_to_site[kinase]:
                print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: There is a label mismatch in {"test" if train_test_split else "validation"} {kinase} with site {site_identifier}')
                return

    print(f'PASSED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: All labels are correct in train and {"test" if train_test_split else "validation"}')

def check_whether_seen_kinases_have_data_in_train(train_test, setup):
    train_rows = get_train_rows(train_test, setup)
    train_kinases = get_train_kinases(train_test, setup)
    test_kinases = get_test_kinases_according_to_split(train_test, setup)

    seen_kinases = train_kinases.intersection(test_kinases)
    seen_kinases_count_in_train = {kinase: 0 for kinase in seen_kinases}

    for row in train_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if kinase in seen_kinases:
                seen_kinases_count_in_train[kinase] += 1

    errored_kinases = {k: v for k, v in seen_kinases_count_in_train.items() if v == 0}
    if len(errored_kinases):
        print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test else "train-validation"} ::: Some seen kinases do not have data inside train {errored_kinases.keys()}')
    else:
        print(f'PASSED TEST : Setup :: {setup} Split :: {"train-test" if train_test else "train-validation"} ::: All seen kinases have data inside train')


def check_kinase_with_less_data_in_test(train_test_split, setup, threshold):
    all_test_rows = get_test_rows_according_to_split(train_test_split, setup)
    test_kinases = get_test_kinases_according_to_split(train_test_split, setup)

    kinase_to_site = get_kinase_to_sites_which_its_been_trained_on(train_test_split, setup)
    kinase_to_test_data_count = dict()

    for row in all_test_rows:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if kinase in test_kinases and kinase in kinase_to_site and site_identifier not in kinase_to_site[kinase]:
                if kinase not in kinase_to_test_data_count:
                    kinase_to_test_data_count[kinase] = 0
                kinase_to_test_data_count[kinase] += 1

    filtered = {key: value for key, value in kinase_to_test_data_count.items() if value < threshold}
    if len(filtered) > 0:
        print(filtered)
        print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: There are {len(filtered)} many kinases with few data but still in {"test" if train_test_split else "validation"}!')
    else:
        print(f'PASSED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: All kinases in test have more data than the {"test" if train_test_split else "validation"} threshold')

def check_whether_similar_kinases_are_all_in_same_dataset(train_test_split, setup, similarity_percentage=90):
    '''
    So here there is kind of problematic situation, because when we add a kinase to train,
    the rows which contain these sites but were put to test seem problematic. However the
    key point here is that we will not evaluate on these train kinases, but we also should
    not remove these train kinasess from those test rows, since if the model preidcts that kinase
    then we should not create an error for the model.
    '''
    all_train_rows = get_train_rows(train_test_split, setup)
    all_test_rows = get_test_rows_according_to_split(train_test_split, setup)

    train_kinases = get_train_kinases(train_test_split, setup)
    test_kinases = get_test_kinases_according_to_split(train_test_split, setup)

    # Only deal with kinases we have in hand, we dont care about the ones which dont exist in our dataset.
    similar_kinases_dict = find_kinases_with_high_sequence_similarity(kinase_similarity_percent=similarity_percentage)
    current_kinases = train_kinases.union(test_kinases)
    similar_kinases_dict = {count : similar_kinases for count, similar_kinases in similar_kinases_dict.items() if len(similar_kinases - current_kinases) == 0} # by adding this check we make sure that

    all_kinases = set()
    for kinases_set in similar_kinases_dict.values():
        all_kinases = all_kinases.union(kinases_set)

    trained_sites__to__kinases = dict()
    train_count = dict()
    for row in all_train_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        trained_sites__to__kinases[site_identifier] = kinases
        for kinase in kinases:
            if kinase in train_kinases:
                if kinase not in train_count:
                    train_count[kinase] = 0
                train_count[kinase] += 1

    test_count = dict()
    for row in all_test_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        for kinase in kinases:
            if kinase in test_kinases:
                if site_identifier in trained_sites__to__kinases and kinase in trained_sites__to__kinases[site_identifier]:
                    # So if this kinase has been trained on this site, just skip it
                    continue
                if kinase not in test_count:
                    test_count[kinase] = 0
                test_count[kinase] += 1

    for count, similar_kinases in similar_kinases_dict.items():
        kinase_in = {kinase: set() for kinase in similar_kinases}
        for kinase in similar_kinases:
            if kinase in test_count and test_count[kinase] > 0:
                kinase_in[kinase].add("test")
            if kinase in train_count and train_count[kinase] > 0:
                kinase_in[kinase].add("train")

        error = not are_there_contradicting_datasets(kinase_in)
        if error:
            print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: Kinases with high similarities do not exist in the same dataset : {similar_kinases}')
            return
    print(f'PASSED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: Kinases with high similarities exist in the same dataset ({"Test" if train_test_split else "Validation"})')

def check_kinase_similarity_within_the_dataset(train_test_split, setup, similarity_percentage=90):
    train_kinases = get_train_kinases(train_test_split, setup)
    validation_kinases = get_validation_kinases(setup)
    test_kinases = get_test_kinases(setup)

    similar_kinases_dict = find_kinases_with_high_sequence_similarity(kinase_similarity_percent=similarity_percentage)
    current_kinases = train_kinases.union(test_kinases)
    current_kinases = current_kinases.union(validation_kinases)
    similar_kinases_dict = {count: similar_kinases for count, similar_kinases in similar_kinases_dict.items() if
                            len(similar_kinases - current_kinases) == 0}  # by adding this check we make sure that

    for count, similar_kinases in similar_kinases_dict.items():
        if train_test_split:
            if (len(similar_kinases.intersection(train_kinases)) > 0 and len(similar_kinases.intersection(test_kinases))) or (len(similar_kinases.intersection(validation_kinases)) > 0 and len(similar_kinases.intersection(test_kinases))):
                print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: (2nd test) Kinases with high similarities do not exist in the same dataset : {similar_kinases}')
                return
        else:
            if (len(similar_kinases.intersection(train_kinases)) > 0 and len(similar_kinases.intersection(test_kinases))) or (len(similar_kinases.intersection(validation_kinases)) > 0 and len(similar_kinases.intersection(test_kinases))) or (len(similar_kinases.intersection(train_kinases)) > 0 and len(similar_kinases.intersection(validation_kinases))):
                print(f'FAILED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: (2nd test) Kinases with high similarities do not exist in the same dataset : {similar_kinases}')
                return

    print(f'PASSED TEST : Setup :: {setup} Split :: {"train-test" if train_test_split else "train-validation"} ::: (2nd test) Kinases with high similarities exist in the same dataset ({"Test" if train_test_split else "Validation"})')

def check_kinase_occurence_and_original_count_match(setup):
    # I want able to apply this check on the train-validation split since I do not know thw exact
    # count of the kinase occurences when we remove the test daatset. However since we know the
    # exact count of the kinases from the original dataset, we will be able to check whether the
    # kinase counts match the original dataset in the final train-test split.

    all_kinases = get_all_kinases_from_kinase_dataset()
    kinase_to_original_occurrence = get_kinase_occurrence_in_the_dataset()

    train_rows = get_train_rows(train_test=True, setup=setup)
    test_rows = get_test_rows_according_to_split(train_test=True, setup=setup)

    train_kinases = get_train_kinases(train_test=True, setup=setup)
    test_kinases = get_test_kinases_according_to_split(train_test=True, setup=setup)

    seen_kinases = train_kinases.intersection(test_kinases)
    unseen_kinases = test_kinases - train_kinases

    kinase_to_occurence = {kinase: 0 for kinase in all_kinases}
    for row in train_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if kinase in train_kinases:
                kinase_to_occurence[kinase] += 1

    kinase_to_sites_its_been_trained_on = get_kinase_to_sites_which_its_been_trained_on(train_test_split=True,
                                                                                        setup=setup)
    for row in test_rows:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if kinase in test_kinases:
                if kinase in kinase_to_sites_its_been_trained_on and site_identifier not in \
                        kinase_to_sites_its_been_trained_on[kinase]:
                    kinase_to_occurence[kinase] += 1
                elif kinase not in kinase_to_sites_its_been_trained_on:
                    kinase_to_occurence[kinase] += 1

    errored_kinases = set()
    for kinase, count_in_dataset in kinase_to_occurence.items():
        original_occurence = kinase_to_original_occurrence[kinase]
        dataset_occurence = kinase_to_occurence[kinase]
        if original_occurence != dataset_occurence:
            errored_kinases.add(kinase)

    # For the ZSL setup, the rows in the test dataset is used only to evaluate unseen kinases. So if a row contains
    # an unseen kinase, even if it also contains train kinases, these train kinases will be removed from that specific
    # row from the test dataset. This is so that we could fit the ZSL setup.
    if (len(errored_kinases) > 0 and setup == "GZSL") \
            or (len(errored_kinases) > 0 and setup == "ZSL" and not errored_kinases.issubset(train_kinases)):
        print(f'FAILED TEST : Setup :: {setup} ::: The original data count of kinase doesn\'t match its occurence in our dataset split')
    else:
        print(f'PASSED TEST : Setup :: {setup} ::: All original data counts of kinases match their occurence in our dataset split')

def seen_kinases_should_have_sites_it_hasnt_been_trained_on_in_test(train_test):
    # This is only for the GZSL setup since we only care about the seen kinases
    test_rows = get_test_rows_according_to_split(train_test=train_test, setup="GZSL")
    train_kinases = get_train_kinases(train_test=train_test, setup="GZSL")
    test_kinases = get_test_kinases_according_to_split(train_test=train_test, setup="GZSL")
    seen_kinases = train_kinases.intersection(test_kinases)
    additional_site_count_for_seen_kinases = {kinase: 0 for kinase in seen_kinases}
    kinase_to_sites_its_been_trained_on = get_kinase_to_sites_which_its_been_trained_on(train_test_split=train_test,
                                                                                        setup="GZSL")

    for row in test_rows:
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        for kinase in kinases:
            if kinase in seen_kinases and site_identifier not in kinase_to_sites_its_been_trained_on[kinase]:
                additional_site_count_for_seen_kinases[kinase] += 1

    filtered = {k: v for k, v in additional_site_count_for_seen_kinases.items() if v == 0}
    if len(filtered) > 0:
        print(f'FAILED TEST : Split :: {"train-test" if train_test else "train-validation"} ::: There are seen kinases which dont have additional sites in test! {filtered}')
    else:
        print(f'PASSED TEST : Split :: {"train-test" if train_test else "train-validation"} ::: All seen kinases have sites they havent been trained on inside the test daatset.')

def ZSL_test_rows_shouldnt_contain_any_train_kinases(train_test):
    # This check is only for the ZSL setup
    train_kinases = get_train_kinases(train_test=train_test, setup="ZSL")
    test_rows = get_test_rows_according_to_split(train_test=train_test, setup="ZSL")
    for row in test_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        if len(kinases.intersection(train_kinases)) > 0:
            print(f'FAILED TEST : setup : ZSL, Split :: {"train-test" if train_test else "train-validation"} ::: test rows contain train kinases : {kinases.intersection(train_kinases)}')
            return
    print(f'PASSED TEST : setup : ZSL, Split :: {"train-test" if train_test else "train-validation"} ::: test rows do not contain train kinases')

def create_folder_if_doesnt_exist(path):
    directory_path = f"{path}"
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

def check_random_seed_stability(train_kinase_to_count, validation_kinase_to_count, test_kinase_to_count):
    create_folder_if_doesnt_exist("pickle_files")

    if not os.path.exists(f'pickle_files/train_kinase_to_count_{RANDOM_SEED}.pkl'):
        with open(f'pickle_files/train_kinase_to_count_{RANDOM_SEED}.pkl', 'wb') as f:
            pickle.dump(train_kinase_to_count, f)
    else:
        with open(f'pickle_files/train_kinase_to_count_{RANDOM_SEED}.pkl', 'rb') as f:
            loaded_object = pickle.load(f)
            print(f'TRAIN_KINASE_TO_COUNT IS THE SAME : {train_kinase_to_count == loaded_object}')

    if not os.path.exists(f'pickle_files/validation_kinase_to_count_{RANDOM_SEED}.pkl'):
        with open(f'pickle_files/validation_kinase_to_count_{RANDOM_SEED}.pkl', 'wb') as f:
            pickle.dump(validation_kinase_to_count, f)
    else:
        with open(f'pickle_files/validation_kinase_to_count_{RANDOM_SEED}.pkl', 'rb') as f:
            loaded_object = pickle.load(f)
            print(f'VALIDATION_KINASE_TO_COUNT IS THE SAME : {validation_kinase_to_count == loaded_object}')

    if not os.path.exists(f'pickle_files/test_kinase_to_count_{RANDOM_SEED}.pkl'):
        with open(f'pickle_files/test_kinase_to_count_{RANDOM_SEED}.pkl', 'wb') as f:
            pickle.dump(test_kinase_to_count, f)
    else:
        with open(f'pickle_files/test_kinase_to_count_{RANDOM_SEED}.pkl', 'rb') as f:
            loaded_object = pickle.load(f)
            print(f'TEST_KINASE_TO_COUNT IS THE SAME : {test_kinase_to_count == loaded_object}')

#####################################################################################
#####################################################################################
#####################################################################################

def get_kinase_to_ideal_dataset(kinase_count_test_threshold, train_kinase_to_count, test_kinase_to_count, kinase_similarity_percent):
    similar_kinases_dict = find_kinases_with_high_sequence_similarity(kinase_similarity_percent=kinase_similarity_percent)
    # Here we should only deal with the kinases in hand, so we should only look into the similarity of kinases
    # who we are currently dealing with. For example we could have a scenario like this:
    # Lets say kinase1 and kinase2 are similar, and in the train test split, we have decided that these kinases should
    # be placed inside test.
    # Then when we are doing the train validation split, these similar kinases are gonna come again. But they actually
    # should be inside the test dataset. (Actually when placing in the kinases to the datasets, we are kind of handling this by checking
    # whether they exist inside the train and validation dictionaries, but I will still add this filter over here.)
    current_kinases = set(train_kinase_to_count.keys())
    current_kinases = current_kinases.union(set(test_kinase_to_count.keys()))
    similar_kinases_dict = {count : similar_kinases for count, similar_kinases in similar_kinases_dict.items() if len(similar_kinases - current_kinases) == 0} # by adding this check we make sure that
                                                                                                                                                                # the kinases whose similarity we're looking at
                                                                                                                                                                # all should be one of the current kinases.

    _, kinase_to_groups = map_kinase_family_group_info()
    kinase_to_occurrence = get_kinase_occurrence_in_the_dataset()
    kinase_to_ideal_dataset = dict()
    remaining_similar_kinases = dict()
    for count, similar_kinases in similar_kinases_dict.items():
        to_train = False
        for kinase in similar_kinases:
            # if even one of the kinases in this group has smaller data than the kinase_count_test_threshold
            # then send all of them to train
            kinase_data_count = kinase_to_occurrence[kinase]
            if kinase_data_count < kinase_count_test_threshold and kinase_data_count != 0:  # I added this not equal to 0 case, because if lets say we have smth like this:
                # kinas1 : 0, kinase2 : 100, and if these are similar kinases, kinase1 having no data
                # shouldnt affect where kinase2 goes.
                to_train = True
        if to_train:
            for kinase in similar_kinases:
                kinase_to_ideal_dataset[kinase] = "train"
        else:
            remaining_similar_kinases[count] = similar_kinases

    # Now for the remaining similar kinases, the kinases similar to themselves, could either all be in train,
    # or could either all be in test. This will be selected in random.
    choices = ["test", "train"]
    probabilities = [0.4, 0.6]  # 40% for "test" and 60% for "train"
    for count, similar_kinases in remaining_similar_kinases.items():
        selected_choice = random.choices(choices, weights=probabilities, k=1)[0]
        for kinase in similar_kinases:
            kinase_to_ideal_dataset[kinase] = selected_choice

    return kinase_to_ideal_dataset

def place_similar_kinases_into_the_same_dataset(train_kinase_to_count, test_kinase_to_count, kinase_count_test_threshold, kinase_similarity_percent):
    kinase_to_dataset = get_kinase_to_ideal_dataset(kinase_count_test_threshold, train_kinase_to_count, test_kinase_to_count, kinase_similarity_percent)
    for kinase in kinase_to_dataset:
        into_dataset = kinase_to_dataset[kinase]
        tot_count_for_kinase = 0
        if kinase in train_kinase_to_count:
            tot_count_for_kinase += train_kinase_to_count[kinase]
            del train_kinase_to_count[kinase]
        if kinase in test_kinase_to_count:
            tot_count_for_kinase += test_kinase_to_count[kinase]
            del test_kinase_to_count[kinase]
        if tot_count_for_kinase != 0:  # If this kinase has previously been added to test, then this kinase will have 0 data and should be deleted.
                                        # Because the executes this part for both train-test and train-validation splits, so when the code is doing the
                                        # train-validation split, test_dict is actually going to be valid_dict.
            # even though it says "train" - "test", for when doing the train validation split, the test dataset
            # is actually the validation dataset, so np
            if into_dataset == "train":
                train_kinase_to_count[kinase] = tot_count_for_kinase
            elif into_dataset == "test":
                test_kinase_to_count[kinase] = tot_count_for_kinase

    calculated_train = set(train_kinase_to_count.keys())
    calculated_test = set(test_kinase_to_count.keys())
    calculated_seen = calculated_train.intersection(calculated_test)
    calculated_unseen = calculated_test - calculated_train
    train_kinases, seen_test_kinases, unseen_test_kinases = calculated_train, calculated_seen, calculated_unseen
    return train_kinases, seen_test_kinases, unseen_test_kinases

def distribute_kinases_wrt_group(train_kinase_to_count,
                                 test_kinase_to_count,
                                 split_into_dataset,
                                 kinase_count_test_threshold,
                                 test_gzsl_percentage,
                                 group_to_kinase_dict,
                                 kinase_groups,
                                 group_to_dataset_count,
                                 kinase_similarity_percent,
                                 take_seq_sim_into_consideration):

    train_kinases = set()
    seen_test_kinases = set()
    unseen_test_kinases = set()

    # 1. Place all similar kinases into the same dataset if take_seq_sim_into_consideration is True
    # We want to make sure that the kinases which have sequence similarity above a threshold are placed
    # in the same dataset. The reason for this is because sequence similarity is very important when representing
    # kinases, and if one of two similar kinases exist in test and the other exists in train, this will be an
    # easy kinase class to test. Because the model has already seen a very similar kinase while training.

    if take_seq_sim_into_consideration:
        place_kinases_with_high_similarity_into_the_same_dataset__group_wise_division(group_to_kinase_dict, group_to_dataset_count, kinase_similarity_percent,
                                                                                      split_into_dataset, kinase_count_test_threshold, train_kinase_to_count,
                                                                                      test_kinase_to_count, train_kinases, unseen_test_kinases)

        remove_already_placed_kinases_group(group_to_kinase_dict, train_kinase_to_count, test_kinase_to_count)

    # 2. place all kinase counts less than the test threshold inside train dataset

    remaining_groups_to_kinase = {group: dict() for group in kinase_groups}
    for group, kinase_dict in group_to_kinase_dict.items():
        for kinase_uniprot, count in kinase_dict.items():
            if count < kinase_count_test_threshold:
                train_kinase_to_count[kinase_uniprot] = count
                train_kinases.add(kinase_uniprot)
            else:
                remaining_groups_to_kinase[group][kinase_uniprot] = count

    # 3. Shuffle the kinases in each group, and randomly start selecting from them.

    for group in remaining_groups_to_kinase:
        kinases = list(set(remaining_groups_to_kinase[group].keys()))
        kinases.sort()
        random.shuffle(kinases)
        shuffled_kinase_dict = {kinase: remaining_groups_to_kinase[group][kinase] for kinase in kinases}
        remaining_groups_to_kinase[group] = shuffled_kinase_dict

    # 4. Now while traversing through each of the remaining data, find how many kinase count is left,
    #    calculate its stratify percentage which is the amount of data which has to be put into the test
    #    set from each group as unseen kinases. And obvy place that amount into the test set, but this part
    #    is like 0-1 knapsack, you either put all of them or none of that kinase.

    for group, kinase_dict in remaining_groups_to_kinase.items():
        count_to_add_to_set = group_to_dataset_count[group][f'unseen_{split_into_dataset}']  # this could either be test or valid
        current_count = 0
        keys_to_delete = []
        for k, v in kinase_dict.items():
            if current_count + v <= count_to_add_to_set:
                test_kinase_to_count[k] = v
                current_count += v
                unseen_test_kinases.add(k)
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del kinase_dict[k]

    # 5. Now finally, the remaining kinases will be the seen classes. So since this a GZSL setup,
    #    some portion will be in the training set some will be in the test set.

    seen_kinase = 0
    for group, kinase_dict in remaining_groups_to_kinase.items():
        for k, v in kinase_dict.items():
            count_to_add_to_test = int(np.floor(v * test_gzsl_percentage))
            if count_to_add_to_test >= kinase_count_test_threshold:
                test_kinase_to_count[k] = count_to_add_to_test
                train_kinase_to_count[k] = v - count_to_add_to_test
                train_kinases.add(k)
                seen_test_kinases.add(k)
                seen_kinase += 1
            else:
                train_kinase_to_count[k] = v
                train_kinases.add(k)

    return train_kinases, seen_test_kinases, unseen_test_kinases


def remove_already_placed_kinases_group(group_to_kinase_dict, train_kinase_to_count, test_kinase_to_count):
    # get the kinase that were placed inside train or test and remove them from remaining_groups_to_kinase
    # so we wouldn't be adding them twice.
    kinases_added_to_train = set(train_kinase_to_count.keys())
    kinases_added_to_test = set(test_kinase_to_count.keys())
    already_handled_kinases = kinases_added_to_train.union(kinases_added_to_test)
    for group, kinase_dict in group_to_kinase_dict.items():
        keys_to_delete = []
        for kinase_uniprot, count in kinase_dict.items():
            if kinase_uniprot in already_handled_kinases:
                keys_to_delete.append(kinase_uniprot)
        for k in keys_to_delete:
            del kinase_dict[k]

def remove_already_placed_kinases(kinase_to_occurence, train_kinase_to_count, test_kinase_to_count):
    # get the kinase that were placed inside train or test and remove them from remaining_groups_to_kinase
    # so we wouldn't be adding them twice.
    kinases_added_to_train = set(train_kinase_to_count.keys())
    kinases_added_to_test = set(test_kinase_to_count.keys())
    already_handled_kinases = kinases_added_to_train.union(kinases_added_to_test)
    keys_to_delete = []
    for kinase, count in kinase_to_occurence.items():
        if kinase in already_handled_kinases:
            keys_to_delete.append(kinase)
    for k in keys_to_delete:
        del kinase_to_occurence[k]

def place_kinases_with_high_similarity_into_the_same_dataset__group_wise_division(group_to_kinase_dict, group_to_dataset_count, kinase_similarity_percent,
                                                                                  split_into_dataset, kinase_count_test_threshold, train_kinase_to_count,
                                                                                  test_kinase_to_count, train_kinases, unseen_test_kinases):
    # Handle putting similar kinases inside the same dataset. Here are the steps:
    # 1. Traverse through the similar kinase sets and even if one of them has data count less than the test/valiation threshld,
    #       place all of theose similar kinases inside the train dataset
    # 2. Then randomly select a dataset for all of those kinases, while adding these kinases to that dataset, if it is test,
    #       reduce the number of kinases that have to be added into test from that group (Updating group_to_kinase_dict).
    similar_kinases_dict = find_kinases_with_high_sequence_similarity(
        kinase_similarity_percent=kinase_similarity_percent)
    cur_kinases = set()
    for group, kinase_dict in group_to_kinase_dict.items():
        cur_kinases = cur_kinases.union(set(kinase_dict.keys()))
    similar_kinases_dict = {count: similar_kinases for count, similar_kinases in similar_kinases_dict.items() if
                            len(similar_kinases - cur_kinases) == 0}    # by adding this check we make sure that
                                                                        # the kinases whose similarity we're looking at
                                                                        # all should be one of the current kinases.

    _, kinase_to_groups = map_kinase_family_group_info()
    remaining_similar_kinases = dict()
    for count, similar_kinases in similar_kinases_dict.items():
        to_train = False
        for kinase in similar_kinases:
            # if even one of the kinases in this group has smaller data than the kinase_count_test_threshold
            # then send all of them to train
            group = kinase_to_groups[kinase]
            kinase_data_count = group_to_kinase_dict[group][kinase]
            if kinase_data_count < kinase_count_test_threshold and kinase_data_count != 0:  # I added this not equal to 0 case, because if lets say we have smth like this:
                # kinas1 : 0, kinase2 : 100, and if these are similar kinases, kinase1 having no data
                # shouldnt affect where kinase2 goes.
                to_train = True
        if to_train:
            for kinase in similar_kinases:
                group = kinase_to_groups[kinase]
                kinase_count = group_to_kinase_dict[group][kinase]
                train_kinase_to_count[kinase] = kinase_count
                train_kinases.add(kinase)
        else:
            remaining_similar_kinases[count] = similar_kinases
    # Now for the remaining similar kinases, the kinases similar to themselves, could either all be in train,
    # or could either all be in test. This will be selected in random.
    choices = ["test", "train"]
    probabilities = [0.6, 0.4]  # 60% for "test" and 40% for "train"
    for count, similar_kinases in remaining_similar_kinases.items():
        selected_choice = random.choices(choices, weights=probabilities, k=1)[0]

        # check its availability to enter into test
        if selected_choice == "test":
            group_to_count = dict()
            for kinase in similar_kinases:
                group = kinase_to_groups[kinase]
                kinase_count = group_to_kinase_dict[group][kinase]
                if group not in group_to_count:
                    group_to_count[group] = 0
                group_to_count[group] += kinase_count
            # now check whether the count that is wanted to be added to this group is valid (should not exceed
            # the unseen_test count defiend beforehand)
            for group, total_addition in group_to_count.items():
                unseen_row_count_to_add = group_to_dataset_count[group][f'unseen_{split_into_dataset}']
                # this means that the data that is wanted to be added to one of the groups is exceeding the data
                # calculated inside group_to_dataset_count.
                if unseen_row_count_to_add < total_addition:
                    selected_choice = "train"
                    break

        if selected_choice == "train":
            for kinase in similar_kinases:
                group = kinase_to_groups[kinase]
                train_kinase_to_count[kinase] = group_to_kinase_dict[group][kinase]
                train_kinases.add(kinase)
        else:
            for kinase in similar_kinases:
                group = kinase_to_groups[kinase]
                data_count = group_to_kinase_dict[group][kinase]
                if data_count >= kinase_count_test_threshold:
                    test_kinase_to_count[kinase] = data_count
                    unseen_test_kinases.add(kinase)
                    group_to_dataset_count[group][f'unseen_{split_into_dataset}'] -= data_count

def place_kinases_with_high_similarity_into_the_same_dataset(kinase_to_occurence, total_unseen_test_data, kinase_similarity_percent,
                                                             split_into_dataset, kinase_count_test_threshold, train_kinase_to_count,
                                                             test_kinase_to_count, train_kinases, unseen_test_kinases):
    # Handle putting similar kinases inside the same dataset. Here are the steps:
    # 1. Traverse through the similar kinase sets ans even if one of them has data count less than the tes/valiation threshld,
    #       place all of theose similar kinases inside the trin dataset
    # 2. Then randomly select a dataset for all of those kinases, while adding these kinases to that dataset, if it is test,
    #       reduce the number of kinases that have to be added into test from that group (Updating group_to_kinase_dict).
    similar_kinases_dict = find_kinases_with_high_sequence_similarity(kinase_similarity_percent=kinase_similarity_percent)
    cur_kinases = set(kinase_to_occurence.keys())
    similar_kinases_dict = {count: similar_kinases for count, similar_kinases in similar_kinases_dict.items() if
                            len(similar_kinases - cur_kinases) == 0}    # by adding this check we make sure that
                                                                        # the kinases whose similarity we're looking at
                                                                        # all should be one of the current kinases.

    _, kinase_to_groups = map_kinase_family_group_info()
    remaining_similar_kinases = dict()
    for count, similar_kinases in similar_kinases_dict.items():
        to_train = False
        for kinase in similar_kinases:
            # if even one of the kinases in this group has smaller data than the kinase_count_test_threshold
            # then send all of them to train
            kinase_data_count = kinase_to_occurence[kinase]
            if kinase_data_count < kinase_count_test_threshold and kinase_data_count != 0:  # I added this not equal to 0 case, because if lets say we have smth like this:
                # kinas1 : 0, kinase2 : 100, and if these are similar kinases, kinase1 having no data
                # shouldnt affect where kinase2 goes.
                to_train = True
        if to_train:
            for kinase in similar_kinases:
                kinase_count = kinase_to_occurence[kinase]
                train_kinase_to_count[kinase] = kinase_count
                train_kinases.add(kinase)
        else:
            remaining_similar_kinases[count] = similar_kinases
    # Now for the remaining similar kinases, the kinases similar to themselves, could either all be in train,
    # or could either all be in test. This will be selected in random.
    choices = ["test", "train"]
    probabilities = [0.6, 0.4]  # 60% for "test" and 40% for "train"
    for count, similar_kinases in remaining_similar_kinases.items():
        selected_choice = random.choices(choices, weights=probabilities, k=1)[0]

        # check its availability to enter into test
        if selected_choice == "test":
            tot_count_to_add = 0
            for kinase in similar_kinases:
                tot_count_to_add += kinase_to_occurence[kinase]
            if tot_count_to_add > total_unseen_test_data:
                selected_choice = "train"

        if selected_choice == "train":
            for kinase in similar_kinases:
                train_kinase_to_count[kinase] = kinase_to_occurence[kinase]
                train_kinases.add(kinase)
        else:
            for kinase in similar_kinases:
                data_count = kinase_to_occurence[kinase]
                test_kinase_to_count[kinase] = data_count
                unseen_test_kinases.add(kinase)
                total_unseen_test_data -= data_count

def distribute_kinases(train_kinase_to_count,
                       test_kinase_to_count,
                       split_into_dataset,
                       kinase_count_test_threshold,
                       test_gzsl_percentage,
                       kinase_to_occurrence,
                       total_unseen_test_data,
                       kinase_similarity_percent,
                       take_seq_sim_into_consideration):


    train_kinases = set()
    seen_test_kinases = set()
    unseen_test_kinases = set()

    # 1. Place all similar kinases into the same dataset if take_seq_sim_into_consideration is True
    # We want to make sure that the kinases which have sequence similarity above a threshold are placed
    # in the same dataset. The reason for this is because sequence similarity is very important when representing
    # kinases, and if one of two similar kinases exist in test and the other exists in train, this will be an
    # easy kinase class to test. Because the model has already seen a very similar kinase while training.

    if take_seq_sim_into_consideration:
        place_kinases_with_high_similarity_into_the_same_dataset(kinase_to_occurrence, total_unseen_test_data, kinase_similarity_percent,
                                                                 split_into_dataset, kinase_count_test_threshold, train_kinase_to_count,
                                                                 test_kinase_to_count, train_kinases, unseen_test_kinases)

        remove_already_placed_kinases(kinase_to_occurrence, train_kinase_to_count, test_kinase_to_count)

    # 2. place all kinase counts less than the test threshold inside the train dataset

    remaining_kinase_to_occurence = dict()
    for kinase, count in kinase_to_occurrence.items():
        if count < kinase_count_test_threshold:
            train_kinase_to_count[kinase] = count
            train_kinases.add(kinase)
        else:
            remaining_kinase_to_occurence[kinase] = count

    # 3. Shuffle the kinases
    kinases = list(set(remaining_kinase_to_occurence.keys()))
    kinases.sort()
    random.shuffle(kinases)
    remaining_kinase_to_occurence = {kinase: remaining_kinase_to_occurence[kinase] for kinase in kinases}


    # 4. Now while traversing through each of the remaining data, find how many kinase count is left,
    #    calculate its stratify percentage which is the amount of data which has to be put into the test
    #    set as unseen kinases. And obvy place that amount into the test set, but this part
    #    is like 0-1 knapsack, you either put all of them or none of that kinase.
    unseen_kinases = 0
    current_unseen_data_count = 0
    keys_to_delete = []
    for k, v in remaining_kinase_to_occurence.items():
        if current_unseen_data_count + v <= total_unseen_test_data:
            test_kinase_to_count[k] = v
            current_unseen_data_count += v
            unseen_test_kinases.add(k)
            keys_to_delete.append(k)
            unseen_kinases += 1
    for k in keys_to_delete:
        del remaining_kinase_to_occurence[k]


    # 5. Now finally, the remaining kinases will be the seen classes. So since this a GZSL setup,
    #    some portion will be in the training set some will be in the test set.

    seen_kinase = 0
    for k, v in remaining_kinase_to_occurence.items():
        count_to_add_to_test = int(np.floor(v * test_gzsl_percentage))
        if count_to_add_to_test >= kinase_count_test_threshold:
            test_kinase_to_count[k] = count_to_add_to_test
            train_kinase_to_count[k] = v - count_to_add_to_test
            train_kinases.add(k)
            seen_test_kinases.add(k)
            seen_kinase += 1
        else:
            train_kinase_to_count[k] = v
            train_kinases.add(k)

    return train_kinases, seen_test_kinases, unseen_test_kinases

# train_kinase_to_count, test_kinase_to_count, train_kinases, seen_test_kinases, unseen_test_kinases,
#                                               all_data_rows, is_validation=False
def distribute_dataset_rows_into_datasets(train_kinase_to_count, test_kinase_to_count,
                                          train_kinases, seen_test_kinases, unseen_test_kinases, all_data_rows):

    # This is added due to keeping track of how many ros odata from a kinase has been added to the test dataset,
    # The same thing isn't being applied to train, since after separating the test data, the rest could be added to train
    remaining_kinases_to_add_to_test = copy.deepcopy(test_kinase_to_count)

    test_kinases = set(test_kinase_to_count.keys())
    only_train_kinases = train_kinases - test_kinases

    # These are going to hold the dataset rows from MULTI_CLASS_KINASE_SUBSTRATE_DATASET_FILE
    # So rows like,
    # SUB_ACC_ID,SUB_MOD_RSD,SITE_+/-7_AA,KINASE_ACC_IDS
    # P68101,S52,MILLSELsRRRIRSI,"P19525,Q9BQI3"
    sites_in_train = list()
    sites_in_test = list()
    masked_rows = list()

    # 1. If a row contains train kinases, add that row into train, however if that row contains
    #    unseen kinases, mask the unseen kinases, i.e. remove the unseen kinases, and still add that
    #    row into train,
    # 2. If a row only contains test kinases add that into test,
    # 3. If a row contains both train and test kinases, and if the row does not contain unseen kinases,
    #    this means the kinases will not be masked. If we put this row into both train and test then there
    #    will be duplicate rows, so for these kind of rows we added them into an array called common_rows,
    #    we will handle them later on (Some will be in train and some will be in test)'

    common_rows = []
    add_to_train = False
    add_to_test = False
    for row in all_data_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))

        train_kinase_intersection = train_kinases.intersection(kinases)
        test_kinase_intersection = test_kinases.intersection(kinases)
        unseen_test_kinase_intersection = unseen_test_kinases.intersection(kinases)

        added_to_train_masked = False
        if len(train_kinase_intersection) > 0:
            new_row = copy.deepcopy(row)
            remove_unseen_kinases = list(kinases - unseen_test_kinase_intersection)
            remove_unseen_kinases.sort()
            new_row['KINASE_ACC_IDS'] = ",".join(remove_unseen_kinases)
            if kinases == set(remove_unseen_kinases): # this means that no kinases were removed, so there weren't any unseen test kinases
                add_to_train = True  # so it will be added without any changes.
            else:
                masked_rows.append([row, new_row])
                sites_in_train.append(new_row)  # Add the masked version, i.e. add the row with the unseen kinases removed.
                added_to_train_masked = True
        if len(test_kinase_intersection) > 0:
            add_to_test = True

        # this means that the same row could be added to both train or test. These rows will be handled later.
        if add_to_train and add_to_test:
            common_rows.append(row)
        elif add_to_train:  # then add_to_test must be False so directly add it to train
            sites_in_train.append(row)
        elif add_to_test: # So here we know that for sure this row has an unseen test kinase, but the masked version might
                          # have been added to train. So this is why this part works: when adding the masked version
                          # add_to_train wasn't set to True (Becuase this version of the rows shouldn't be added to
                          # train). However the version where the unseen test kinase exists should be added to test
                          #
                          # Another thing is that, the code will only enter this condition if add_to_train is False.
                          # And there are 2 possibiltiies for this:
                          # 1. add_train is False because there are no train kinases. So if there are no train kinases,
                          # this means that there are also no seen test kinases (Becasue train kinases is calculated like
                          # train_dict.keys() which also contains the seen test kinases.). So definietly there should be
                          # unseen test kinases here.
                          # 2. add_train is False because there were unseen_test kinases and this row was masked and added
                          # to train. So it definietly contains unseen test kinases.
            sites_in_test.append(row)  # WE'LL KEEP THE TRAIN KINASES IN THE TEST ROWS BUT WILL NOT
                                        # EVALUATE ON THEM!!! YES WE WILL NOT TEST ON THESE KINASES BUT
                                        # IF THE MODEL PREDICTS THESE IN EVALUATION WE ALSO CANNOT SAY
                                        # THAT IT IS WRONG!!!

            for kinase in kinases:
                if kinase in remaining_kinases_to_add_to_test:
                    if kinase in seen_test_kinases and added_to_train_masked:
                        # skip these because since the same site
                        # and kinase pair exists in train, we wont
                        # be able to evaluate on this. So dont subtract
                        # from this kinase's test data because it
                        # won't count as test data for this kinase.
                        continue
                    else:
                        # 2 possibilities:
                        # 1. The kinase is not a seen_test_kinase, so then it should be an unseen kinase (because it is
                        # also in remaining_kinases_to_add_to_test), then decrease its remaining data to add to test
                        # 2. Or the kinase might a seen_test_kinase, and since added_to_train_masked is False, this seen
                        # test kinase was not added to train, thus we could count this rows as this kinases test data.
                        remaining_kinases_to_add_to_test[kinase] -= 1

        add_to_test = False
        add_to_train = False

    # we removed the unseen kinases from the train row, so in these common rows we
    # should not be having any unseen kinase. (They are all in test)

    # first add all of the rows which have the train kinases with few data into train,
    # Since if they are only train kinases, this means that they have to have data below the test threshold,
    # so just place all of these kind of rows to train.
    rows_to_keep = []
    for row in common_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        if len(kinases.intersection(only_train_kinases)) > 0:
            sites_in_train.append(row)
        else:
            rows_to_keep.append(row)

    common_rows = rows_to_keep

    # now we know how many data has to be added from which kinase into test
    # so we'll do that (inside remaining_kinases_to_add_to_test)
    ###############################################################################
    ###############################################################################
    # So in here there are seen kinases and also train kinases. Because all rows which had unseen test kinases
    # are already added to inside the test rows.

    add_to_test = False
    for row in common_rows:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        intersection_with_seen_test = kinases.intersection(seen_test_kinases)
        for kinase in intersection_with_seen_test:
            if kinase in remaining_kinases_to_add_to_test and remaining_kinases_to_add_to_test[kinase] > 0:
                add_to_test = True
                break
        if add_to_test:
            for kinase in intersection_with_seen_test:
                if kinase in remaining_kinases_to_add_to_test:
                    remaining_kinases_to_add_to_test[kinase] -= 1
            sites_in_test.append(row)  # YES WE WILL NOT TEST ON THE TRAIN KINASES WHO ALSO EXIST IN THIS ROWS
                                        # BUT IF THE MODEL PREDICTS THESE IN EVALUATION WE ALSO CANNOT SAY
                                        # IT IS WRONG!!! SO WE'LL KEEP THE TRAIN KINASES IN TEST ROWS BUT
                                        # WILL NOT EVALUATE ON THEM!!!
            add_to_test = False
        else:
            sites_in_train.append(row)



    train_kinase_to_count_new = {kinase: 0 for kinase in train_kinase_to_count.keys()}
    trained_sites__to__kinases = dict()  # So this is going to keep track of the kinases which have been trained on this site
    for row in sites_in_train:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        trained_sites__to__kinases[site_identifier] = kinases
        for kinase in kinases:
            if kinase in train_kinase_to_count:
                train_kinase_to_count_new[kinase] += 1
            else:
                print(f'{kinase} ORIGINALLY NOT IN TRAIN_KINASE_TO_COUNT')
                print(f'{kinase in train_kinases} in train kinases')
                print(f'{kinase in seen_test_kinases} in seen test')
                print(f'{kinase in unseen_test_kinases} in unseen test')

    test_kinase_to_count_new = {kinase: 0 for kinase in test_kinase_to_count.keys()}
    for row in sites_in_test:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        site_identifier = f'{row["SUB_ACC_ID"]}|{row["SUB_MOD_RSD"]}|{row["SITE_+/-7_AA"]}'
        for kinase in kinases:
            if kinase in test_kinase_to_count:
                # trained_sites__to__kinases[site_identifier] keeps track of the kinases which are trained on this specific
                # site, so if it was already trained on this, do not count it as test data. --> or it would be data leak!
                if site_identifier in trained_sites__to__kinases:
                    if kinase in trained_sites__to__kinases[site_identifier] and kinase in seen_test_kinases:
                        continue
                test_kinase_to_count_new[kinase] += 1

    only_training_kinases_calculated = set(train_kinase_to_count_new.keys()) - set(test_kinase_to_count_new.keys())
    seen_test_kinases_calculated = set(train_kinase_to_count_new.keys()).intersection(set(test_kinase_to_count_new.keys()))
    unseen_test_kinases_calculated = set(test_kinase_to_count_new.keys()) - set(train_kinase_to_count_new.keys())

    print('#######################################################################')
    print(f'IN THE LAST STAGE')
    print(f'{len(train_kinase_to_count)} Training kinases')
    print(f'{len(only_training_kinases_calculated)} Only training kinases')
    print(f'{len(seen_test_kinases_calculated)} Seen test kinases')
    print(f'{len(unseen_test_kinases_calculated)} Unseen test kinases')
    print('#######################################################################')

    print('#######################################################################')
    seen_kinases_test_count = {key: value for key, value in test_kinase_to_count.items() if key in seen_test_kinases_calculated}
    unseen_kinases_test_count = {key: value for key, value in test_kinase_to_count.items() if key in unseen_test_kinases_calculated}
    print(f'Seen data : {len(seen_kinases_test_count.keys())} kinase : {sum(seen_kinases_test_count.values())}')
    print(f'Unseen data : {len(unseen_kinases_test_count.keys())} kinase : {sum(unseen_kinases_test_count.values())}')
    print('#######################################################################')

    return train_kinase_to_count_new, test_kinase_to_count_new, sites_in_train, sites_in_test, masked_rows

def check_dataframe_stability(dataframe_name, df, old_file_name):
    if os.path.exists(old_file_name):
        old_df = pd.read_csv(old_file_name)
        error = False
        if (len(old_df) != len(df)):
            error = True
        else:
            for i in range(len(df)):
                row1 = df.iloc[i]
                row2 = old_df.iloc[i]
                if not row1.equals(row2):
                    error = True
        if error:
            print(f'FAILED TEST : DATAFRAME IS NOT EQUAL WITH THE PREVIOUS VERSION : {dataframe_name} : {len(df)} != {len(old_df)}')
            return False
        else:
            # print(f'PASSED TEST : ALL ROWS ARE EQUAL WITH THE PREVIOUS VERSION WHEN CREATING NEW DATAFRAME : {dataframe_name}')
            return True

def merge_sites_in_train_and_validation(sites_in_train, sites_in_validation):
    all_sites_in_train = copy.deepcopy(sites_in_train)
    all_sites_in_validation = copy.deepcopy(sites_in_validation)
    train_rows = dict()
    for row in all_sites_in_train:
        site_identifier = f'{row["SUB_ACC_ID"]}_{row["SUB_MOD_RSD"]}_{row["SITE_+/-7_AA"]}'
        train_rows[site_identifier] = row
    for row in all_sites_in_validation:
        site_identifier = f'{row["SUB_ACC_ID"]}_{row["SUB_MOD_RSD"]}_{row["SITE_+/-7_AA"]}'
        if site_identifier in train_rows:
            row_in_train = train_rows[site_identifier]
            kinases_in_train = row_in_train["KINASE_ACC_IDS"]
            kinases_in_train = set(kinases_in_train.split(','))

            kinases_in_valid = row["KINASE_ACC_IDS"]
            kinases_in_valid = set(kinases_in_valid.split(','))

            kinases_in_train = list(kinases_in_train.union(kinases_in_valid))
            kinases_in_train.sort()
            row_in_train['KINASE_ACC_IDS'] = ",".join(kinases_in_train)
        else:
            train_rows[site_identifier] = row

    train_rows = list(train_rows.values())
    return train_rows

def merge_train_and_validation_kinases(train_kinases, validation_kinases):
    all_kinases = train_kinases.union(validation_kinases)
    all_kinases = list(all_kinases)
    all_kinases.sort()
    return all_kinases

def create_dataset_csv_files(sites_in_train, sites_in_valid, sites_in_test,
                            train_kinase_to_count_new, validation_kinase_to_count_new, test_kinase_to_count_new, random_seed):
    random_seed_folder = f'random_seed_{random_seed}'
    directory = f'datasets/{random_seed_folder}'
    zsl_directory = f'{directory}/ZSL'
    zsl_directory_train_test = f'{directory}/ZSL/train_test_split'
    gzsl_directory = f'{directory}/GZSL'
    gzsl_directory_train_test = f'{directory}/GZSL/train_test_split'

    create_folder_if_non_existent(zsl_directory_train_test)
    create_folder_if_non_existent(gzsl_directory_train_test)

    '''
        The structure of the folders will be like this:
        GZSL
        -----------
        train_kinases
        validation_kinases
        test_kinases
        train_data
        validation_data
        test_data

            train_test_split (subfolder)
            ------------------
            train_kinases (this will be the train and validation kinases merged)
            test_kinases 
            train_data (This will be the train and validation rows merged)
            test_data
        '''

    stable = True

    # First Save for the GZSL setup
    # Before saving this into a csv file, lets check whether it is the exact same equal of the previously created dataset
    stable = save_dataset_files(f"{gzsl_directory}/train_data_{random_seed_folder}.csv", "GZSL root folder train data", sites_in_train, stable)
    stable = save_dataset_files(f"{gzsl_directory}/GZSL_validation_data_{random_seed_folder}.csv", "GZSL root folder validation data", sites_in_valid, stable)
    stable = save_dataset_files(f"{gzsl_directory}/GZSL_test_data_{random_seed_folder}.csv", "GZSL root folder test data", sites_in_test, stable)

    stable = save_kinase_dataset_files(f"{gzsl_directory}/train_kinases_{random_seed_folder}.csv", "GZSL root folder train kinases dataframe", train_kinase_to_count_new.keys(), stable)
    stable = save_kinase_dataset_files(f"{gzsl_directory}/GZSL_validation_kinases_{random_seed_folder}.csv", "GZSL root folder validation kinases dataframe", validation_kinase_to_count_new.keys(), stable)
    stable = save_kinase_dataset_files(f"{gzsl_directory}/GZSL_test_kinases_{random_seed_folder}.csv", "GZSL root folder test kinases dataframe", test_kinase_to_count_new.keys(), stable)

    # Now save for the train test split
    merged_train_rows = merge_sites_in_train_and_validation(sites_in_train, sites_in_valid)
    merged_train_and_validation_kinases = merge_train_and_validation_kinases(set(train_kinase_to_count_new.keys()),
                                                                             set(validation_kinase_to_count_new.keys()))
    stable = save_dataset_files(f"{gzsl_directory_train_test}/train_data_{random_seed_folder}.csv", "GZSL train_test split folder train dataframe", merged_train_rows, stable)
    stable = save_dataset_files(f"{gzsl_directory_train_test}/GZSL_test_data_{random_seed_folder}.csv", "GZSL train_test split folder test dataframe", sites_in_test, stable)

    stable = save_kinase_dataset_files(f"{gzsl_directory_train_test}/train_kinases_{random_seed_folder}.csv", "GZSL train_test split folder train kinases dataframe", merged_train_and_validation_kinases, stable)
    stable = save_kinase_dataset_files(f"{gzsl_directory_train_test}/GZSL_test_kinases_{random_seed_folder}.csv", "GZSL train_test split folder test kinases dataframe", test_kinase_to_count_new.keys(), stable)

    # GZSL datasets finished:
    if stable == None:
        print(f'CANNOT RUN TEST : FIRST TIME GENERATING THIS DATASET, PREVIOUS VERSION DOESN\'T EXIST SO CANNOT CHECK THE STABILITY WITH PREVIOUS VERSION')
    elif stable:
        print(f'PASSED TEST : ALL ROWS ARE EQUAL WITH THE PREVIOUS VERSION WHEN CREATING NEW DATAFRAME FOR GZSL')
    else:
        print(f'FAILED TEST : NOT ALL ROWS ARE EQUAL WITH THE PREVIOUS VERSION WHEN CREATING NEW DATAFRAME FOR GZSL (problematic datasets are listed above)')

    # Now save for the ZSL setup
    # We have to delete all of the seen kinases from the test rows.
    all_train = set(train_kinase_to_count_new.keys())
    all_valid = set(validation_kinase_to_count_new.keys())
    all_test = set(test_kinase_to_count_new.keys())
    unseen_test_kinases = all_test - (all_train.union(all_valid))  # Same as : all_test - set(merged_train_and_validation_kinases)
    unseen_valid_kinases = all_valid - all_train

    ZSL_sites_in_test = create_ZSL_sites(sites_in_test, unseen_test_kinases)
    ZSL_sites_in_valid = create_ZSL_sites(sites_in_valid, unseen_valid_kinases)

    stable = True
    stable = save_dataset_files(f"{zsl_directory}/train_data_{random_seed_folder}.csv", "ZSL root folder train dataframe", sites_in_train, stable)
    stable = save_dataset_files(f"{zsl_directory}/ZSL_validation_data_{random_seed_folder}.csv", "ZSL root folder validation dataframe", ZSL_sites_in_valid, stable)
    stable = save_dataset_files(f"{zsl_directory}/ZSL_test_data_{random_seed_folder}.csv", "ZSL root folder test dataframe", ZSL_sites_in_test, stable)

    stable = save_kinase_dataset_files(f"{zsl_directory}/train_kinases_{random_seed_folder}.csv", "ZSL root folder train kinases dataframe", train_kinase_to_count_new.keys(), stable)
    stable = save_kinase_dataset_files(f"{zsl_directory}/ZSL_validation_kinases_{random_seed_folder}.csv", "ZSL root folder validation kinases dataframe", unseen_valid_kinases, stable)
    stable = save_kinase_dataset_files(f"{zsl_directory}/ZSL_test_kinases_{random_seed_folder}.csv", "ZSL root folder test kinases dataframe", unseen_test_kinases, stable)

    # Now save for the ZSL train test split
    merged_train_rows = merge_sites_in_train_and_validation(sites_in_train, sites_in_valid)
    merged_train_and_validation_kinases = merge_train_and_validation_kinases(set(train_kinase_to_count_new.keys()),
                                                                             set(unseen_valid_kinases))

    stable = save_dataset_files(f"{zsl_directory_train_test}/train_data_{random_seed_folder}.csv", "ZSL train_test split folder train dataframe", merged_train_rows, stable)
    stable = save_dataset_files(f"{zsl_directory_train_test}/ZSL_test_data_{random_seed_folder}.csv", "ZSL train_test split folder test dataframe", ZSL_sites_in_test, stable)

    stable = save_kinase_dataset_files(f"{zsl_directory_train_test}/train_kinases_{random_seed_folder}.csv", "ZSL train_test split folder train kinases dataframe", merged_train_and_validation_kinases, stable)
    stable = save_kinase_dataset_files(f"{zsl_directory_train_test}/ZSL_test_kinases_{random_seed_folder}.csv", "ZSL train_test split folder test kinases dataframe", unseen_test_kinases, stable)

    # ZSL datasets finished:
    if stable == None:
        print(f'CANNOT RUN TEST : FIRST TIME GENERATING THIS DATASET, PREVIOUS VERSION DOESN\'T EXIST SO CANNOT CHECK THE STABILITY WITH PREVIOUS VERSION')
    elif stable:
        print(f'PASSED TEST : ALL ROWS ARE EQUAL WITH THE PREVIOUS VERSION WHEN CREATING NEW DATAFRAME FOR ZSL')
    else:
        print(f'FAILED TEST : NOT ALL ROWS ARE EQUAL WITH THE PREVIOUS VERSION WHEN CREATING NEW DATAFRAME FOR ZSL (problematic datasets are listed above)')


def create_ZSL_sites(sites_in_test, unseen_test_kinases):
    ZSL_sites_in_test = list()
    for row in sites_in_test:
        kinases = row["KINASE_ACC_IDS"]
        kinases = set(kinases.split(','))
        unseen_kinases = unseen_test_kinases.intersection(kinases)
        if len(unseen_kinases) > 0:
            new_row = row.copy()
            only_unseen_kinases = list(unseen_kinases)
            only_unseen_kinases.sort()
            new_row['KINASE_ACC_IDS'] = ",".join(only_unseen_kinases)
            ZSL_sites_in_test.append(new_row)
    return ZSL_sites_in_test


def save_kinase_dataset_files(filename, dataset_information, kinases, stable):
    sorted_kinases = list(kinases)
    sorted_kinases.sort()
    df = pd.DataFrame(sorted_kinases, columns=["Kinase"])
    stable = stable and check_dataframe_stability(dataset_information, df, filename)
    df.to_csv(filename, index=False)
    return stable


def save_dataset_files(filename, dataset_information, site_rows, stable):
    df = pd.DataFrame(site_rows)
    stable = stable and check_dataframe_stability(dataset_information, df, filename)
    df.to_csv(filename, index=False)
    return stable

def fill_subplots_bar_graph(axes, column, set_name, dataset_dict, group_distribution, group_count_distribution,
                            test_seen_kinase, test_unseen_kinases, valid_seen_kinase, valid_unseen_kinases):
    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1

    x_labels_kinase_group = []
    x_values_kinase_group = []
    for k, v in group_distribution.items():
        x_labels_kinase_group.append(k)
        x_values_kinase_group.append(v)

    x_labels_kinase_count_group = []
    x_values_kinase_count_group = []
    for k, v in group_count_distribution.items():
        x_labels_kinase_count_group.append(k)
        x_values_kinase_count_group.append(v)

    axes[0, column].set_axis_off()
    text = f'Number of unique kinases {len(dataset_dict.keys())}\n' \
           f'Number of total data {sum(dataset_dict.values())}'
    if set_name == "Test":
        text += f'\nSeen classes {len(test_seen_kinase)} - Unseen classes {len(test_unseen_kinases)}'
    elif set_name == "Valid":
        text += f'\nSeen classes {len(valid_seen_kinase)} - Unseen classes {len(valid_unseen_kinases)}'
    axes[0, column].text(0.5, 0.5, text, ha='center', va='top')

    # Plot for train set distribution
    axes[1, column].bar(x_labels_kinase_group, x_values_kinase_group, color=grayish_color)
    axes[1, column].set_xlabel('Kinase Groups', fontsize=8, fontweight='bold')
    axes[1, column].set_ylabel('Unique Kinase Count', fontsize=8, fontweight='bold')
    axes[1, column].tick_params(axis='x', rotation=45, labelsize=8)
    for i, count in enumerate(x_values_kinase_group):
        axes[1, column].text(i, count + 0.5, str(count), ha='center')
    axes[1, column].set_axisbelow(True)
    axes[1, column].grid(axis='y', zorder=0)
    # Remove the line boundaries
    axes[1, column].spines['top'].set_visible(False)
    axes[1, column].spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    axes[1, column].spines['left'].set_visible(False)
    axes[1, column].tick_params(axis='both', which='both', length=0)

    # Plot for train set distribution
    axes[2, column].bar(x_labels_kinase_count_group, x_values_kinase_count_group, color=grayish_color)
    axes[2, column].set_xlabel('Kinase Groups', fontsize=8, fontweight='bold')
    axes[2, column].set_ylabel('Total Phosphosite-Kinase pairs', fontsize=8, fontweight='bold')
    axes[2, column].tick_params(axis='x', rotation=45, labelsize=8)
    for i, count in enumerate(x_values_kinase_count_group):
        axes[2, column].text(i, count + 0.5, str(count), ha='center')
    axes[2, column].set_axisbelow(True)
    axes[2, column].grid(axis='y', zorder=0)
    # Remove the line boundaries
    axes[2, column].spines['top'].set_visible(False)
    axes[2, column].spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    axes[2, column].spines['left'].set_visible(False)
    axes[2, column].tick_params(axis='both', which='both', length=0)

def visualize_group_distributions(train_kinase_to_count, validation_kinase_to_count, test_kinase_to_count,
                                  seen_test_kinases, unseen_test_kinases,
                                  valid_seen_test_kinases, valid_unseen_test_kinases,
                                  include_validation):
    _, kinase_to_groups = map_kinase_family_group_info()
    kinase_groups = list(set(kinase_to_groups.values()) - {"missing"})
    kinase_groups.sort()

    train_group_distribution = {group: 0 for group in kinase_groups}
    train_group_count_distribution = {group: 0 for group in kinase_groups}
    valid_group_distribution = {group: 0 for group in kinase_groups}
    valid_group_count_distribution = {group: 0 for group in kinase_groups}
    test_group_distribution = {group: 0 for group in kinase_groups}
    test_group_count_distribution = {group: 0 for group in kinase_groups}

    for k, v in train_kinase_to_count.items():
        group = kinase_to_groups[k]
        train_group_distribution[group] += 1
        train_group_count_distribution[group] += v
        if group == "missing":
            print(f'{k} has missing group info')

    for k, v in test_kinase_to_count.items():
        group = kinase_to_groups[k]
        test_group_distribution[group] += 1
        test_group_count_distribution[group] += v
        if group == "missing":
            print(f'{k} has missing group info')

    if include_validation:
        for k, v in validation_kinase_to_count.items():
            group = kinase_to_groups[k]
            valid_group_distribution[group] += 1
            valid_group_count_distribution[group] += v
            if group == "missing":
                print(f'{k} has missing group info')

    columns = 2
    width_ratios = [1, 1]
    if include_validation:
        columns = 3
        width_ratios = [1, 1, 1]
    fig, axs = plt.subplots(nrows=3, ncols=columns, figsize=(10, 10),
                            gridspec_kw={
                                'width_ratios': width_ratios,
                                'height_ratios': [0.1, 3, 3],
                                'wspace': 0.4,
                                'hspace': 0.6})

    size = 8

    plt.rcParams['font.size'] = size
    pad = 5  # in points
    cols = ["Train set", "Test set"]
    if include_validation:
        cols = ["Train set", "Validation set", "Test set"]
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    fill_subplots_bar_graph(axs, 0, "Train", train_kinase_to_count, train_group_distribution, train_group_count_distribution,
                            seen_test_kinases, unseen_test_kinases, valid_seen_test_kinases, valid_unseen_test_kinases)
    if include_validation:
        fill_subplots_bar_graph(axs, 1, "Valid", validation_kinase_to_count, valid_group_distribution,
                                valid_group_count_distribution, seen_test_kinases, unseen_test_kinases,
                                valid_seen_test_kinases, valid_unseen_test_kinases)
        fill_subplots_bar_graph(axs, 2, "Test", test_kinase_to_count, test_group_distribution, test_group_count_distribution,
                                seen_test_kinases, unseen_test_kinases, valid_seen_test_kinases,
                                valid_unseen_test_kinases)
    else:
        fill_subplots_bar_graph(axs, 1, "Test", test_kinase_to_count, test_group_distribution, test_group_count_distribution,
                                seen_test_kinases, unseen_test_kinases, valid_seen_test_kinases,
                                valid_unseen_test_kinases)
    # Adjust the spacing between subplots
    fig.tight_layout()
    plt.show()

def fill_in_seen_unseen_subplots(axes, col_no, cols, grayish_color, seen_test_kinases, test_dict_new,
                                 unseen_test_kinases):
    seen_test_kinase_values = [test_dict_new[seen_kinase] for seen_kinase in seen_test_kinases]
    unseen_test_kinase_values = [test_dict_new[unseen_kinase] for unseen_kinase in unseen_test_kinases]
    x_labels_test = ["Seen test", "Unseen test"]
    x_values_test_kinase_count = [len(seen_test_kinases), len(unseen_test_kinases)]
    x_values_test_kinase_data_count = [sum(seen_test_kinase_values), sum(unseen_test_kinase_values)]
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    axes[0, col_no].set_axis_off()
    text = f'Number of unique seen kinases {len(seen_test_kinases)}\n' \
           f'Number of total data {sum([test_dict_new[kinase] for kinase in seen_test_kinases])} \n' \
           f'Number of unique unseen kinases {len(unseen_test_kinases)}\n' \
           f'Number of total data {sum([test_dict_new[kinase] for kinase in unseen_test_kinases])} \n'
    axes[0, col_no].text(0.5, 0.5, text, ha='center', va='top')

    # Plot for train set distribution
    axes[1, col_no].bar(x_labels_test, x_values_test_kinase_count, color=grayish_color)
    axes[1, col_no].set_xlabel('Test Kinases', fontsize=8, fontweight='bold')
    axes[1, col_no].set_ylabel('Test Kinase Count', fontsize=8, fontweight='bold')
    axes[1, col_no].tick_params(axis='x', rotation=45, labelsize=8)
    for i, count in enumerate(x_values_test_kinase_count):
        axes[1, col_no].text(i, count + 0.5, str(count), ha='center')
    axes[1, col_no].set_axisbelow(True)
    axes[1, col_no].grid(axis='y', zorder=0)
    # Remove the line boundaries
    axes[1, col_no].spines['top'].set_visible(False)
    axes[1, col_no].spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    axes[1, col_no].spines['left'].set_visible(False)
    axes[1, col_no].tick_params(axis='both', which='both', length=0)
    # Plot for train set distribution
    axes[2, col_no].bar(x_labels_test, x_values_test_kinase_data_count, color=grayish_color)
    axes[2, col_no].set_xlabel('Test Kinases', fontsize=8, fontweight='bold')
    axes[2, col_no].set_ylabel('Test Kinase Data Count', fontsize=8, fontweight='bold')
    axes[2, col_no].tick_params(axis='x', rotation=45, labelsize=8)
    for i, count in enumerate(x_values_test_kinase_data_count):
        axes[2, col_no].text(i, count + 0.5, str(count), ha='center')
    axes[2, col_no].set_axisbelow(True)
    axes[2, col_no].grid(axis='y', zorder=0)
    # Remove the line boundaries
    axes[2, col_no].spines['top'].set_visible(False)
    axes[2, col_no].spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    axes[2, col_no].spines['left'].set_visible(False)
    axes[2, col_no].tick_params(axis='both', which='both', length=0)

def plot_seen_unseen_distribution(validation_kinase_to_count, test_kinase_to_count,
                                  seen_test_kinases, unseen_test_kinases,
                                  valid_seen_test_kinases, valid_unseen_test_kinases):
    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1
    width_ratios = [1, 1]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10),
                             gridspec_kw={
                                 'width_ratios': width_ratios,
                                 'height_ratios': [0.1, 3, 3],
                                 'wspace': 0.4,
                                 'hspace': 0.6})
    cols = ["Test set", "Validation set"]

    fill_in_seen_unseen_subplots(axes, 0, cols, grayish_color, seen_test_kinases, test_kinase_to_count, unseen_test_kinases)
    fill_in_seen_unseen_subplots(axes, 1, cols, grayish_color, valid_seen_test_kinases, validation_kinase_to_count,
                                 valid_unseen_test_kinases)
    plt.show()

def seen_unseen_group_distibutions_fill_subplots_bar_graph(axes, column, group_distribution, group_count_distribution):
    grayish_color = (0.35, 0.35, 0.35)  # RGB values range from 0 to 1

    x_labels_kinase_group = []
    x_values_kinase_group = []
    for k, v in group_distribution.items():
        x_labels_kinase_group.append(k)
        x_values_kinase_group.append(v)

    x_labels_kinase_count_group = []
    x_values_kinase_count_group = []
    for k, v in group_count_distribution.items():
        x_labels_kinase_count_group.append(k)
        x_values_kinase_count_group.append(v)

    # Plot for train set distribution
    axes[0, column].bar(x_labels_kinase_group, x_values_kinase_group, color=grayish_color)
    axes[0, column].set_xlabel('Kinase Groups', fontsize=8, fontweight='bold')
    axes[0, column].set_ylabel('Unique Kinase Count', fontsize=8, fontweight='bold')
    axes[0, column].tick_params(axis='x', rotation=45, labelsize=8)
    for i, count in enumerate(x_values_kinase_group):
        axes[0, column].text(i, count + 0.5, str(count), ha='center')
    axes[0, column].set_axisbelow(True)
    axes[0, column].grid(axis='y', zorder=0)
    # Remove the line boundaries
    axes[0, column].spines['top'].set_visible(False)
    axes[0, column].spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    axes[0, column].spines['left'].set_visible(False)
    axes[0, column].tick_params(axis='both', which='both', length=0)

    # Plot for train set distribution
    axes[1, column].bar(x_labels_kinase_count_group, x_values_kinase_count_group, color=grayish_color)
    axes[1, column].set_xlabel('Kinase Groups', fontsize=8, fontweight='bold')
    axes[1, column].set_ylabel('Total Phosphosite-Kinase pairs', fontsize=8, fontweight='bold')
    axes[1, column].tick_params(axis='x', rotation=45, labelsize=8)
    for i, count in enumerate(x_values_kinase_count_group):
        axes[1, column].text(i, count + 0.5, str(count), ha='center')
    axes[1, column].set_axisbelow(True)
    axes[1, column].grid(axis='y', zorder=0)
    # Remove the line boundaries
    axes[1, column].spines['top'].set_visible(False)
    axes[1, column].spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    axes[1, column].spines['left'].set_visible(False)
    axes[1, column].tick_params(axis='both', which='both', length=0)

    axes[0, column].set_ylim(0, max(x_values_kinase_group) + 2)  # Adjust the "+ 2" as needed
    axes[1, column].set_ylim(0, max(x_values_kinase_count_group) + 2)  # Adjust the "+ 2" as needed

def plot_seen_unseen_group_distributions(validation_kinase_to_count, test_kinase_to_count,
                                         seen_test_kinases, unseen_test_kinases,
                                         valid_seen_test_kinases, valid_unseen_test_kinases,
                                         include_validation):
    _, kinase_to_groups = map_kinase_family_group_info()
    kinase_groups = list(set(kinase_to_groups.values())-{"missing"})
    kinase_groups.sort()
    seen_test_group_distribution = {group: 0 for group in kinase_groups}
    seen_test_group_count_distribution = {group: 0 for group in kinase_groups}
    unseen_test_group_distribution = {group: 0 for group in kinase_groups}
    unseen_test_group_count_distribution = {group: 0 for group in kinase_groups}
    seen_valid_group_distribution = {group: 0 for group in kinase_groups}
    seen_valid_group_count_distribution = {group: 0 for group in kinase_groups}
    unseen_valid_group_distribution = {group: 0 for group in kinase_groups}
    unseen_valid_group_count_distribution = {group: 0 for group in kinase_groups}
    for kinase in seen_test_kinases:
        group = kinase_to_groups[kinase]
        seen_test_group_distribution[group] += 1
        seen_test_group_count_distribution[group] += test_kinase_to_count[kinase]
    for kinase in unseen_test_kinases:
        group = kinase_to_groups[kinase]
        unseen_test_group_distribution[group] += 1
        unseen_test_group_count_distribution[group] += test_kinase_to_count[kinase]
    if include_validation:
        for kinase in valid_seen_test_kinases:
            group = kinase_to_groups[kinase]
            seen_valid_group_distribution[group] += 1
            seen_valid_group_count_distribution[group] += validation_kinase_to_count[kinase]

        for kinase in valid_unseen_test_kinases:
            group = kinase_to_groups[kinase]
            unseen_valid_group_distribution[group] += 1
            unseen_valid_group_count_distribution[group] += validation_kinase_to_count[kinase]
    columns = 2
    width_ratios = [1, 1]
    if include_validation:
        columns = 4
        width_ratios = [1, 1, 1, 1]
    fig, axs = plt.subplots(nrows=2, ncols=columns, figsize=(10, 10),
                            gridspec_kw={
                                'width_ratios': width_ratios,
                                'height_ratios': [3, 3],
                                'wspace': 0.4,
                                'hspace': 0.6})
    size = 8

    plt.rcParams['font.size'] = size
    pad = 5  # in points
    cols = ["Seen Test", "Unseen Test"]
    if include_validation:
        cols = ["Seen Test", "Unseen Test", "Seen valid", "Unseen valid"]
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    seen_unseen_group_distibutions_fill_subplots_bar_graph(axs, 0, seen_test_group_distribution,
                                                           seen_test_group_count_distribution)
    seen_unseen_group_distibutions_fill_subplots_bar_graph(axs, 1, unseen_test_group_distribution,
                                                           unseen_test_group_count_distribution)
    if include_validation:
        seen_unseen_group_distibutions_fill_subplots_bar_graph(axs, 2, seen_valid_group_distribution,
                                                               seen_valid_group_count_distribution)
        seen_unseen_group_distibutions_fill_subplots_bar_graph(axs, 3, unseen_valid_group_distribution,
                                                               unseen_valid_group_count_distribution)
    # Adjust the spacing between subplots
    fig.tight_layout()
    plt.show()

def visualize_dataset_sizes_inner(train_kinase_to_count, validation_kinase_to_count, test_kinase_to_count, include_validation):
    sets = ["Train set", "Test set"]
    values = [sum(train_kinase_to_count.values()), sum(test_kinase_to_count.values())]
    if include_validation:
        values = [sum(train_kinase_to_count.values()), sum(validation_kinase_to_count.values()), sum(test_kinase_to_count.values())]
        sets = ["Train set", "Validation set", "Test set"]
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
    plt.show()

def create_GZSL_dataset(kinase_count_test_threshold, stratify_percentage_for_unseen_test_kinase, test_gzsl_percentage,
                        kinase_count_validation_threshold, stratify_percentage_for_unseen_validation_kinase, valid_gzsl_percentage,
                        kinase_similarity_percent, include_validation, random_seed,
                        take_seq_sim_into_consideration, divide_wrt_group):

    # 1. So first we have to place all kinases below the test threshold into the training set
    # 2. Then we have to receive a stratify percentage : Among all values above this test threshold,
    # this much percentage of it must directly go to test and since this is GZSL these will be unseen classes.
    # 3. For the remaining kinases, some percentage of it must go to train and some must stay in test,
    # since this is GZSL, we will define this as the GZSL threshold. These classes will be the seen classes.

    train_kinase_to_count = dict()
    validation_kinase_to_count = dict()
    test_kinase_to_count = dict()

    kinase_to_occurrence = get_kinase_occurrence_in_the_dataset()
    print("Starting Train-Test Split...")
    if divide_wrt_group:
        _, kinase_to_groups = map_kinase_family_group_info()
        # Add each kinase to the group it belongs to
        group_to_kinase_dict = dict()
        for k, v in kinase_to_occurrence.items():
            group = kinase_to_groups[k]
            if group not in group_to_kinase_dict:
                group_to_kinase_dict[group] = dict()
            group_to_kinase_dict[group][k] = v

        # decide the number of kinases to be added to the test set from each group
        group_to_dataset_count = dict()
        for k, v in group_to_kinase_dict.items():
            total_group_count = sum(group_to_kinase_dict[k].values())
            unseen_test_group_count = np.ceil(total_group_count * stratify_percentage_for_unseen_test_kinase)
            unseen_valid_group_count = np.ceil(
                total_group_count * stratify_percentage_for_unseen_validation_kinase) if include_validation else 0
            group_to_dataset_count[k] = {"total": total_group_count,
                                         "unseen_valid": unseen_valid_group_count,
                                         "unseen_test": unseen_test_group_count}

        kinase_groups = list(set(kinase_to_groups.values()))
        kinase_groups.sort()

        train_kinases, seen_test_kinases, unseen_test_kinases = distribute_kinases_wrt_group(train_kinase_to_count,
                                                                                             test_kinase_to_count,
                                                                                               "test",
                                                                                             kinase_count_test_threshold,
                                                                                             test_gzsl_percentage,
                                                                                             group_to_kinase_dict,
                                                                                             kinase_groups,
                                                                                             group_to_dataset_count,
                                                                                             kinase_similarity_percent,
                                                                                             take_seq_sim_into_consideration)
    else:
        total_kinase_data = sum(kinase_to_occurrence.values())
        total_unseen_test_data = np.floor(total_kinase_data * stratify_percentage_for_unseen_test_kinase)
        total_unseen_validation_data = np.floor(total_kinase_data * stratify_percentage_for_unseen_validation_kinase)

        train_kinases, seen_test_kinases, unseen_test_kinases = distribute_kinases(train_kinase_to_count,
                                                                                   test_kinase_to_count,
                                                                                    "test",
                                                                                   kinase_count_test_threshold,
                                                                                   test_gzsl_percentage,
                                                                                   kinase_to_occurrence,
                                                                                   total_unseen_test_data,
                                                                                   kinase_similarity_percent,
                                                                                   take_seq_sim_into_consideration)

    file = MULTI_CLASS_KINASE_SUBSTRATE_DATASET_FILE
    df_original = pd.read_csv(file)
    df_shuffled = df_original.sample(frac=1).reset_index(drop=True)
    all_data_rows = [row for index, row in df_shuffled.iterrows()]

    train_kinase_to_count_new, test_kinase_to_count_new, sites_in_train, sites_in_test, masked_rows_test = \
                                                        distribute_dataset_rows_into_datasets(train_kinase_to_count,
                                                                                              test_kinase_to_count,
                                                                                              train_kinases,
                                                                                              seen_test_kinases,
                                                                                              unseen_test_kinases,
                                                                                              all_data_rows)


    print("Train-Test Split Finished...\n")

    ################################################################################################
    ############################## TRAIN-TEST SPLIT TESTS ##########################################
    ################################################################################################

    print("Starting tests on train-test split...")
    site_count_check_after_train_test_split(all_data_rows, masked_rows_test, sites_in_test, sites_in_train, kinase_to_occurrence)
    data_count_check_after_train_test_split(test_kinase_to_count_new, train_kinase_to_count_new)
    print("Train-test split tests done...\n")

    #################################################################################################
    ##############################   VALIDATION SPLIT   #############################################
    #################################################################################################

    valid_train_kinases, valid_seen_test_kinases, valid_unseen_test_kinases = set(), set(), set()
    if include_validation:
        print("Starting Train-Validation Split...")
        if divide_wrt_group:
            group_to_kinase_dict = dict()
            for k, v in train_kinase_to_count_new.items():  # This is because we are now going to split train into train and validation
                group = kinase_to_groups[k]
                if group not in group_to_kinase_dict:
                    group_to_kinase_dict[group] = dict()
                group_to_kinase_dict[group][k] = v
            train_kinase_to_count = dict()
            validation_kinase_to_count = dict()


            valid_train_kinases, valid_seen_test_kinases, valid_unseen_test_kinases = distribute_kinases_wrt_group(train_kinase_to_count,
                                                                                                                   validation_kinase_to_count,
                                                                                                                     "valid",
                                                                                                                   kinase_count_validation_threshold,
                                                                                                                   valid_gzsl_percentage,
                                                                                                                   group_to_kinase_dict,
                                                                                                                   kinase_groups,
                                                                                                                   group_to_dataset_count,
                                                                                                                   kinase_similarity_percent,
                                                                                                                   take_seq_sim_into_consideration)
        else:
            train_kinase_to_count_to_be_plit = copy.deepcopy(train_kinase_to_count)
            train_kinase_to_count = dict()
            validation_kinase_to_count = dict()
            valid_train_kinases, valid_seen_test_kinases, valid_unseen_test_kinases = distribute_kinases(train_kinase_to_count,
                                                                                                         validation_kinase_to_count,
                                                                                                            "valid",
                                                                                                         kinase_count_validation_threshold,
                                                                                                         valid_gzsl_percentage,
                                                                                                         train_kinase_to_count_to_be_plit,
                                                                                                         total_unseen_validation_data,
                                                                                                         kinase_similarity_percent,
                                                                                                         take_seq_sim_into_consideration)

        train_kinase_to_count_new, validation_kinase_to_count_new, sites_in_train, sites_in_valid, masked_rows_valid = distribute_dataset_rows_into_datasets(
                                                                                                        train_kinase_to_count,
                                                                                                        validation_kinase_to_count,
                                                                                                        valid_train_kinases,
                                                                                                        valid_seen_test_kinases,
                                                                                                        valid_unseen_test_kinases,
                                                                                                        sites_in_train)
        print("Train-Validation Split Finished...\n")

    # Remove kinases with 0 data (They don't have to appear in the statistical analysis)
    train_kinase_to_count_new = {kinase: train_kinase_to_count_new[kinase] for kinase, count in train_kinase_to_count_new.items() if count != 0}
    train_kinases = set(train_kinase_to_count_new.keys())

    print(f'Starting tests after train-validation-test splits...')
    check_total_row_counts_from_dictionaries(train_kinase_to_count_new, validation_kinase_to_count_new, test_kinase_to_count_new, kinase_to_occurrence)
    all_seen_sites_should_have_unseen_test_kinase(sites_in_test, sites_in_train, unseen_test_kinases)
    check_duplciate_rows(sites_in_train, sites_in_test, train_test=True)
    check_duplciate_rows(sites_in_train, sites_in_valid, train_test=False)
    print(f'Train-validation-test splits are done...\n')

    print(f'Creating the datasets...')
    create_dataset_csv_files(sites_in_train, sites_in_valid, sites_in_test,
                             train_kinase_to_count_new, validation_kinase_to_count_new, test_kinase_to_count_new, random_seed)
    print(f'Creating the datasets finished...\n')

    # After creating the datasets, we will now do some checks on these folders
    # GZSL
    print(f'Starting final tests for GZSL...')
    check_whether_unseen_test_data_exists_in_train(train_test_split=True, setup="GZSL")
    check_whether_unseen_test_data_exists_in_train(train_test_split=False, setup="GZSL")
    check_for_duplicate_rows(train_test_split=True, setup="GZSL")
    check_for_duplicate_rows(train_test_split=False, setup="GZSL")
    check_site_to_label_correctness(train_test_split=True, setup="GZSL")
    check_site_to_label_correctness(train_test_split=False, setup="GZSL")
    check_whether_seen_kinases_have_data_in_train(train_test=True, setup="GZSL")
    check_whether_seen_kinases_have_data_in_train(train_test=False, setup="GZSL")
    check_kinase_with_less_data_in_test(train_test_split=True, setup="GZSL", threshold=kinase_count_test_threshold)
    check_kinase_with_less_data_in_test(train_test_split=False, setup="GZSL", threshold=kinase_count_validation_threshold)
    check_whether_similar_kinases_are_all_in_same_dataset(train_test_split=True, setup="GZSL", similarity_percentage=kinase_similarity_percent)
    check_whether_similar_kinases_are_all_in_same_dataset(train_test_split=False, setup="GZSL", similarity_percentage=kinase_similarity_percent)
    check_kinase_similarity_within_the_dataset(train_test_split=True, setup="GZSL", similarity_percentage=kinase_similarity_percent)
    check_kinase_similarity_within_the_dataset(train_test_split=False, setup="GZSL", similarity_percentage=kinase_similarity_percent)
    check_kinase_occurence_and_original_count_match(setup="GZSL")
    seen_kinases_should_have_sites_it_hasnt_been_trained_on_in_test(train_test=True)
    seen_kinases_should_have_sites_it_hasnt_been_trained_on_in_test(train_test=False)
    print(f'Final tests for GZSL finished...\n')

    # ZSL
    print(f'Starting final tests for ZSL...')
    check_whether_unseen_test_data_exists_in_train(train_test_split=True, setup="ZSL")
    check_whether_unseen_test_data_exists_in_train(train_test_split=False, setup="ZSL")
    check_for_duplicate_rows(train_test_split=True, setup="ZSL")
    check_for_duplicate_rows(train_test_split=False, setup="ZSL")
    check_site_to_label_correctness(train_test_split=True, setup="ZSL")
    check_site_to_label_correctness(train_test_split=False, setup="ZSL")
    check_kinase_with_less_data_in_test(train_test_split=True, setup="ZSL", threshold=kinase_count_test_threshold)
    check_kinase_with_less_data_in_test(train_test_split=False, setup="ZSL", threshold=kinase_count_validation_threshold)
    check_whether_similar_kinases_are_all_in_same_dataset(train_test_split=True, setup="ZSL", similarity_percentage=kinase_similarity_percent)
    check_whether_similar_kinases_are_all_in_same_dataset(train_test_split=False, setup="ZSL", similarity_percentage=kinase_similarity_percent)
    check_kinase_similarity_within_the_dataset(train_test_split=True, setup="ZSL", similarity_percentage=kinase_similarity_percent)
    check_kinase_similarity_within_the_dataset(train_test_split=False, setup="ZSL", similarity_percentage=kinase_similarity_percent)
    check_kinase_occurence_and_original_count_match(setup="ZSL")
    ZSL_test_rows_shouldnt_contain_any_train_kinases(train_test=True)
    ZSL_test_rows_shouldnt_contain_any_train_kinases(train_test=False)
    print(f'Final tests for ZSL finished...\n')

    return train_kinase_to_count_new, validation_kinase_to_count_new, test_kinase_to_count_new, train_kinases, seen_test_kinases, unseen_test_kinases, valid_train_kinases, valid_seen_test_kinases, valid_unseen_test_kinases

def create_and_visualize_dataset(kinase_count_test_threshold, stratify_percentage_for_unseen_test_kinase, test_gzsl_percentage,
                                 kinase_count_validation_threshold, stratify_percentage_for_unseen_validation_kinase, valid_gzsl_percentage,
                                 kinase_similarity_percent, include_validation, random_seed,
                                 take_seq_sim_into_consideration, divide_wrt_group):
    train_kinase_to_count, validation_kinase_to_count, test_kinase_to_count, \
        train_kinases, seen_test_kinases, unseen_test_kinases, \
        valid_train_kinases, valid_seen_test_kinases, valid_unseen_test_kinases = \
        create_GZSL_dataset(kinase_count_test_threshold, stratify_percentage_for_unseen_test_kinase, test_gzsl_percentage,
                            kinase_count_validation_threshold, stratify_percentage_for_unseen_validation_kinase, valid_gzsl_percentage,
                            kinase_similarity_percent, include_validation, random_seed,
                            take_seq_sim_into_consideration, divide_wrt_group)

    # I added this check because there was a bug in the code, and the datasets weren't
    # always created the same. The problem was at where I shuffled remaining_groups_to_kinase.
    # Since there I was first getting the keys, then converting that into a list, it was always
    # getting a different shuffle (because .keys() returns a set and random_seed doesn't have
    # control over how sets are being ordered within itself).
    check_random_seed_stability(train_kinase_to_count, validation_kinase_to_count, test_kinase_to_count)

    visualize_dataset_sizes_inner(train_kinase_to_count, validation_kinase_to_count, test_kinase_to_count,
                                  include_validation)

    visualize_group_distributions(train_kinase_to_count, validation_kinase_to_count, test_kinase_to_count,
                                  seen_test_kinases, unseen_test_kinases,
                                  valid_seen_test_kinases, valid_unseen_test_kinases,
                                  include_validation)

    plot_seen_unseen_distribution(validation_kinase_to_count, test_kinase_to_count,
                                  seen_test_kinases, unseen_test_kinases,
                                  valid_seen_test_kinases, valid_unseen_test_kinases)

    plot_seen_unseen_group_distributions(validation_kinase_to_count, test_kinase_to_count,
                                         seen_test_kinases, unseen_test_kinases,
                                         valid_seen_test_kinases, valid_unseen_test_kinases,
                                         include_validation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Split Parser')

    parser.add_argument('--RANDOM_SEED', type=int, default=12345)
    parser.add_argument('--KINASE_SIMILARITY_PERCENT', type=int, default=90)

    parser.add_argument('--KINASE_COUNT_TEST_THRESHOLD', type=int, default=15)
    parser.add_argument('--STRATIFY_PERCENTAGE_FOR_UNSEEN_TEST_KINASE', type=float, default=0.10)
    parser.add_argument('--TEST_GZSL_PERCENTAGE', type=float, default=0.15)

    parser.add_argument('--KINASE_COUNT_VALIDATION_THRESHOLD', type=int, default=10)
    parser.add_argument('--STRATIFY_PERCENTAGE_FOR_UNSEEN_VALIDATION_KINASE', type=float, default=0.10)
    parser.add_argument('--VALIDATION_GZSL_PERCENTAGE', type=float, default=0.15)

    # Flags
    parser.add_argument('--INCLUDE_VALIDATION', type=bool, default=True)
    parser.add_argument('--TAKE_SEQUENCE_SIMILARITY_INTO_CONSIDERATION', type=bool, default=True)
    parser.add_argument('--DIVIDE_WRT_GROUP', type=bool, default=True)

    args = parser.parse_args()

    set_random_seed(args.RANDOM_SEED)
    create_and_visualize_dataset(args.KINASE_COUNT_TEST_THRESHOLD, args.STRATIFY_PERCENTAGE_FOR_UNSEEN_TEST_KINASE, args.TEST_GZSL_PERCENTAGE,
                                 args.KINASE_COUNT_VALIDATION_THRESHOLD, args.STRATIFY_PERCENTAGE_FOR_UNSEEN_VALIDATION_KINASE, args.VALIDATION_GZSL_PERCENTAGE,
                                 args.KINASE_SIMILARITY_PERCENT, args.INCLUDE_VALIDATION, args.RANDOM_SEED,
                                 args.TAKE_SEQUENCE_SIMILARITY_INTO_CONSIDERATION, args.DIVIDE_WRT_GROUP)
