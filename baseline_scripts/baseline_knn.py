import pickle
import numpy as np  
import csv
import pandas as pd  
from numpy import dot
from collections import Counter
import time
import json
from numpy.linalg import norm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--traindata', help='train data path', type=str)
parser.add_argument('--testdata', help='test data path', type=str)
parser.add_argument('--kinase_properties_file', type=str)
parser.add_argument('--k', help='K- nearest neihgbor', type=int)
parser.add_argument('--phosphosite_similarity_file', type=str)
parser.add_argument('--kinase_similarity_file', type=str)
parser.add_argument("--prediction_save_path", type=str)

args = parser.parse_args()


def family_group_dict(kinases):
    family = {}
    group = {}
    for i, row in kinases.iterrows():
        f = row["Family"]
        g = row["Group"]
        unip = row["Kinase"]
        if f not in family:
            family[f] = [unip]
        if g not in group:
            group[g] = [unip]
        if f in family and unip not in family[f] :
            family[f].append(unip)
        if g in group and unip not in group[g] :
            group[g].append(unip)

    return family, group

# read train, val, test data
def csv_reader(file):
    data_rows = []
    with open(file) as csvfile:
        Sub_DS = csv.reader(csvfile, delimiter=',')
        headers = next(Sub_DS) 
        for row in Sub_DS:
            row[2] = row[2].upper()
            #data_rows.append([col.upper().replace("_","-") for col in row])
            data_rows.append(row)
    return data_rows

# Create kinase_domain : embedding dictionary
# data = train or test
def create_kinase_domain_similarity_dict(data, kinase_similarity_m):
    kinase_domain_similarity = {}
    for line in data:
        kin_id = line[3]
        try:
            # single label
            if "," not in kin_id:
                domain = id_group_domain.loc[id_group_domain["Kinase"] == kin_id]["Kinase_Domain"].tolist()
                kinase_domain_similarity[domain[0]] = kinase_similarity_m[domain[0]]
            # multi label
            else: 
                kin_ids = kin_id.split(",")
                for ki in kin_ids:
                    domain = id_group_domain.loc[id_group_domain["Kinase"] == ki]["Kinase_Domain"].tolist()
                    kinase_domain_similarity[domain[0]] = kinase_similarity_m[domain[0]]

        except:
            print("Kinase not exist in embedding dictionary:", kin_id)


    indexes = []
    keys_kinases = list(kinase_similarity_m.keys())
    for k in kinase_domain_similarity.keys():
        indexes.append(keys_kinases.index(k))
            
    return indexes ,kinase_domain_similarity

# Create phosphosite_sequence : [similarity, class_id(kinase_id)]
def create_phosphosite_seq_similarity_dict(data, aminoasit_similarity_m):
    phosphosite_seq_similarity = {}
    idx=0
    for line in data:
        try:
            # single label 
            if "," not in line[3]:
                if line[2] in phosphosite_seq_similarity:
                    phosphosite_seq_similarity[line[2]].append(line[3]) 
                else: 
                    phosphosite_seq_similarity[line[2]] = [aminoasit_similarity_m[line[2]], line[3]]

            # multi label
            else:
                idx +=1
                kin_ids = line[3].split(",")
                for ki in kin_ids:
                    if line[2] in phosphosite_seq_similarity:    
                        phosphosite_seq_similarity[line[2]].append(ki) # update list
                    else: 
                        phosphosite_seq_similarity[line[2]] = [aminoasit_similarity_m[line[2]], ki]
        except:
            print(line[2])
                
            
    print("# of multilabel aa:", idx)

    indexes = []
    keys_seqs = list(aminoasit_similarity_m.keys())
    for k in phosphosite_seq_similarity.keys():
        indexes.append(keys_seqs.index(k))
    return indexes, phosphosite_seq_similarity

def flatten(mainlist):
    flat_list = [item for sublist in mainlist for item in sublist]

    return flat_list

def ground_truth(testdata, kinase_info):
    y_true = []
    y_true_group = []
    for line in testdata:
        label = line[3] 
        y_true.append(label.split(","))
        sub = []
        for l in label.split(","):
            kin = kinase_info.loc[kinase_info["Kinase"] == l]["Group"].tolist()[0]
            sub.append(kin)
        y_true_group.append(sub)


    return y_true, y_true_group


def return_kinase_ids(list_of_aa, aminoasit):
    ids = []
    for aa in list_of_aa:
        if len(aminoasit[aa]) > 2:
            for cls in aminoasit[aa][1:]:
                ids.append(cls)
        else:
            ids.append(aminoasit[aa][1])
            
        
    return ids

def return_kinase_id_by_seq(seq, kinase_csv):
    kin_id = kinase_csv.loc[kinase_csv["Kinase_Domain"] == seq]["Kinase"].tolist()

    return kin_id[0]

# return top class, first look for majority vote, else take the most similar one
def majority_vote(list_of_candidates):
    counter = Counter(list_of_candidates)
    counter_val = list(counter.values())
    max_occurence = max(counter_val)
    if max_occurence >1 and counter_val.count(max_occurence) == 1:
        most_common_elements = [elem for elem, count in counter.items() if count == max_occurence]
        return [most_common_elements[-1]]
    else:
        if max_occurence >1 and counter_val.count(max_occurence) != 1:
            most_common_elements = [elem for elem, count in counter.items() if count == max_occurence]
            return most_common_elements
        else:
            return  [list_of_candidates[-1]]
    
def return_kinase_similarity_from_csv_by_id(kin_id, kinase_csv, kinases_dict):
    try:
        domain = kinase_csv.loc[kinase_csv["Kinase"] == kin_id]["Kinase_Domain"].tolist()
        similarity_matrix = kinases_dict[domain[0]]
        return domain, similarity_matrix
    except:
        print(kin_id)

def all_unseen_kinase(file):
    data_rows = []
    with open(file) as csvfile:
        Sub_DS = csv.reader(csvfile)
        headers = next(Sub_DS) 
        for row in Sub_DS:
            data_rows.append(row[0])

    return data_rows

def calc_accuracy(y_true, y_true_group, preds, group_preds):
    num_of_sample = len(preds)
    acc = 0
    accg = 0
    for gt, p, gtg, pg in zip(y_true, preds,y_true_group,group_preds):
        if p in gt:
            acc = acc + 1
        if pg in gtg:
            accg = accg + 1


    return acc/num_of_sample, accg/num_of_sample



def knn(k, kinase_csv, all_phosphosite, train, test, seen_kinase_domain_similarity,seen_seq_indexes, unseen_kinase_indexes, unseen_phosphosite_seq_similarity, family, group, outfile, y_true,y_true_group):#, seen_aminoasit, seen_kinases, unseeen_aminoasit, unseen_kinases):
    preds=[]
    sample_class_probs = {}
    group_preds=[]
    with open(outfile, "w+") as outf:
        for testsample in test:
            unseen_seq = testsample[2]
            ## mevcut unseen'in diğer hepsiyle olan benzreliği bu. Buna başka bişey daha düşün.
            unseen_similarity = unseen_phosphosite_seq_similarity[unseen_seq][0] # current sequence's similarity matrix
            similarity_max_indices = np.argsort(unseen_similarity) # artana doğru
            try:
                seen_similarity_max_indices = []
                for i in similarity_max_indices[::-1]: ## reverse
                    if i in seen_seq_indexes:
                        seen_similarity_max_indices.append(i)
                    if len(seen_similarity_max_indices) == k:
                        break
                
                seen_similarity_max_indices = seen_similarity_max_indices[::-1] ## [en azdan en çoka]
                #seen_similarity_max_indices = [item for item in similarity_max_indices if item in seen_seq_indexes][-k:]
                # top_k_kinase_prediction => bu da en azdan en çoka
                top_k_kinase_prediction = flatten([train[all_phosphosite[i]].split(",") for i in seen_similarity_max_indices]) # multilabelcase
                predicted_seen_class = majority_vote(top_k_kinase_prediction)
                # testte olan site, trainde de geçebiliyor, o zaman en yakın birden fazla tahmin çıkabiliyor
                kinase_similarity = []
                for pcs in predicted_seen_class:
                    _, seen_kin_similarity = return_kinase_similarity_from_csv_by_id(pcs, id_group_domain, seen_kinase_domain_similarity)
                    ## tahmin edilen seen kinase ile beraber un seen kinase'leri dolaş en benzer unseen'i seç
                    kinase_similarity = kinase_similarity + seen_kin_similarity
                
                
                similarity_kinase_max_indices = np.argsort(kinase_similarity)
                for kinase_index in similarity_kinase_max_indices[::-1]: ## seen'ler de bunun içinde o yüzden unseen olmasını check etmek gerekiyor
                    if  kinase_index % len(seen_kin_similarity) in unseen_kinase_indexes:
                        pred_kinase_index = kinase_index % len(seen_kin_similarity)
                        break

                which_portion = pred_kinase_index // len(seen_kin_similarity)
                class_probs = kinase_similarity[len(seen_kin_similarity)*which_portion:len(seen_kin_similarity)*which_portion + len(seen_kin_similarity)]
                class_probs = [class_probs[i] for i in unseen_kinase_indexes]
                sample_class_probs[testsample[0]+"_"+testsample[1]+ "_" + testsample[2]] = class_probs
                predicted_kinase_domain = list(all_kinase_similarity.keys())[pred_kinase_index]
                predicted_unseen_class = kinase_csv.loc[kinase_csv["Kinase_Domain"] == predicted_kinase_domain]["Kinase"].tolist()[0]
                predicted_group = kinase_csv.loc[kinase_csv["Kinase"] == predicted_unseen_class]["Group"].tolist()[0]
                outf.write(predicted_unseen_class)
                outf.write("\n")
                preds.append(predicted_unseen_class)
                group_preds.append(predicted_group)
            except:
                print("-, Problemmatic kinase, not exist in embedding dict: " + predicted_seen_class)

                preds.append("-")

        accuracy, accuracy_group = calc_accuracy(y_true, y_true_group, preds, group_preds)
        outf.write("Accuracy:" + str(accuracy) +  "\n")
        outf.write("Group Accuracy:" + str(accuracy_group))
        jsondump = json.dumps(sample_class_probs)
        jsonFile = open(outfile.split(".")[0] + ".json", "w")
        jsonFile.write(jsondump)
        jsonFile.close()
        #outf.write("\n")
        #outf.flush()
    outf.close()
    return preds
              
start_time = time.time()
# Kinases info (.csv)
kinases_csv = pd.read_csv(args.kinase_properties_file)
id_group_domain = kinases_csv[["Kinase", "Group", "Kinase_Domain"]]

# ZSL Setting test classes
#all_unseen = all_unseen_kinase("/cta/users/zisik/DeepKinZero/Dataset/train_val_test_splits/datasets/v1_random_seed_12345/ZSL/ZSL_all_test_kinases_v1_random_seed_12345.csv")

family, group = family_group_dict(kinases_csv)
# Train (.csv), list of parsed rows
train = csv_reader(args.traindata)
# (args.traindata)#
train_dict = {} ## {sequence: labels}
for i in train:
    train_dict[i[2]]=i[3]

# Test (.csv), list of parsed rows
test = csv_reader(args.testdata)
# (args.testdata)#

# Ground Truth, labels 
y_true, y_true_group = ground_truth(test, id_group_domain) 


with open(args.kinase_similarity_file) as f:
   all_kinase_similarity = json.load(f)

with open(args.phosphosite_similarity_file) as f:
   all_phosphosite_similarity= json.load(f)

all_phosphosite_similarity = {k.replace("-", "_"): v for k,v in all_phosphosite_similarity.items()}

# Seen class embeddings and AA embeddings
seen_kinase_indexes, seen_kinase_domain_similarity_dict = create_kinase_domain_similarity_dict(train, all_kinase_similarity)
seen_phosphosite_indexes, seen_phosphosite_seq_similarity_dict  = create_phosphosite_seq_similarity_dict(train, all_phosphosite_similarity)


# Unseen class embeddings and AA embeddings
unseen_kinase_indexes, unseen_kinase_domain_similarity_dict = create_kinase_domain_similarity_dict(test, all_kinase_similarity)
unseen_phosphosite_indexes, unseen_phosphosite_seq_similarity_dict  = create_phosphosite_seq_similarity_dict(test, all_phosphosite_similarity)

# Run knn

knn_k = args.k
outfile = args.prediction_save_path + "/" +str(knn_k) +".txt"
preds = knn(knn_k, id_group_domain, list(all_phosphosite_similarity.keys()), train_dict, test, seen_kinase_domain_similarity_dict, seen_phosphosite_indexes, unseen_kinase_indexes, unseen_phosphosite_seq_similarity_dict,family, group,outfile, y_true,y_true_group)

