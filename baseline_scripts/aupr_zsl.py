from sklearn.metrics import auc, precision_recall_curve, average_precision_score
import json
import csv
import pandas as pd
import numpy as np
import argparse 
#from matplotlib import pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--testdata', help='test data path', type=str)
parser.add_argument('--kinase_properties_file', type=str)
parser.add_argument('--kinase_similarity', type=str)
parser.add_argument('--prediction_scores', type=str)
parser.add_argument('--k', type=int)
parser.add_argument('--outputpath', type=str)

args = parser.parse_args()
np.seterr(invalid='ignore')

def csv_reader(file):
    data_rows = []
    with open(file) as csvfile:
        Sub_DS = csv.reader(csvfile, delimiter=',')
        headers = next(Sub_DS) 
        for row in Sub_DS:
            data_rows.append(row)

    return data_rows

def all_unseen_kinase(file):
    data_rows = []
    with open(file) as csvfile:
        Sub_DS = csv.reader(csvfile)
        headers = next(Sub_DS) 
        for row in Sub_DS:
            data_rows.append(row[0])

    return data_rows

def ground_truth(testdata, all_unseen):
    classname_dict = {}
    for unseen in all_unseen:
        classname_dict[unseen] = []
        for line in testdata:
            label = line[3].split(",")
            if unseen in label:
                classname_dict[unseen].append(1)
            else:
                classname_dict[unseen].append(0)#classname_dict[unseen][line[0] + "_" + line[1] + "_" + line[2]] = 0

    return classname_dict


def predicted_kinase_sampledict(predictions, all_unseen):
    kinase_sampledict = {}
    for i,unseen in enumerate(all_unseen):
        for sample, scores in predictions.items():
            if unseen not in kinase_sampledict:
                kinase_sampledict[unseen] = [scores[i]]  
            else:  
                kinase_sampledict[unseen].append(scores[i])

    return kinase_sampledict

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


    indexes = {}
    keys_kinases = list(kinase_similarity_m.keys())
    for k in kinase_domain_similarity.keys():
        unipid = id_group_domain.loc[id_group_domain["Kinase_Domain"] == k]["Kinase"].tolist()[0]
        indexes[unipid] = keys_kinases.index(k)
            
    return indexes ,kinase_domain_similarity

def generate_fake(testdata, all_unseen):
    listt = [0.97]*len(all_unseen)
    listt[4] = 0.98
    classname_dict = {}
    for line in testdata:
        label = line[3].split(",")
        classname_dict[line[0] + "_" + line[1] + "_" + line[2]] = listt

    return classname_dict

kinases_csv = pd.read_csv(args.kinase_properties_file)
id_group_domain = kinases_csv[["Kinase", "Group", "Kinase_Domain"]]
test = csv_reader(args.testdata)


with open(args.kinase_similarity) as f:
   all_kinase_similarity = json.load(f)

indexes, _ = create_kinase_domain_similarity_dict(test, all_kinase_similarity)

######################################## LOAD PREDICTION SCORES 
with open(args.prediction_scores) as f:
   predictions = json.load(f)

y_true = ground_truth(test, list(indexes.keys())) # test unseen 
#fake_preds = generate_fake(test, list(indexes.keys()))
kinase_sampledict = predicted_kinase_sampledict(predictions, list(indexes.keys()))


######################################## GET AP RESULTSSS
outfile = args.outputpath + "/LOG_" + str(args.k) + ".txt"

class_aupr_scores= {}

for kinase, pred_scores in kinase_sampledict.items():
    total_score = sum(pred_scores)
    normalized_pred_scores = [score / total_score for score in pred_scores]
    # aupr calculation with average_precision_score:
    #prec, recall, th = precision_recall_curve(y_true[kinase], pred_scores)
    class_aupr = average_precision_score(y_true[kinase],normalized_pred_scores)
    #class_aupr = auc(recall,prec)
    class_aupr_scores[kinase] = class_aupr
        
macro_aupr = np.mean(list(class_aupr_scores.values()))
sorted_dict = sorted(class_aupr_scores.items(), key=lambda item: item[1], reverse=True)
with open(outfile, "w") as file:
    # Iterate through the list of tuples
    for item in sorted_dict:
        # Extract the elements of the tuple
        kinase, score = item
        
        # Write the formatted line to the file
        file.write(f"{kinase} : {score}\n")

    file.write("Average : " + str(macro_aupr))



