from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import csv
import pickle
import json
from collections import namedtuple
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import argparse
from distutils.util import strtobool

def str2bool(v):
    try:
        return bool(strtobool(v))
    except ValueError:
        raise argparse.ArgumentTypeError()
    
parser = argparse.ArgumentParser()
parser.add_argument('--phosphosite_emb_file', type=str)
parser.add_argument('--kinase_emb_file', type=str)
parser.add_argument('--kinase_properties_file', type=str)
parser.add_argument('--taken_token', type=str2bool)
parser.add_argument('--group', type=str2bool)
parser.add_argument("--family",  type=str2bool)
parser.add_argument("--ec",  type=str2bool)
parser.add_argument("--cls_or_avg",  type=str)
parser.add_argument("--savepath",  type=str)

args = parser.parse_args()


def csv_reader(file):
    data_rows = []
    with open(file) as csvfile:
        Sub_DS = csv.reader(csvfile, delimiter=',')
        headers = next(Sub_DS) 
        for row in Sub_DS:
            data_rows.append(row)

    return data_rows

def read_emb(path):
    if path.endswith(".pt"):
        k_emb = torch.load(path, map_location=torch.device('cpu'))
    else:
        with open(path, "rb") as f:
            k_emb = pickle.load(f)
    return k_emb

def load_kinase_data_from_csv(filename):
    kinase_data = {}
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        uniprot_id = row['Kinase_Domain']
        data_dict = {
            "family": row['Family'],
            "group": row['Group'],
            "enzymes_vec": list(map(float, list(row['EC']))),
        }

        kinase_data[uniprot_id] = data_dict
    return kinase_data

def _onehot_encode(class_list, encoders=None):
        if not class_list:
            raise ValueError("Input 'classes' array is empty or None.")

        class_list = np.array(class_list)

        if encoders is None:
            # Convert the labels to integers
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(class_list)
            # Binary encode
            onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            encoders = {'onehot': onehot_encoder, 'label': label_encoder}
        else:
            integer_encoded = encoders['label'].transform(class_list)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = encoders['onehot'].transform(integer_encoded)
        return onehot_encoded.tolist(), encoders

def _get_onehot_encoding_dict(class_list, encoders=None):
    onehot_encoded, encoders = _onehot_encode(class_list, encoders)
    class_onehot_dict = {encoded_class: onehot_encoded[i] for i, encoded_class in enumerate(encoders['label'].classes_)}
    return class_onehot_dict, encoders

def similarity_matrix(vectors):
    vector_matrix = np.array(vectors)#.astype(np.float16)
    pairwise_similarities = cosine_similarity(vector_matrix)
    
    return pairwise_similarities

def similarity_avg(phosphosite_emb, kinase_emb, kinase_data, encoded_dictf, encoded_dictg, group,family,ec,savepath, taketoken):
    vectors_ph = []
    vectors_kin = []
    domain_kin = []
    
    ## Phosphosite
    for k, v in phosphosite_emb.items():
        if v.dtype == torch.float32:
            v = v.tolist()
        subv = []
        ## padleri discard et
        if "ProtVec" in savepath:
            for i in v:
                if not all(element == 0 for element in i):
                    subv.append(i)
            vectors_ph.append(np.mean(subv, axis=0))
            
        elif taketoken == False:
            vectors_ph.append(v)
        else:
            vectors_ph.append(np.mean(v, axis=0))
    sim_matrix_ph = similarity_matrix(vectors_ph)

    dict_sim_ph = {}
    for k,vec in zip(phosphosite_emb.keys(),sim_matrix_ph):
        dict_sim_ph[k]=vec.tolist()
    
    jsondump = json.dumps(dict_sim_ph)
    jsonFile = open(savepath + "/" + "PhosphositeAvg.json", "w")
    jsonFile.write(jsondump)
    jsonFile.close()
    
    """
    #nopadd_v = v[1:len(k)+1] 
    #vectors_ph_mean.append(np.mean(nopadd_v, axis=0)) ## avg
    """
    
    ### Kinase
    for k, v in kinase_emb.items():
        if v.dtype == torch.float32:
            v = v.tolist()
        if taketoken == False:
            if k not in kinase_data:
                pass
            else:
                if "ProtGPT" in savepath:
                    v=v.tolist()
                if group:
                    additionalinfog = kinase_data[k]["group"]
                    v = v + encoded_dictg[additionalinfog]
                if family:
                    additionalinfof = kinase_data[k]["family"] 
                    v = v + encoded_dictf[additionalinfof]
                if ec:
                    additionalinfoec = kinase_data[k]["enzymes_vec"]
                    v = v + additionalinfoec
                 
                vectors_kin.append(v) 
                domain_kin.append(k)
        else: 
            if k not in kinase_data:
                pass
            else:
                nopadd_v = v[:len(k)+2]
                v = np.mean(nopadd_v, axis=0).tolist()

                if group:
                    additionalinfog = kinase_data[k]["group"]
                    v = v + encoded_dictg[additionalinfog]
                if family:
                    additionalinfof = kinase_data[k]["family"] 
                    v = v + encoded_dictf[additionalinfof]
                if ec:
                    additionalinfoec = kinase_data[k]["enzymes_vec"]
                    v = v + additionalinfoec

                vectors_kin.append(v) 
                domain_kin.append(k) 

    sim_matrix_kin = similarity_matrix(vectors_kin)

    dict_sim_kin = {}
    for k,vec in zip(domain_kin,sim_matrix_kin):
        dict_sim_kin[k]=vec.tolist()

    jsondump = json.dumps(dict_sim_kin)
    jsonFile = open(savepath + "/" + "KinaseAvg.json", "w")
    jsonFile.write(jsondump)
    jsonFile.close()
    
def similarity_cls(phosphosite_emb, kinase_emb, kinase_data, encoded_dictf, encoded_dictg, group,family,ec, savepath, taketoken):
    vectors_ph = []
    vectors_kin = []
    domain_kin = []
    
    ## Phosphosite
    
    for k, v in phosphosite_emb.items():
        if v.dtype == torch.float32:
            v = v.tolist()
        if taketoken == False:
            vectors_ph.append(v)
        else:
            vectors_ph.append(v[0])
    sim_matrix_ph = similarity_matrix(vectors_ph)

    dict_sim_ph = {}
    for k,vec in zip(phosphosite_emb.keys(),sim_matrix_ph):
        dict_sim_ph[k]=vec.tolist()

    jsondump = json.dumps(dict_sim_ph)
    jsonFile = open(savepath + "/" + "PhosphositeCLS.json", "w")
    jsonFile.write(jsondump)
    jsonFile.close()
    
    
    ## Kinase
    for k, v in kinase_emb.items():
        if v.dtype == torch.float32:
            v = v.tolist()
        if taketoken == False:
            if k not in kinase_data:
                pass
            else:
                
                if group:
                    additionalinfog = kinase_data[k]["group"]
                    v = v + encoded_dictg[additionalinfog]
                if family:
                    additionalinfof = kinase_data[k]["family"] 
                    v = v + encoded_dictf[additionalinfof]
                if ec:
                    additionalinfoec = kinase_data[k]["enzymes_vec"]
                    v = v + additionalinfoec

                vectors_kin.append(v)
                domain_kin.append(k)
        else: 
            v = v[0]
            if k not in kinase_data:
                pass
            else:
                
                if group:
                    additionalinfog = kinase_data[k]["group"]
                    v = v + encoded_dictg[additionalinfog]
                if family:
                    additionalinfof = kinase_data[k]["family"] 
                    v = v + encoded_dictf[additionalinfof]
                if ec:
                    additionalinfoec = kinase_data[k]["enzymes_vec"]
                    v = v + additionalinfoec
                
                vectors_kin.append(v) 
                domain_kin.append(k)

    sim_matrix_kin = similarity_matrix(vectors_kin)

    dict_sim_kin = {}
    for k,vec in zip(domain_kin,sim_matrix_kin):
        dict_sim_kin[k]=vec.tolist()

    jsondump = json.dumps(dict_sim_kin)
    jsonFile = open(savepath + "/" + "KinaseCLS.json", "w")
    jsonFile.write(jsondump)
    jsonFile.close()
    
## BLOSUM, Onehot, NLF
def similarity_irregular(phosphosite_emb, kinase_emb, kinase_data, encoded_dictf, encoded_dictg, group,family,ec, savepath):
    vectors_ph = []
    vectors_kin = []
    domain_kin = []
    
    ## Phosphosite
    for k, v in phosphosite_emb.items():
        v = np.array(v).reshape(1, -1).tolist()[0]
        vectors_ph.append(v)

    sim_matrix_ph = similarity_matrix(vectors_ph)

    dict_sim_ph = {}
    for k,vec in zip(phosphosite_emb.keys(),sim_matrix_ph):
        dict_sim_ph[k]=vec.tolist()

    jsondump = json.dumps(dict_sim_ph)
    jsonFile = open(savepath + "/" + "PhosphositeConcat.json", "w")
    jsonFile.write(jsondump)
    jsonFile.close()
    
    ## Kinase
    for k, v in kinase_emb.items():
        v = np.array(v).reshape(1, -1).tolist()[0]
        if k not in kinase_data:
            pass
        else:
            if group:
                additionalinfog = kinase_data[k]["group"]
                v = v + encoded_dictg[additionalinfog]
            if family:
                additionalinfof = kinase_data[k]["family"] 
                v = v + encoded_dictf[additionalinfof]
            if ec:
                additionalinfoec = kinase_data[k]["enzymes_vec"]
                v = v + additionalinfoec
            vectors_kin.append(v) 
            domain_kin.append(k)

    sim_matrix_kin = similarity_matrix(vectors_kin)

    dict_sim_kin = {}
    for k,vec in zip(domain_kin,sim_matrix_kin):
        dict_sim_kin[k]=vec.tolist()

    jsondump = json.dumps(dict_sim_kin)
    jsonFile = open(savepath + "/" + "KinaseConcat.json", "w")
    jsonFile.write(jsondump)
    jsonFile.close()

# Kinase additional features
kinase_data = load_kinase_data_from_csv(args.kinase_properties_file)
encoders = None
unique_families = sorted(list(set(item['family'] for item in kinase_data.values())))
unique_groups = sorted(list(set(item['group'] for item in kinase_data.values())))
encoded_family_dict, family_encoder = _get_onehot_encoding_dict(unique_families, encoders['family'] if encoders is not None else None)
encoded_group_dict, group_encoder = _get_onehot_encoding_dict(unique_groups, encoders['group'] if encoders is not None else None)

phosphosite_emb = read_emb(args.phosphosite_emb_file)
kinase_emb = read_emb(args.kinase_emb_file)

if args.cls_or_avg == "cls":
    similarity_cls(phosphosite_emb, kinase_emb, kinase_data, encoded_family_dict, encoded_group_dict, args.group, args.family, args.ec, args.savepath, args.taken_token)
elif args.cls_or_avg == "avg":
    similarity_avg(phosphosite_emb, kinase_emb, kinase_data, encoded_family_dict, encoded_group_dict, args.group, args.family, args.ec, args.savepath, args.taken_token)
else:
    similarity_irregular(phosphosite_emb, kinase_emb, kinase_data, encoded_family_dict, encoded_group_dict, args.group, args.family, args.ec, args.savepath)


