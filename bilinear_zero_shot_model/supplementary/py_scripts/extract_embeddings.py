from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer, AlbertTokenizer, pipeline
import pickle
import pandas as pd
from Bio.SubsMat import MatrixInfo
from tqdm import tqdm
import numpy as np
import re

'''
NOTE: The embedding extraction code is mostly taken from Zeynep's files, here I am just adapting them so that
I could get the embeddings of the active sites
'''

MAX_LEN_ESM = 1024
MAX_LEN_PROT_T5 = 1024
MAX_LEN_ACTIVE_SITE = 29
AA = ["A", "C", "D", "E", "F","G", "H", "I", "K" , "L", "M", "N", "P", "Q", "R" , "S", "T", "V", "W", "X","Y"]
blosum62 = MatrixInfo.blosum62

def find_max_len(sequences):
    max_len = 0
    for seq in sequences:
        if len(seq) >= max_len:
            max_len = len(seq)

    return max_len

def get_kinase_kinase_Domain_sequences(filename):
    # Kinase,Family,Group,Kinase_Domain,EC,EC_Level2,EC_Level3,EC_Level4,Kin2Vec,Kegg_Pathways,Kinase_Type,active_site,active_site_indices,Kinase_Active_Site_Kin2vec,Kinase_Active_Site_Kin2vec_with_context
    df = pd.read_csv(filename)
    kinase_to_kinase_domain = dict()
    for index, row in df.iterrows():
        kinase_UniprotID = row['Kinase']
        kinase_domain = row['Kinase_Domain']
        kinase_to_kinase_domain[kinase_UniprotID] = kinase_domain
    return kinase_to_kinase_domain

def get_kinase_active_site_sequecnes(filename):
    # Kinase,Family,Group,Kinase_Domain,EC,EC_Level2,EC_Level3,EC_Level4,Kin2Vec,Kegg_Pathways,Kinase_Type,active_site,active_site_indices,Kinase_Active_Site_Kin2vec,Kinase_Active_Site_Kin2vec_with_context
    df = pd.read_csv(filename)
    kinase_to_active_site = dict()
    for index, row in df.iterrows():
        kinase_UniprotID = row['Kinase']
        kinase_domain = row['Kinase_Domain']
        active_Site = row['active_site']
        if pd.isna(active_Site):
            kinase_to_active_site[kinase_UniprotID] = kinase_domain
        else:
            kinase_to_active_site[kinase_UniprotID] = active_Site
    return kinase_to_active_site

def get_phosphosite_sequences():
    df = pd.read_csv(f'/truba/scratch/esunar/DeepKinZero/DeepKinZero/Dataset_new/Curated_Dataset/Clean_Kinase_Substrate_Dataset.csv')
    all_sites = set(df['SITE_+/-7_AA'])

    rs = f'0'

    df_train = pd.read_csv(f'/truba/scratch/esunar/DeepKinZero/DeepKinZero/dataset/new_dataset/zsl/seed_{rs}/train_data.csv')
    df_validation = pd.read_csv(f'/truba/scratch/esunar/DeepKinZero/DeepKinZero/dataset/new_dataset/zsl/seed_{rs}/validation_data.csv')
    df_test = pd.read_csv(f'/truba/scratch/esunar/DeepKinZero/DeepKinZero/dataset/new_dataset/zsl/seed_{rs}/test_data.csv')

    train_sites = set(df_train['SITE_+/-7_AA'])
    validation_sites = set(df_validation['SITE_+/-7_AA'])
    test_sites = set(df_test['SITE_+/-7_AA'])

    print(f'in rs {rs} {train_sites-all_sites} dont exist in train')
    print(f'in rs {rs} {validation_sites-all_sites} dont exist in validation')
    print(f'in rs {rs} {test_sites-all_sites} dont exist in test')

    return all_sites

def prepare_ESM1B_2_1v(modelname, sequences, choice, savepath, max_len):
    embs_matrix = {}

    tokenizer = AutoTokenizer.from_pretrained(modelname, do_lower_case=False)
    model = AutoModel.from_pretrained(modelname)

    for seq in tqdm(sequences):
        print(seq)
        prep_seq = seq.replace("_", "-").upper()

        if len(prep_seq) == 29:
            encoded_input = tokenizer.encode_plus(prep_seq, return_tensors='pt', padding="max_length", max_length=MAX_LEN_ACTIVE_SITE+2, truncation=True)
        else:
            encoded_input = tokenizer.encode_plus(prep_seq, return_tensors='pt')

        if choice == "noattentionmask":
            outputs = model(encoded_input["input_ids"])

            if seq not in embs_matrix:
                embs_matrix[seq] = outputs.last_hidden_state[0].tolist()
                #embs_pooled[seq] = outputs.pooler_output[0].tolist()

        else:
            outputs = model(**encoded_input)

            if seq not in embs_matrix:
                embs_matrix[seq] = outputs.last_hidden_state[0].tolist()

            # Accessing input_ids
            input_ids = encoded_input["input_ids"].squeeze().tolist()
            decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)

            # Printing decoded tokens
            print("Decoded Tokens:", decoded_tokens)

    for seq, embed in embs_matrix.items():
        embed_array = np.array(embed)
        print(f'Shape is : {embed_array.shape}')

    with open(savepath + "/" + savepath.split("/")[-1] + "ActiveSiteEmb_NoTrim_" + choice + ".pickle", 'wb') as handle:
        pickle.dump(embs_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model: ", modelname, "matrix form" , " Choice: ", choice , "# of data Check (544): ", len(embs_matrix))

## No CLS token defined and trained so average of embedding is taken
def prepare_ProtT5XL_kinase_domain(modelname, sequences, savepath):
    embs_avg = {}

    tokenizer = T5Tokenizer.from_pretrained(modelname, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(modelname)  
    for seq in tqdm(sequences): 
        prep_seq = " ".join(list(seq.upper()))
        prep_seq = re.sub(r"[UZOB]", "X", prep_seq).replace("_","-")
        encoded_input = tokenizer.encode_plus(prep_seq, return_tensors='pt', padding="max_length", max_length=MAX_LEN_PROT_T5, truncation=True)
        outputs = model(encoded_input["input_ids"])
        if seq not in embs_avg:
            embs_avg[seq] = outputs.last_hidden_state[0].tolist()

    with open(savepath + "/" + savepath.split("/")[-1] + "KinaseDomainEmb_NoAvg.pickle", 'wb') as handle:
        pickle.dump(embs_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for seq, embed in embs_avg.items():
        embed_array = np.array(embed)
        print(f'Shape is : {embed_array.shape}')

    print("Model: ", modelname, "matrix form" , "# of data Check: ", len(embs_avg))

## I think this is the same a sthe kinase function but even so i just got it from Zeynep's
def prepare_ProtT5XL_site(modelname, sequences, savepath):
    embs = {}

    tokenizer = T5Tokenizer.from_pretrained(modelname, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(modelname)  
    for seq in tqdm(sequences): 
        prep_seq = " ".join(list(seq.upper()))
        prep_seq = re.sub(r"[UZOB]", "X", prep_seq).replace("_","-")
        encoded_input = tokenizer.encode_plus(prep_seq, return_tensors='pt', padding="max_length", max_length=16, truncation=True)
        outputs = model(encoded_input["input_ids"])
        if seq not in embs:
            embs[seq] = outputs.last_hidden_state[0].tolist()

    with open(savepath + "/" + savepath.split("/")[-1] + "_SeqEmb_NoAvg.pickle", 'wb') as handle:
        pickle.dump(embs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for seq, embed in embs.items():
        embed_array = np.array(embed)
        print(f'Shape is : {embed_array.shape}')

    print("Model: ", modelname, "matrix form" , "# of data Check: ", len(embs))

# Kinase,Family,Group,Kinase_Domain,EC,EC_Level2,EC_Level3,EC_Level4,Kin2Vec,Kegg_Pathways,Kinase_Type,active_site,active_site_indices,Kinase_Active_Site_Kin2vec,Kinase_Active_Site_Kin2vec_with_context
Kinase_feature_file = f'/truba/scratch/esunar/DeepKinZero/DeepKinZero/dataset/new_dataset/zsl/seed_12345/kinase_properties.csv'
kinase_to_kinase_domain = get_kinase_kinase_Domain_sequences(Kinase_feature_file)
kinase_to_active_site = get_kinase_active_site_sequecnes(Kinase_feature_file)
site_sequences = get_phosphosite_sequences()

kinase_domain_sequences = list(set(kinase_to_kinase_domain.values()))
kinase_domain_sequences.sort()
active_site_sequences = list(set(kinase_to_active_site.values()))
active_site_sequences.sort()
max_len = find_max_len(active_site_sequences)

# active_site_sequences = ['-----FGKVAKDQALMEAGEGGESNVVDL', 'CGGGSFGSVAKELIITEYASLGDRNVCDA']
# prepare_ESM1B_2_1v("facebook/esm1b_t33_650M_UR50S", active_site_sequences, "withattentionmask", savepath="/truba/scratch/esunar/DeepKinZero/DeepKinZero/embeddings/active_sites/Esm1B", max_len=max_len)
# prepare_ProtT5XL("Rostlab/prot_t5_xl_uniref50", active_site_sequences, "/truba/scratch/esunar/DeepKinZero/DeepKinZero/embeddings/active_sites/ProtT5XL")
# prepare_ProtT5XL_site("Rostlab/prot_t5_xl_uniref50", active_site_sequences, "/truba/scratch/esunar/DeepKinZero/DeepKinZero/embeddings/kinase_domains/ProtT5XL")
# kinase_domain_sequences = ['FGKVAKDQALMEAGEG', 'CGGGSFGSVAKELI']
# prepare_ProtT5XL_kinase_domain("Rostlab/prot_t5_xl_uniref50",kinase_domain_sequences, "/truba/scratch/esunar/DeepKinZero/DeepKinZero/embeddings/kinase_domains")

# print(site_sequences)
# site_sequences = ['FGKVAKDQALMEAGEG', 'CGGGSFGSVAKELI']
# prepare_ProtT5XL_site("Rostlab/prot_t5_xl_uniref50", site_sequences, "/truba/scratch/esunar/DeepKinZero/DeepKinZero/embeddings/ProtT5XL")