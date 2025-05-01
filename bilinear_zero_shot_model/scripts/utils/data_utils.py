import torch
import pickle
import pandas as pd

from scripts.data.data_processors.protvec_processor import ProtVecProcessor
from scripts.data.data_processors.hf_processor import HFProcessor
from scripts.data.data_processors.msa_processor import MsaProcessor


def load_phosphosite_data(filename):
        if filename.endswith('.txt'):
            return load_phosphosite_data_from_txt(filename)
        elif filename.endswith('.csv'):
            return load_phosphosite_data_from_csv(filename)
        else:
            raise Exception(f"Only .txt and .csv files are accepted")

def load_phosphosite_data_in_separate_rows(filename, config=None):
    if filename.endswith('.txt'):
        return load_phosphosite_data_from_txt(filename)
    elif filename.endswith('.csv'):
        return load_phosphosite_data_according_to_config(filename, config)
    else:
        raise Exception(f"Only .txt and .csv files are accepted")

def load_phosphosite_data_from_txt(filename):
    data = {
        "phosphosite_ids" : [],
        "sub_mods" : [],
        "phosphosite_sequences" : [],
        "kinase_ids" : [],
        "unique_kinases" : []
    }
    unique_kinases = []
    with open(filename, 'r') as file:
        for line in file:
            stripped_line = line.strip().split('\t')
            data["phosphosite_ids"].append(data[0])
            data["sub_mods"].append(data[1])
            data["phosphosite_sequences"].append(data[2].upper())
            data["kinase_ids"].append(data[3])
            data['unique_kinases'].extend(data[3].split(','))
    data['unique_kinases'] = sorted(list(set(data['unique_kinases'])))
    return data

def load_phosphosite_data_from_csv(filename):
    df = pd.read_csv(filename)

    phosphosite_ids = df["SUB_ACC_ID"].tolist()
    sub_mods = df["SUB_MOD_RSD"].tolist()
    phosphosite_sequences = df["SITE_+/-7_AA"].tolist()
    kinase_ids = df["KINASE_ACC_IDS"].tolist()
    unique_kinases = sorted(list(set([item.strip() for kinase_id in kinase_ids for item in kinase_id.split(',')])))

    return {
        "phosphosite_ids" : phosphosite_ids,
        "sub_mods" : sub_mods,
        "phosphosite_sequences" : phosphosite_sequences,
        "kinase_ids" : kinase_ids,
        "unique_kinases" : unique_kinases
    }

def load_phosphosite_data_according_to_config(filename, config=None):
    if config.get("training", {}).get("augment_aligned_unlabaled_sites", False):
        return load_phosphosite_data_from_csv_in_separate_rows_with_unlabaled_sites(filename, config)
    elif config.get("training", {}).get("add_cluster_data_to_eachother", False):
        return load_phosphosite_data_from_csv_in_separate_rows_by_adding_cluster_data_to_eachother(filename, config)
    else:
        return load_phosphosite_data_from_csv_in_separate_rows(filename)

def convert_site_Xs_to_underscores(site):
    return site.replace("X","_")

def get_aligned_unlabeled_data(config):
    df = pd.read_csv(config["phosphosite"]["dataset"]["augment_aligned_unlabeled_sites"])
    labeled_site_to_unlabeled_site = {}
    
    for index, row in df.iterrows():
        labeled_site = row["labeled_site"]
        labeled_site = convert_site_Xs_to_underscores(labeled_site)
        unlabeled_sites = row["unlabeled_site_identifiers"].split(";")
        unlabeled_sites = [convert_site_Xs_to_underscores(unlabaled_site.split("_")[0]) for unlabaled_site in unlabeled_sites]
        labeled_site_to_unlabeled_site[labeled_site] = unlabeled_sites
    
    return labeled_site_to_unlabeled_site
         

def load_phosphosite_data_from_csv_in_separate_rows_with_unlabaled_sites(filename, config):
    df = pd.read_csv(filename)

    labeled_site_to_unlabaled_site = get_aligned_unlabeled_data(config)
    
    phosphosite_ids = []
    sub_mods = []
    phosphosite_sequences = []
    kinase_ids = []
    unique_kinases = []
    total_augmented_aligned_unlabaled_data = 0
    for index, row in df.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = list(set(kinases.split(',')))
        kinases.sort()
        for kinase in kinases:
            site = row["SITE_+/-7_AA"].upper()
            if site in labeled_site_to_unlabaled_site:
                for unlabaled_site in labeled_site_to_unlabaled_site[site]:
                    phosphosite_ids.append(row["SUB_ACC_ID"])
                    sub_mods.append(row["SUB_MOD_RSD"])
                    phosphosite_sequences.append(unlabaled_site)
                    kinase_ids.append(kinase)
                    total_augmented_aligned_unlabaled_data += 1
            phosphosite_ids.append(row["SUB_ACC_ID"])
            sub_mods.append(row["SUB_MOD_RSD"])
            phosphosite_sequences.append(row["SITE_+/-7_AA"])
            kinase_ids.append(kinase)
    
    unique_kinases = sorted(list(set(kinase_ids)))    

    print(f"Total aligned unlabeled data added into train : {total_augmented_aligned_unlabaled_data}")

    return {
        "phosphosite_ids" : phosphosite_ids,
        "sub_mods" : sub_mods,
        "phosphosite_sequences" : phosphosite_sequences,
        "kinase_ids" : kinase_ids,
        "unique_kinases" : unique_kinases
    }

def get_cluster_to_kinase_list():
    kinase_to_fine_Grained_cluster = load_kinase_to_fine_grained_cluster()
    all_clusters = sorted(list(set(kinase_to_fine_Grained_cluster.values())))
    cluster_to_kinase_set = {cluster : set() for cluster in all_clusters}
    for kinase, cluster in kinase_to_fine_Grained_cluster.items():
        cluster_to_kinase_set[cluster].add(kinase)
    
    cluster_to_kinase_list = {cluster : sorted(list(kinase_set)) for cluster, kinase_set in cluster_to_kinase_set.items()}
    return cluster_to_kinase_list

def get_unique_kinases(filename):
    kinase_set = set()
    
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        kinase_ids = set(row['KINASE_ACC_IDS'].split(','))
        kinase_set.update(kinase_ids)

    return kinase_set

def load_phosphosite_data_from_csv_in_separate_rows_by_adding_cluster_data_to_eachother(filename, config):
    df = pd.read_csv(filename)

    cluster_to_kinase_list = get_cluster_to_kinase_list()
    kinase_to_fine_Grained_cluster = load_kinase_to_fine_grained_cluster()
    unique_train_kinases = get_unique_kinases(filename)

    phosphosite_ids = []
    sub_mods = []
    phosphosite_sequences = []
    kinase_ids = []
    unique_kinases = []
    additional_data = 0
    for index, row in df.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = list(set(kinases.split(',')))
        kinases.sort()
        for kinase in kinases:
            phosphosite_ids.append(row["SUB_ACC_ID"])
            sub_mods.append(row["SUB_MOD_RSD"])
            phosphosite_sequences.append(row["SITE_+/-7_AA"])
            kinase_ids.append(kinase)
            fine_grained_cluster = kinase_to_fine_Grained_cluster[kinase]
            cluster_kinase_list = cluster_to_kinase_list[fine_grained_cluster]
            for similar_kinase in cluster_kinase_list:
                if similar_kinase in unique_train_kinases:
                    phosphosite_ids.append(row["SUB_ACC_ID"])
                    sub_mods.append(row["SUB_MOD_RSD"])
                    phosphosite_sequences.append(row["SITE_+/-7_AA"])
                    kinase_ids.append(similar_kinase)
                    additional_data += 1
    
    unique_kinases = sorted(list(set(kinase_ids)))    

    print(f"Total additional data into train : {additional_data}")

    return {
        "phosphosite_ids" : phosphosite_ids,
        "sub_mods" : sub_mods,
        "phosphosite_sequences" : phosphosite_sequences,
        "kinase_ids" : kinase_ids,
        "unique_kinases" : unique_kinases
    }

def load_phosphosite_data_from_csv_in_separate_rows(filename):
    df = pd.read_csv(filename)

    phosphosite_ids = []
    sub_mods = []
    phosphosite_sequences = []
    kinase_ids = []
    unique_kinases = []
    for index, row in df.iterrows():
        kinases = row["KINASE_ACC_IDS"]
        kinases = list(set(kinases.split(',')))
        kinases.sort()
        for kinase in kinases:
            phosphosite_ids.append(row["SUB_ACC_ID"])
            sub_mods.append(row["SUB_MOD_RSD"])
            phosphosite_sequences.append(row["SITE_+/-7_AA"])
            kinase_ids.append(kinase)
    
    unique_kinases = sorted(list(set(kinase_ids)))    

    return {
        "phosphosite_ids" : phosphosite_ids,
        "sub_mods" : sub_mods,
        "phosphosite_sequences" : phosphosite_sequences,
        "kinase_ids" : kinase_ids,
        "unique_kinases" : unique_kinases
    }


def load_kinase_data(filename, config=None):
    if filename.endswith('.txt'):
        return load_kinase_data_from_txt(filename)
    elif filename.endswith('.csv'):
        return load_kinase_data_from_csv(filename, config)
    else:
        raise Exception(f"Only .txt and .csv files are accepted")


def load_kinase_data_from_txt(filename):
    kinase_data = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            
            kinase_id = parts[0]
            kinase_data[kinase_id] = dict()
            kinase_data[kinase_id]["family"] = parts[1]
            kinase_data[kinase_id]["group"] = parts[2]
            kinase_data[kinase_id]["enzymes_vec"] = parts[3]
            kinase_data[kinase_id]["domain"] = parts[4]
            kinase_data[kinase_id]["kin2vec"] = parts[5]
            kinase_data[kinase_id]["pathway"] = parts[6]
    return kinase_data

def load_kinase_similarity(filename):
    df = pd.read_csv(filename, index_col=0)
    similarity_data = {}
    for kinase in df.columns:
        vector = torch.tensor(df[kinase].values, dtype=torch.float32)
        normalized_vector = vector / torch.norm(vector)  # L2 normalization
        similarity_data[kinase] = normalized_vector
    return similarity_data, df.shape[0]

def load_kinase_to_fine_grained_cluster():
    file = "Fine_Grained_Classes/kinase_to_cluster_v2.csv"
    kinase_to_fine_Grained_cluster = dict()
    df = pd.read_csv(file)
    for index, row in df.iterrows():
        kinase = row["KIN_ACC_ID"]
        cluster = row["Fine_Grained_Cluster"]
        kinase_to_fine_Grained_cluster[kinase] = cluster

    return kinase_to_fine_Grained_cluster

def load_kinase_fine_grained_cluster_vectors(config):
    use_fine_grained_clustering_vector = config["kinase"]["dataset"]["processor"].get("use_fine_grained_clustering_binary_vector", False)
    if not use_fine_grained_clustering_vector:
        return {}
    df = pd.read_csv(config["kinase"]["fine_grained_clustering_vector_file"])
    kinase_to_vector = {}
    for index, row in df.iterrows():
        kinase = row['Kinase']
        vector = list(map(int, list(row['Fine_Grained_Cluster_Vector'])))
        tensor = torch.tensor(vector, dtype=torch.float32)
        kinase_to_vector[kinase] = tensor
    return kinase_to_vector

def load_kinase_data_from_csv(filename, config=None):
    # Kinase Similarity Data Loading
    kinase_to_fine_Grained_cluster = load_kinase_to_fine_grained_cluster()
    kinase_to_fine_grained_cluster_vector = load_kinase_fine_grained_cluster_vectors(config)
    kinase_similarity_file = f"dataset/new_dataset/kinase_pairwise_identity_similarity_scores.csv"
    try:
        similarity_data, similarity_size = load_kinase_similarity(kinase_similarity_file)
    except Exception as e:
        print(f"Error loading kinase similarity data: {e}")
        similarity_data = {}
        similarity_size = 1

    kinase_data = {}
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        uniprot_id = row['Kinase']
        data_dict = {
            "fine_grained_cluster": kinase_to_fine_Grained_cluster[uniprot_id],
            "family": row['Family'],
            "group": row['Group'],
            "enzymes_vec": torch.tensor(list(map(float, list(row['EC']))), dtype=torch.float32),
            "domain" : row['Kinase_Domain'],
            "kin2vec": torch.stack([torch.tensor(float(value), dtype=torch.float32) for value in row['Kin2Vec'].strip("[]").split()]),
            "pathway": torch.tensor([int(bit) for bit in row['Kegg_Pathways']], dtype=torch.float32),
            "kinase_similarity_vector": similarity_data.get(uniprot_id, torch.empty(similarity_size, dtype=torch.float32)),
            "kinase_fine_grained_cluster_vector": kinase_to_fine_grained_cluster_vector.get(uniprot_id, torch.empty(1, dtype=torch.float32))
        }
        if 'active_site' in df.columns:
            data_dict["active_site"] = row['active_site']
            if not pd.isna(row["active_site_indices"]):
                data_dict["active_site_indices"] = torch.stack(
                    [torch.tensor(int(value), dtype=torch.int) for value in row["active_site_indices"].strip("[]").split(",")]
                )
            else:
                data_dict["active_site"] = None
                data_dict["active_site_indices"] = None

            data_dict["active_site_kin2vec"] = torch.stack(
                [torch.tensor(float(value), dtype=torch.float32) for value in row['Kinase_Active_Site_Kin2vec'].strip("[]").split()]
            )
            data_dict["active_site_from_context_kin2vec"] = torch.stack(
                [torch.tensor(float(value), dtype=torch.float32) for value in row['Kinase_Active_Site_Kin2vec_with_context'].strip("[]").split()]
            )
        

        kinase_data[uniprot_id] = data_dict
    return kinase_data


def encode_kinase_labels(kinase_ids, unique_kinases):
    encoded_labels = []
    for kinase_id in kinase_ids:
        labels = kinase_id.split(',')
        binary_encoding = [1 if label in labels else 0 for label in unique_kinases]
        encoded_labels.append(binary_encoding)
    return torch.tensor(encoded_labels, dtype=torch.int8)


def get_processor(processor_config):
    processor_type = processor_config['processor_type']
    if processor_type == 'protvec':
        return ProtVecProcessor(processor_config)
    elif processor_type == 'hf':
        return HFProcessor(processor_config)
    elif processor_type == 'msa':
        return MsaProcessor(processor_config)
    else:
        return None


def read_embedding_from_feather(filepath, sequences):
    df_from_feather = pd.read_feather(filepath)
    sequence_embedding_dict = dict()
    for _, row in df_from_feather.iterrows():
        sequence = row['sequence']
        embedding = row['embedding']
        sequence_embedding_dict[sequence.upper()] = torch.stack(embedding)
    embeddings = torch.stack([sequence_embedding_dict[seq] for seq in sequences])
    return embeddings


def read_torch_embedding(embedding_path, sequences, embedding_mode):
    try:
        loaded_embeddings_dict = {key.upper(): value for key, value in torch.load(embedding_path).items()}
        result_list = []
        for sequence in sequences:
            embedding_tensor = loaded_embeddings_dict.get(sequence.upper(), None)
            if embedding_tensor is not None:
                embedding_tensor = select_embedding_slice(
                    embedding_tensor,
                    embedding_mode,
                    len(sequence)
                )
                result_list.append(embedding_tensor)
            else:
                print(f"Embedding for sequence '{sequence}' not found.")
        
        result_tensor = torch.stack(result_list)
        return result_tensor
    except Exception as e:
        print(f"Error reading torch embedding: {e}")
        return None
    

def read_pickle_embedding(embedding_path, sequences):
    try:
        with open(embedding_path, 'rb') as file:
            loaded_embeddings_dict = pickle.load(file)
        result_list = []
        for sequence in sequences:
            embedding_data = loaded_embeddings_dict.get(sequence, None)
            if embedding_data is not None:
                result_list.append(torch.tensor(embedding_data, dtype=torch.float32))
            else:
                print(f"Embedding for sequence '{sequence}' not found.")
        
        result_tensor = torch.stack(result_list)
        return result_tensor
    except Exception as e:
        print(f"Error reading torch embedding: {e}")
        return None


def get_active_sites(embedding_path, kinase_sequences, kinase_active_site_sequences, kinase_active_site_indices, from_context, embedding_mode):
    try:
        loaded_embeddings_dict = torch.load(embedding_path)
        result_list = []
        if from_context:
            for idx, sequence in enumerate(kinase_sequences):
                embedding_tensor = loaded_embeddings_dict.get(sequence, None)
                if embedding_tensor is not None:
                    active_sites_embeddings = []
                    data_shape = embedding_tensor.size()
                    active_site_embedding = torch.tensor([])
                    model_name = f'{embedding_path.split("/")[-1]}'
                    if 'ProtT5_XL' not in model_name:                   
                        # first get the cls token
                        if len(data_shape) == 2:
                            active_site_embedding = embedding_tensor[0, :]
                        elif len(data_shape) == 3:
                            active_site_embedding = embedding_tensor[:, 0, :]
                        # first add the cls token
                        active_sites_embeddings.append(active_site_embedding)
                        active_site_locator = 1
                    else:
                        active_site_locator = 0
                    active_site_indices = kinase_active_site_indices[idx]

                    if active_site_indices != None:
                        for residue_num in active_site_indices:
                            # residue num -1 olabilir, bu durumda da +1 yapinca 0'a cikip clsi alir bence fena bir mantik degil
                            actual_index = residue_num + active_site_locator # first token is cls
                            if actual_index == -1:
                                continue
                            if len(data_shape) == 2:
                                residue_embedding = embedding_tensor[actual_index, :]
                            elif len(data_shape) == 3:
                                residue_embedding = embedding_tensor[:, actual_index, :]
                            active_sites_embeddings.append(residue_embedding)
                        active_site_embedding = torch.stack(active_sites_embeddings)
                    else:
                        active_site_embedding = embedding_tensor
                    # active site embedding with context is created now
                    # embedding mode could be avg, or cls_avg
                    embedding_tensor = select_embedding_slice_active_site(
                        active_site_embedding,
                        embedding_mode
                    )
                    result_list.append(embedding_tensor)
                else:
                    print(f"Embedding for sequence '{sequence}' not found.")
            result_tensor = torch.stack(result_list)
            return result_tensor
        else:
            # not from context, this means I will get the embedding of the 29 residue active sites directly
            for idx, active_site in enumerate(kinase_active_site_sequences):
                # active site is equal to kinase sequence if not found (handled in kinase_embedding_generator.py)
                embedding_tensor = loaded_embeddings_dict.get(active_site, None)

                if embedding_tensor is not None:
                    # embedding mode could be cls, avg, or cls_avg. There is no padding mask
                    #embedding_tensor = select_embedding_slice_active_site(
                    #    embedding_tensor,
                    #    embedding_mode
                    #)
                    # I added here not to average the paddings for the active site embeddings
                    embedding_tensor = select_embedding_slice(
                        embedding_tensor,
                        embedding_mode,
                        len(active_site)
                    )
                    result_list.append(embedding_tensor)
                else:
                    print(f"Embedding for sequence '{active_site}' not found.")

        result_tensor = torch.stack(result_list)
        return result_tensor
    except Exception as e:
        print(f"Error reading torch embedding: Burda hata veriyor!!! {e}")
        return None


def select_embedding_slice(embedding_tensor, embedding_mode, sequence_length):
    data_shape = embedding_tensor.size()
    if embedding_mode != 'sequence':
        if embedding_mode == 'cls':
            if len(data_shape) == 2:
                return embedding_tensor[0, :]
            elif len(data_shape) == 3:
                return embedding_tensor[:, 0, :]
            else:
                raise NotImplementedError
        elif embedding_mode == 'avg':
            if isinstance(sequence_length, torch.Tensor):
                # For cls token
                sequence_length = sequence_length + 1
                # Mask tensor to zero out padded elements
                mask = torch.arange(embedding_tensor.size(1))[None, :].to(embedding_tensor.device) < sequence_length[:, None]
                # Apply mask to input tensor
                masked_x = embedding_tensor * mask.unsqueeze(2)
                # Compute sum along the second dimension
                sum_along_dim2 = masked_x.sum(dim=1)
                # Count non-padded elements along the second dimension
                count_non_padded = mask.sum(dim=1, dtype=torch.float)
                # Compute average along the second dimension
                return sum_along_dim2 / count_non_padded[:, None]
            elif isinstance(sequence_length, int):
                if sequence_length + 2 <= embedding_tensor.size()[0]:
                    return torch.mean(embedding_tensor[:sequence_length+2], dim=0)
                elif sequence_length <= embedding_tensor.size()[0]:
                    return torch.mean(embedding_tensor[:sequence_length], dim=0)
                else:
                    raise NotImplementedError
        elif embedding_mode == 'middle':
            if len(data_shape) == 2:
                middle_index = data_shape[0]//2
                return embedding_tensor[middle_index, :]
            elif len(data_shape) == 3:
                middle_index = data_shape[1]//2
                return embedding_tensor[:, middle_index, :]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    return embedding_tensor


def select_embedding_slice_active_site(embedding_tensor, embedding_mode):
    data_shape = embedding_tensor.size()
    if embedding_mode != 'sequence':
        if embedding_mode == 'cls':
            if len(data_shape) == 2:
                return embedding_tensor[0, :]
            elif len(data_shape) == 3:
                return embedding_tensor[:, 0, :]
            else:
                raise NotImplementedError
        elif embedding_mode == 'avg':
            # Compute the average of all embeddings except the cls token (embedding_tensor[1:])
            return torch.mean(embedding_tensor[1:], dim=0)
        elif embedding_mode == 'cls_avg':
            return torch.mean(embedding_tensor, dim=0)
        else:
            raise NotImplementedError
    return embedding_tensor