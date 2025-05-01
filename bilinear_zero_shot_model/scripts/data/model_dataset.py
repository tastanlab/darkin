import os
import pickle
import random
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn.functional as F
from scripts.utils.data_utils import load_phosphosite_data, load_kinase_data, load_phosphosite_data_in_separate_rows, get_processor, encode_kinase_labels, read_torch_embedding, select_embedding_slice
from scripts.data.data_processors.kinase_embedding_generator import KinaseEmbeddingGenerator

class ZeroShotDataset(torch.utils.data.Dataset):
    def __init__(self, phosphosite_data, kinase_data, labels, kinase_info_dict, phosphosite_data_dict, class_counts):
        
        # Data Properties
        self.phosphosite_data = phosphosite_data
        self.kinase_data = kinase_data
        self.labels = labels
        self.unseen_data = {
            'sequences' : [],
            'properties' : [],
            'active_sites' : [],
            'sequence_lengths' : [],
            'active_site_lengths' : [],
            'att_mask_sequences' : [],
            'att_mask_active_sites' : []
        }

        for unique in phosphosite_data_dict['unique_kinases']:
            self.unseen_data['sequences'].append(kinase_data[unique]['sequences'])
            self.unseen_data['properties'].append(kinase_data[unique]['properties'])
            self.unseen_data['active_sites'].append(kinase_data[unique]['active_sites'])
            self.unseen_data['sequence_lengths'].append(kinase_data[unique]['sequence_length'])
            self.unseen_data['active_site_lengths'].append(kinase_data[unique]['active_site_length'])
            self.unseen_data['att_mask_sequences'].append(kinase_data[unique]['att_mask_sequences'])
            self.unseen_data['att_mask_active_sites'].append(kinase_data[unique]['att_mask_active_sites'])

        self.unseen_data['sequences'] = torch.stack(self.unseen_data['sequences'], dim=0)
        self.unseen_data['properties'] = torch.stack(self.unseen_data['properties'], dim=0)
        self.unseen_data['active_sites'] = torch.stack(self.unseen_data['active_sites'], dim=0)
        self.unseen_data['sequence_lengths'] = torch.tensor(self.unseen_data['sequence_lengths'])
        self.unseen_data['active_site_lengths'] = torch.tensor(self.unseen_data['active_site_lengths'])
        self.unseen_data['att_mask_sequences'] = torch.stack(self.unseen_data['att_mask_sequences'], dim=0)
        self.unseen_data['att_mask_active_sites'] = torch.stack(self.unseen_data['att_mask_active_sites'], dim=0)
        
        # Information Properties
        self.kinase_info_dict = kinase_info_dict
        self.phosphosite_data_dict = phosphosite_data_dict
        self.label_mapping = {i : kinase_id for i, kinase_id in enumerate(phosphosite_data_dict['unique_kinases'])}
        self.label2idx_mapping = {kinase_id : i for i, kinase_id in enumerate(phosphosite_data_dict['unique_kinases'])}
        self.class_counts = class_counts

        # Scalers
        self.phosphosite_embed_scaler = None
        self.kinase_embed_scaler = None
        
    def __len__(self):
        return len(self.phosphosite_data)

    def __getitem__(self, idx):
        random_selected_kinase = random.choice(self.phosphosite_data_dict['kinase_ids'][idx].split(','))
        selected_kinase_data = self.kinase_data[random_selected_kinase]

        kinase_sequences = torch.tensor([]) if len(selected_kinase_data['sequences']) == 0 else selected_kinase_data['sequences']
        kinase_properties = torch.tensor([]) if len(selected_kinase_data['properties']) == 0 else selected_kinase_data['properties']
        kinase_active_sites = torch.tensor([]) if len(selected_kinase_data['active_sites']) == 0 else selected_kinase_data['active_sites']

        # Masks
        kinase_sequence_mask = torch.tensor([]) if len(selected_kinase_data['sequences']) == 0 else selected_kinase_data['att_mask_sequences']
        kinase_active_site_mask = torch.tensor([]) if len(selected_kinase_data['active_sites']) == 0 else selected_kinase_data['att_mask_active_sites']

        # Purpose of lengths are to properly getting the average embeddings from protein sequences of different lengths
        phosphosite_length = len(self.phosphosite_data_dict['phosphosite_sequences'][idx])
        kinase_length = torch.tensor([])
        active_site_length = torch.tensor([])

        if len(selected_kinase_data['sequences']) != 0:
            kinase_length = len(self.kinase_info_dict[random_selected_kinase]['domain'])
        if len(selected_kinase_data['active_sites']) == 0:
            if self.kinase_info_dict[random_selected_kinase]['active_site'] is not None:
                active_site_length = len(self.kinase_info_dict[random_selected_kinase]['active_site'])
            else:
                active_site_length = len(self.kinase_info_dict[random_selected_kinase]['domain'])

        return {
            'phosphosites' : self.phosphosite_data[idx],
            'kinases' : {
                'sequences' : kinase_sequences,
                'properties' : kinase_properties,
                'active_sites' : kinase_active_sites,
                'att_mask_sequences' : kinase_sequence_mask,
                'att_mask_active_sites' : kinase_active_site_mask
            },
            'labels' : self.labels[idx],
            'sequence_lengths' : {
                'phosphosite' : phosphosite_length,
                'kinase' : kinase_length,
                'active_site' : active_site_length
            },
            'kinase_idx' : self.label2idx_mapping[random_selected_kinase]
        }

    def _save_embed_scaler(self, config, embed_scaler, data_type):
        scaler_path = config['logging']['local']['saved_model_path']
        ckpt_name = config['logging']['local']['checkpoint_file_name']
        os.makedirs(scaler_path, exist_ok=True)

        # Save the scaler
        scaler_file_path = os.path.join(
            scaler_path,
            ckpt_name,
            f'embed_scaler_{data_type}.pkl')
        with open(scaler_file_path, 'wb') as f:
            pickle.dump(embed_scaler, f)

        print(f"Scaler saved to {scaler_file_path}")

    def _load_embed_scaler(self, config, data_type):
        scaler_path = config['logging']['local']['saved_model_path']
        ckpt_name = config['logging']['local']['checkpoint_file_name']
        scaler_file_path = os.path.join(
            scaler_path,
            ckpt_name,
            f'embed_scaler_{data_type}.pkl')
        if not os.path.exists(scaler_file_path):
            print(f"Scaler file not found at {scaler_file_path}")
            return None
        with open(scaler_file_path, 'rb') as f:
            embed_scaler = pickle.load(f)
        return embed_scaler

    def _normalize_data(self, config, fit_to_data=False):
        if config['training']['normalize_phosphosite_data'] and not config['helpers']['phosphosite']['has_token_id']:
            self._normalize_phosphosite_data(config, fit_to_data = fit_to_data)
        if config['training']['normalize_kinase_data'] and not config['helpers']['kinase']['has_token_id']:
            self._normalize_kinase_data(config, fit_to_data = fit_to_data)

    def _normalize_phosphosite_data(self, config, fit_to_data=False):
        data_shape = self.phosphosite_data.size()
        if len(data_shape) == 3:
            self.phosphosite_data = self.phosphosite_data.view((data_shape[0], data_shape[1] * data_shape[2]))
        if fit_to_data:
            self.phosphosite_embed_scaler = preprocessing.StandardScaler().fit(self.phosphosite_data)
            self._save_embed_scaler(config, self.phosphosite_embed_scaler, data_type = 'phosphosite')
        else:
            self.phosphosite_embed_scaler = self._load_embed_scaler(config, data_type = 'phosphosite')
        self.phosphosite_data = self.phosphosite_embed_scaler.transform(self.phosphosite_data)
        self.phosphosite_data = torch.from_numpy(self.phosphosite_data).to(torch.float32)
        if len(data_shape) == 3:
            self.phosphosite_data = self.phosphosite_data.view((data_shape[0], data_shape[1], data_shape[2]))
    
    def _normalize_kinase_data(self, config, fit_to_data=False):
        data_shape = self.kinase_data.size()
        if len(data_shape) == 3:
            self.kinase_data = self.kinase_data.view((data_shape[0], data_shape[1] * data_shape[2]))
        if fit_to_data:
            self.kinase_embed_scaler = preprocessing.StandardScaler().fit(self.kinase_data)
            self._save_embed_scaler(config, self.kinase_embed_scaler, data_type = 'kinase')
        else:
            self.kinase_embed_scaler = self._load_embed_scaler(config, data_type = 'kinase')
        self.kinase_data = self.kinase_embed_scaler.transform(self.kinase_data)
        self.kinase_data = torch.from_numpy(self.kinase_data).to(torch.float32)
        if len(data_shape) == 3:
            self.kinase_data = self.kinase_data.view((data_shape[0], data_shape[1], data_shape[2]))


def create_zero_shot_dataset(
    config,
    data_type
):
    # File Paths
    phosphosite_filename = config['phosphosite']['dataset'][data_type]
    kinase_filename = config['kinase']['dataset'][data_type]
    phosphosite_processor_config = config['phosphosite']['dataset']['processor']
    kinase_processor_config = config['kinase']['dataset']['processor']

    # Precalculated encoders if validation or test data is being used
    encoders = None
    if data_type in ['validation', 'test']:
        kinase_encoder_path = os.path.join(
            config['logging']['local']['saved_model_path'],
            config['logging']['local']['checkpoint_file_name'],
            "kinase_encoder.pkl"            
        )
        with open(kinase_encoder_path, 'rb') as file:
            encoders = pickle.load(file)
    
    # Load data
    if data_type in ["train", "train_val"]:
        if config['phosphosite']['dataset']['processor'].get('split_multilabel_rows', True):
            phosphosite_data_dict = load_phosphosite_data_in_separate_rows(phosphosite_filename, config)
        else:
            phosphosite_data_dict = load_phosphosite_data(phosphosite_filename)
    else:
        phosphosite_data_dict = load_phosphosite_data(phosphosite_filename)
    kinase_data_dict = load_kinase_data(kinase_filename, config)

    # Process phosphosite
    if phosphosite_processor_config['read_embeddings']:
        phosphosite_data = read_torch_embedding(
            phosphosite_processor_config['phosphosite_embedding_path'],
            phosphosite_data_dict['phosphosite_sequences'],
            config['phosphosite']['model']['embedding_mode']
        )
    else:
        phosphosite_processor = get_processor(phosphosite_processor_config)
        phosphosite_data = phosphosite_processor.process_phosphosite_sequence(
            phosphosite_data_dict['phosphosite_sequences'],
            config['phosphosite']['sequence_model']['use_sequence_model']
        )
        
    # Process kinase
    kinase_embedding_generator = KinaseEmbeddingGenerator(kinase_data_dict, kinase_processor_config, encoders)

    # Unseen kinases
    kinase_data = kinase_embedding_generator.create_kinase_embeddings(
        unique_kinase_ids=phosphosite_data_dict['unique_kinases'],
        embedding_mode=config['kinase']['model']['embedding_mode']
    )

    # Encode kinase labels
    kinase_labels = encode_kinase_labels(phosphosite_data_dict['kinase_ids'], phosphosite_data_dict['unique_kinases'])

    # Calculate class counts for each kinase (If data_type is train, we calculate the class counts for each kinase, for validation and test data, we calculate the most similar train class and use its class counts)
    class_counts = calculate_train_class_counts(data_type, config)
    
    # If it is training data, we save the encoders for validation and test
    if data_type in ['train', 'train_val']:
        encoders = {
            'family' : kinase_embedding_generator.family_encoder,
            'group' : kinase_embedding_generator.group_encoder
        }
        # directory = '/'.join(config['logging']['local']['kinase_encoder_save_path'].split('/')[:-1])
        save_filepath = os.path.join(
            config['logging']['local']['saved_model_path'],
            f"{config['logging']['local']['checkpoint_file_name']}"            
        )
        os.makedirs(save_filepath, exist_ok=True)
        with open(os.path.join(save_filepath, f'kinase_encoder.pkl'), 'wb') as file:
            pickle.dump(encoders, file)

    return ZeroShotDataset(
        phosphosite_data = phosphosite_data,
        kinase_data = kinase_data,
        labels = kinase_labels,
        kinase_info_dict = kinase_data_dict,
        phosphosite_data_dict = phosphosite_data_dict,
        class_counts = class_counts
    )


def calculate_train_class_counts(data_type, config):
    '''
        For train data, we calculate the class counts for each kinase
        For validation and test data, we calculate the most similar train class and use its class counts
    '''
    assert data_type in ['train', 'validation', 'test', 'train_val']

    def get_kinase_count(filename):
        df = pd.read_csv(filename)

        kinase_counts = {}
        for _, row in df.iterrows():
            kinases = list(set(row["KINASE_ACC_IDS"].split(',')))
            for kinase in kinases:
                if kinase not in kinase_counts:
                    kinase_counts[kinase] = 0
                kinase_counts[kinase] += 1
        
        unique_kinases = sorted(list(kinase_counts.keys()))    
        return kinase_counts, unique_kinases

    def get_family_group_counts(config, eval_unique_kinases, train_kinase_counts):
        assert config['hyper_parameters']['loss_weight_type'] in ['family', 'group']
        # Read Kinase Data
        kinase_data = {}
        df = pd.read_csv(config['kinase']['dataset']['train'])
        for _, row in df.iterrows():
            kinase_data[row['Kinase']] = {"family": row['Family'], "group": row['Group']}
        # Calculate family or group count in training data
        eval_kinase_counts = {}
        for eval_kinase in eval_unique_kinases:
            total_count = 0
            if eval_kinase in kinase_data:
                eval_kin_att = kinase_data[eval_kinase][config['hyper_parameters']['loss_weight_type']]
                for train_kinase, count in train_kinase_counts.items():
                    if train_kinase in kinase_data:
                        train_kin_att = kinase_data[train_kinase][config['hyper_parameters']['loss_weight_type']]
                        if train_kin_att == eval_kin_att:
                            total_count += count
            eval_kinase_counts[eval_kinase] = total_count
        return eval_kinase_counts

    # Load phosphosite train data and get the unique kinases
    train_kinase_counts, train_unique_kinases = get_kinase_count(config['phosphosite']['dataset']['train'])

    if data_type in ['train', 'train_val']:
        if config['hyper_parameters'].get('loss_weight_type', 'pairwise_sim') == 'pairwise_sim':
            return train_kinase_counts
        else:
            return get_family_group_counts(config, train_unique_kinases, train_kinase_counts)
    else:
        similarity_threshold = 50
        eval_kinase_counts = {}
        _, eval_unique_kinases = get_kinase_count(config['phosphosite']['dataset'][data_type]) # Unique kinase list in evaluation dataset

        if config['hyper_parameters'].get('loss_weight_type', 'pairwise_sim') == 'pairwise_sim': # If not added to config file, it will use pairwise similarity counts as weights
            kinase_similarity_filename = "dataset/new_dataset/kinase_pairwise_identity_similarity_scores.csv"
            similarity_df = pd.read_csv(kinase_similarity_filename, index_col=0)

            for eval_kinase in eval_unique_kinases:
                total_count = 0
                for train_kinase, count in train_kinase_counts.items():
                    similarity = similarity_df.loc[eval_kinase, train_kinase]
                    if similarity >= similarity_threshold:
                        total_count += count
                eval_kinase_counts[eval_kinase] = total_count
        else:
            eval_kinase_counts = get_family_group_counts(config, eval_unique_kinases, train_kinase_counts)

        return eval_kinase_counts