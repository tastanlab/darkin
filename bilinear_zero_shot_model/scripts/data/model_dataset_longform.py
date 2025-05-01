import os
import pickle
import random
from sklearn import preprocessing

import torch
import torch.nn.functional as F
from scripts.utils.data_utils import load_phosphosite_data, load_kinase_data, load_phosphosite_data_in_separate_rows, get_processor, encode_kinase_labels, read_pickle_embedding, read_torch_embedding
from scripts.data.data_processors.kinase_embedding_generator import KinaseEmbeddingGenerator

class ZeroShotDataset(torch.utils.data.Dataset):
    def __init__(self, phosphosite_data, kinase_data, labels, kinase_info_dict, phosphosite_data_dict):
        
        # Data Properties
        self.phosphosite_data = phosphosite_data
        self.kinase_data = kinase_data
        self.labels = labels
        self.unseen_data = {
            'sequences' : [],
            'properties' : [],
            'active_sites' : [],
            'sequence_lengths' : [],
            'active_site_lengths' : []
        }

        for unique in phosphosite_data_dict['unique_kinases']:
            self.unseen_data['sequences'].append(kinase_data[unique]['sequences'])
            self.unseen_data['properties'].append(kinase_data[unique]['properties'])
            self.unseen_data['active_sites'].append(kinase_data[unique]['active_sites'])
            self.unseen_data['sequence_lengths'].append(kinase_data[unique]['sequence_length'])
            self.unseen_data['active_site_lengths'].append(kinase_data[unique]['active_site_length'])

        self.unseen_data['sequences'] = torch.stack(self.unseen_data['sequences'], dim=0)
        self.unseen_data['properties'] = torch.stack(self.unseen_data['properties'], dim=0)
        self.unseen_data['active_sites'] = torch.stack(self.unseen_data['active_sites'], dim=0)
        self.unseen_data['sequence_lengths'] = torch.tensor(self.unseen_data['sequence_lengths'])
        self.unseen_data['active_site_lengths'] = torch.tensor(self.unseen_data['active_site_lengths'])
        
        # Information Properties
        self.kinase_info_dict = kinase_info_dict
        self.phosphosite_data_dict = phosphosite_data_dict
        self.label_mapping = {i : kinase_id for i, kinase_id in enumerate(phosphosite_data_dict['unique_kinases'])}

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
            'kinases' : {'sequences' : kinase_sequences, 'properties' : kinase_properties, 'active_sites' : kinase_active_sites},
            'labels' : self.labels[idx],
            'sequence_lengths' : {
                'phosphosite' : phosphosite_length,
                'kinase' : kinase_length,
                'active_site' : active_site_length
            }
        }

    def _save_embed_scaler(self, config, embed_scaler, data_type):
        scaler_path = config['logging']['local']['saved_model_path']
        os.makedirs(scaler_path, exist_ok=True)

        # Save the scaler
        scaler_file_path = os.path.join(
            scaler_path,
            f'embed_scaler_{data_type}.pkl'        )
        with open(scaler_file_path, 'wb') as f:
            pickle.dump(embed_scaler, f)

        print(f"Scaler saved to {scaler_file_path}")

    def _load_embed_scaler(self, config, data_type):
        scaler_path = config['logging']['local']['saved_model_path']
        scaler_file_path = os.path.join(
            scaler_path,
            f'embed_scaler_{data_type}.pkl'
        )
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
    phosphosite_long_form_filename = config['phosphosite']['dataset']["longform"]
    kinase_filename = config['kinase']['dataset'][data_type]
    phosphosite_processor_config = config['phosphosite']['dataset']['processor']
    kinase_processor_config = config['kinase']['dataset']['processor']
    

    # Precalculated encoders if validation or test data is being used
    encoders = None
    if data_type in ['validation', 'test']:
        with open(config['logging']['local']['kinase_encoder_save_path'], 'rb') as file:
            encoders = pickle.load(file)
    
    # Load data
    if data_type == "train":
        phosphosite_data_dict = load_phosphosite_data_in_separate_rows(config, phosphosite_filename, phosphosite_long_form_filename)
    else:
        phosphosite_data_dict = load_phosphosite_data(phosphosite_filename, phosphosite_long_form_filename)
    kinase_data_dict = load_kinase_data(kinase_filename)


    # Process phosphosite
    if phosphosite_processor_config['read_embeddings']:
        phosphosite_data = read_torch_embedding( ####### torch.stack(list of tensors)
            phosphosite_processor_config['phosphosite_embedding_path'],
            phosphosite_data_dict['phosphosite_sequences'],
            config['phosphosite']['model']['embedding_mode']
        )
    else:
        phosphosite_processor = get_processor(phosphosite_processor_config)
        phosphosite_data = phosphosite_processor.process_phosphosite_sequence(
            phosphosite_data_dict['256long_phosphosite_sequences'], #phosphosite_data_dict['phosphosite_sequences'],
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
    
    # If it is training data, we save the encoders for validation and test
    if data_type == 'train':
        encoders = {
            'family' : kinase_embedding_generator.family_encoder,
            'group' : kinase_embedding_generator.group_encoder
        }
        directory = '/'.join(config['logging']['local']['kinase_encoder_save_path'].split('/')[:-1])
        os.makedirs(directory, exist_ok=True)
        with open(config['logging']['local']['kinase_encoder_save_path'], 'wb') as file:
            pickle.dump(encoders, file)

    return ZeroShotDataset(
        phosphosite_data = phosphosite_data,
        kinase_data = kinase_data,
        labels = kinase_labels,
        kinase_info_dict = kinase_data_dict,
        phosphosite_data_dict = phosphosite_data_dict
    )
