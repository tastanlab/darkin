import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scripts.utils.data_utils import get_processor, read_torch_embedding, get_active_sites


class KinaseEmbeddingGenerator:
    def __init__(self, kinase_properties_dict, kinase_processor_config, encoders=None):
        self.kinase_properties_dict = kinase_properties_dict
        self.kinase_processor_config = kinase_processor_config
        self.processor = get_processor(kinase_processor_config)

        # Encode families and groups
        self.unique_families = sorted(list(set(item['family'] for item in kinase_properties_dict.values())))
        self.unique_groups = sorted(list(set(item['group'] for item in kinase_properties_dict.values())))
        self.encoded_family_dict, self.family_encoder = self._get_onehot_encoding_dict(self.unique_families, encoders['family'] if encoders is not None else None)
        self.encoded_group_dict, self.group_encoder = self._get_onehot_encoding_dict(self.unique_groups, encoders['group'] if encoders is not None else None)


    def create_kinase_embeddings(self, unique_kinase_ids, embedding_mode):
        kinase_sequences = []
        kinase_active_site_sequences, kinase_active_site_indices = [], []
        kinase_embeddings_dict = {
            kin_id : {'sequences': [], 'active_sites': [], 'properties': [], 'sequence_length': 0, 'active_site_length': 0} for kin_id in unique_kinase_ids
        }

        for kinase_id in unique_kinase_ids:
            kinase_properties = self.kinase_properties_dict.get(kinase_id, None)

            if kinase_properties:
                embeddings = []

                if self.kinase_processor_config.get('use_family', False):
                    family_embedding = self.process_family(kinase_properties['family'])
                    embeddings.append(family_embedding)

                if self.kinase_processor_config.get('use_group', False):
                    group_embedding = self.process_group(kinase_properties['group'])
                    embeddings.append(group_embedding)

                if self.kinase_processor_config.get('use_enzymes', False):
                    enzyme_embedding = self.process_enzymes(kinase_properties['enzymes_vec'])
                    embeddings.append(enzyme_embedding)
                
                if self.kinase_processor_config.get('use_kin2vec', False):
                    kin2vec_embedding = self.process_kin2vec(kinase_properties['kin2vec'])
                    embeddings.append(kin2vec_embedding)
                
                if self.kinase_processor_config.get('use_pathway', False):
                    pathway_embedding = self.process_pathway(kinase_properties['pathway'])
                    embeddings.append(pathway_embedding)

                if self.kinase_processor_config.get('use_kinase_similarity_vector', False):
                    process_kinase_similarity_embedding = self.process_kinase_similarity_vector(kinase_properties['kinase_similarity_vector'])
                    embeddings.append(process_kinase_similarity_embedding)
                
                if self.kinase_processor_config.get('use_fine_grained_clustering_binary_vector', False):
                    kinase_fine_grained_clustering_embedding = self.process_kinase_fine_grained_clustering_vector(kinase_properties['kinase_fine_grained_cluster_vector'])
                    embeddings.append(kinase_fine_grained_clustering_embedding)

                # Concatenate the embeddings into a single tensor
                if len(embeddings) > 0:
                    kinase_embeddings_dict[kinase_id]['properties'] = torch.cat(embeddings, dim=-1).to(torch.float32)
                else:
                    kinase_embeddings_dict[kinase_id]['properties'] = torch.tensor([])
            
                # For efficient using, instead of one by one, tokenization of sequence will be done with all sequence
                if self.kinase_processor_config['use_domain']:
                    kinase_sequences.append(kinase_properties['domain'])
                    kinase_embeddings_dict[kinase_id]['sequence_length'] = len(kinase_properties['domain'])

                if self.kinase_processor_config['active_site']['use_active_site']:
                    if kinase_properties['active_site'] is not None:
                        kinase_active_site_sequences.append(kinase_properties['active_site'])
                        kinase_active_site_indices.append(kinase_properties['active_site_indices'])
                        kinase_embeddings_dict[kinase_id]['active_site_length'] = len(kinase_properties['active_site'])
                    else:
                        kinase_active_site_sequences.append(kinase_properties['domain'])
                        kinase_active_site_indices.append(None)
                        kinase_embeddings_dict[kinase_id]['active_site_length'] = len(kinase_properties['domain'])
                    
                    if not self.kinase_processor_config['use_domain']: # This provide the whole kinase context for processing active sites
                        kinase_sequences.append(kinase_properties['domain'])
                        kinase_embeddings_dict[kinase_id]['sequence_length'] = len(kinase_properties['domain'])

                
        if self.kinase_processor_config['use_domain']:
            if self.kinase_processor_config['read_embeddings']:
                kinase_sequence_embeddings = read_torch_embedding(
                    self.kinase_processor_config['kinase_embedding_path'],
                    kinase_sequences,
                    embedding_mode
                )
                kinase_sequence_att_mask = None
            else:
                kinase_sequence_embeddings, kinase_sequence_att_mask = self.processor.process_kinase_sequence(kinase_sequences)

            for kin_idx, kinase_id in enumerate(unique_kinase_ids):
                kinase_embeddings_dict[kinase_id]['sequences'] = kinase_sequence_embeddings[kin_idx]
                kinase_embeddings_dict[kinase_id]['att_mask_sequences'] = kinase_sequence_att_mask[kin_idx] if kinase_sequence_att_mask is not None else torch.tensor([])

        else:
            for kin_idx, kinase_id in enumerate(unique_kinase_ids):
                kinase_embeddings_dict[kinase_id]['sequences'] = torch.tensor([])
                kinase_embeddings_dict[kinase_id]['att_mask_sequences'] = torch.tensor([])
        
        kinase_embeddings_dict = self.process_active_site(
            unique_kinase_ids,
            kinase_sequences,
            kinase_active_site_sequences,
            kinase_active_site_indices,
            kinase_embeddings_dict
        )

        return kinase_embeddings_dict
    

    def process_family(self, feature):
        return self.encoded_family_dict[feature]
    
    def process_group(self, feature):
        return self.encoded_group_dict[feature]

    def process_enzymes(self, feature):
        return feature

    def process_kin2vec(self, feature):
        return feature

    def process_pathway(self, feature):
        return feature

    def process_domain(self, feature):
        return feature
    
    def process_kinase_similarity_vector(self, feature):
        return feature
    
    def process_kinase_fine_grained_clustering_vector(self, feature):
        return feature
    
    def process_active_site(self, unique_kinase_ids, kinase_sequences, kinase_active_site_sequences, kinase_active_site_indices, kinase_embeddings_dict):
        if self.kinase_processor_config['active_site']['use_active_site']:
            if self.kinase_processor_config['read_embeddings']:
                kinase_active_site_embeddings = get_active_sites(
                    self.kinase_processor_config['kinase_embedding_path'],
                    kinase_sequences,
                    kinase_active_site_sequences, 
                    kinase_active_site_indices,
                    from_context=self.kinase_processor_config['active_site']['from_context'],
                    embedding_mode=self.kinase_processor_config['active_site']['embedding_mode']
                )
                active_site_att_mask = None
            else:
                kinase_active_site_embeddings, active_site_att_mask = self.processor.process_kinase_sequence(kinase_active_site_sequences)#??????

            for kin_idx, kinase_id in enumerate(unique_kinase_ids):
                kinase_embeddings_dict[kinase_id]['active_sites'] = kinase_active_site_embeddings[kin_idx]
                kinase_embeddings_dict[kinase_id]['att_mask_active_sites'] = active_site_att_mask[kin_idx] if active_site_att_mask is not None else torch.tensor([])

        else:
            for kin_idx, kinase_id in enumerate(unique_kinase_ids):
                kinase_embeddings_dict[kinase_id]['active_sites'] = torch.tensor([])
                kinase_embeddings_dict[kinase_id]['att_mask_active_sites'] = torch.tensor([])

        return kinase_embeddings_dict

    
    def _get_onehot_encoding_dict(self, class_list, encoders=None):
        onehot_encoded, encoders = self._onehot_encode(class_list, encoders)
        class_onehot_dict = {encoded_class: torch.from_numpy(onehot_encoded[i]) for i, encoded_class in enumerate(encoders['label'].classes_)}
        return class_onehot_dict, encoders


    def _onehot_encode(self, class_list, encoders=None):
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
        return onehot_encoded, encoders
