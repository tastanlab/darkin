import os, pickle

import torch
import torch.nn as nn

class Zero_DKZ(nn.Module):
    def __init__(self, w_dim, phosphosite_model, kinase_model, sequence_model, config):
        super(Zero_DKZ, self).__init__()
        
        self.W = torch.nn.Parameter(torch.rand(w_dim[0], w_dim[1]) * 0.05)
        self.phosphosite_model = phosphosite_model
        self.kinase_model = kinase_model
        self.sequence_model = sequence_model
        self.config = config
        #self.batchnorm = nn.BatchNorm1d(w_dim[0]-1) # experiment: batch norm adding


    def forward(self, phosphosites, kinases, unseen_kinases, sequence_lengths):
        '''
            Process phosphosite input
        '''
        if self.phosphosite_model is not None:
            if self.config['phosphosite']['model']['freeze'] and len(self.config['helpers']['phosphosite']['unfrozen_layers']) == 0 and not self.config['phosphosite']['model']['lora']:
                with torch.no_grad():
                    phosphosites = self.phosphosite_model(phosphosites, sequence_lengths['phosphosite'], None)
            else:
                phosphosites = self.phosphosite_model(phosphosites, sequence_lengths['phosphosite'], None)
            
            if self.config['helpers']['phosphosite']['has_token_id'] and self.config['training']['normalize_phosphosite_data']:
                #phosphosites = self.batchnorm(phosphosites) # experiment: batch norm adding
                #phosphosites = phosphosites / phosphosites.norm(dim=1, keepdim=True, p=2) # experiment: norm
                phosphosites = self._scale_embeddings(phosphosites, 'phosphosite')
        
        '''
            Process sequence of phosphosite embeddings
        '''
        if self.sequence_model is not None:
            phosphosites = self.sequence_model(phosphosites)
        
        '''
            Process kinase input
        '''
        if self.kinase_model is not None:
            if self.config['kinase']['model']['freeze'] and len(self.config['helpers']['kinase']['unfrozen_layers']) == 0 and not self.config['kinase']['model']['lora']:
                with torch.no_grad():
                    if kinases['sequences'].size()[1] > 0:
                        unseen_kinases['sequences'] = self._calculate_kinase_embeddings_batched(unseen_kinases['sequences'], unseen_kinases['sequence_lengths'], unseen_kinases['att_mask_sequences'])
                    if kinases['active_sites'].size()[1] > 0:
                        unseen_kinases['active_sites'] = self._calculate_kinase_embeddings_batched(unseen_kinases['active_sites'], unseen_kinases['active_site_lengths'], unseen_kinases['att_mask_active_sites'])
            else:
                if kinases['sequences'].size()[1] > 0:
                    unseen_kinases['sequences'] = self._calculate_kinase_embeddings_batched(unseen_kinases['sequences'], unseen_kinases['sequence_lengths'], unseen_kinases['att_mask_sequences'])
                if kinases['active_sites'].size()[1] > 0:
                    unseen_kinases['active_sites'] = self._calculate_kinase_embeddings_batched(unseen_kinases['active_sites'], unseen_kinases['active_site_lengths'], unseen_kinases['att_mask_active_sites'])

            if self.config['helpers']['kinase']['has_token_id'] and self.config['training']['normalize_kinase_data']:
                # kinases['sequences'] = kinases['sequences'] / kinases['sequences'].norm(dim=1, keepdim=True, p=2)
                # unseen_kinases['sequences'] = unseen_kinases['sequences'] / unseen_kinases['sequences'].norm(dim=1, keepdim=True, p=2) 
                if kinases['sequences'].size()[1] > 0:
                    unseen_kinases['sequences'] = self._scale_embeddings(unseen_kinases['sequences'], 'kinase')
                if kinases['active_sites'].size()[1] > 0:
                    unseen_kinases['active_sites'] = self._scale_embeddings(unseen_kinases['active_sites'], 'kinase')
        
        # Concatenate kinase embeddings and properties
        unseen_kinases = torch.cat(
            [unseen_kinases['sequences'], unseen_kinases['properties'], unseen_kinases['active_sites']], dim=1
        )

        # Add 1 for W matrix
        phosphosites = torch.nn.functional.pad(phosphosites, (0, 1), value=1)
        unseen_kinases = torch.nn.functional.pad(unseen_kinases, (0, 1), value=1)

        compatibility = torch.matmul(phosphosites, self.W)
        unique_logits = torch.matmul(compatibility, unseen_kinases.permute(1,0))
            
        return {
            'unique_logits':unique_logits
        }

    def _calculate_kinase_embeddings_batched(self, sequences, sequence_lengths, att_mask_sequences):
        chunk_size = 8
        unseen_kinases_sequences = []
        for i in range(0, sequences.size(0), chunk_size):
            with torch.no_grad():
                unseen_chunk = self.kinase_model(
                    sequences[i:i + chunk_size],
                    sequence_lengths[i:i + chunk_size],
                    att_mask_sequences[i:i + chunk_size]
                ).detach()
                unseen_kinases_sequences.append(unseen_chunk)
        return torch.cat(unseen_kinases_sequences, dim=0)
    
    def _scale_embeddings(self, embeddings, data_type):
        scaler_path = self.config['logging']['local']['saved_model_path']
        ckpt_name = self.config['logging']['local']['checkpoint_file_name']
        scaler_file_path = os.path.join(
            scaler_path,
            ckpt_name,
            f'embed_scaler_{data_type}.pkl')
        if not os.path.exists(scaler_file_path):
            print(f"Scaler file not found at {scaler_file_path}. Stopping..")
            quit()
        with open(scaler_file_path, 'rb') as f:
            embed_scaler = pickle.load(f)
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        std_vals = torch.from_numpy(embed_scaler.scale_).to(embeddings.dtype).to(device)
        mean_vals = torch.from_numpy(embed_scaler.mean_).to(embeddings.dtype).to(device)
        embeddings = (embeddings - mean_vals) / std_vals
        return embeddings
