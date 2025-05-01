import torch
import torch.nn as nn
from transformers import AutoModel, EsmModel, T5EncoderModel, BertModel
from transformers import AutoConfig, T5Config, BertConfig

from scripts.utils.data_utils import select_embedding_slice

class HFModel(nn.Module):
    def __init__(self, model_name, embedding_mode='avg', is_pretrained=False):
        super(HFModel, self).__init__()
        
        self.embedding_mode = embedding_mode
        
        if model_name.startswith('esm'):
            if is_pretrained:
                self.embedding_model = EsmModel.from_pretrained(f"facebook/{model_name}")
            else:
                self.embedding_model = EsmModel(config = AutoConfig.from_pretrained(f"facebook/{model_name}"))
        elif model_name == 'prott5xl':
            if is_pretrained:
                self.embedding_model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
            else:
                self.embedding_model = T5EncoderModel(config = T5Config.from_pretrained('Rostlab/prot_t5_xl_uniref50'))
        elif model_name == 'protbert':
            if is_pretrained:
                self.embedding_model = BertModel.from_pretrained('Rostlab/prot_bert')
            else:
                self.embedding_model = BertModel(config = BertConfig.from_pretrained('Rostlab/prot_bert'))
        elif model_name == 'distilprotbert':
            if is_pretrained:
                self.embedding_model = BertModel.from_pretrained('yarongef/DistilProtBert')
            else:
                self.embedding_model = BertModel(config = BertConfig.from_pretrained('yarongef/DistilProtBert'))
        elif model_name == 'saprot':
            if is_pretrained:
                self.embedding_model = EsmModel.from_pretrained('Takagi-san/SaProt_650M_AF2')
            else:
                self.embedding_model = EsmModel(config = AutoConfig.from_pretrained('Takagi-san/SaProt_650M_AF2'))
        elif model_name == 'protalbert':
            raise NotImplementedError
            #if is_pretrained:
            #    self.embedding_model = AutoModel.from_pretrained('Rostlab/prot_albert')
            #else:
            #    self.embedding_model = AutoModel(config = AutoConfig.from_pretrained('Rostlab/prot_albert'))
        elif model_name == 'protgpt2':
            raise NotImplementedError
            #if is_pretrained:
            #    self.embedding_model = AutoModel.from_pretrained('nferruz/ProtGPT2')
            #else:
            #    self.embedding_model = AutoModel(config = AutoConfig.from_pretrained('nferruz/ProtGPT2'))

    def forward(self,X, sequence_lengths, attention_mask=None):
        X = self.embedding_model(input_ids = X, attention_mask = attention_mask).last_hidden_state
        X = select_embedding_slice(
            embedding_tensor = X,
            embedding_mode = self.embedding_mode,
            sequence_length = sequence_lengths
        )
        return X
    
    def _freeze_embedding_model(self):
        for param in self.embedding_model.parameters():
            param.requires_grad = False
    
    def _unfreeze_given_encoder_layers(self, unfreeze_layer_list = [4,5]):
        '''
            EXPERIMENTAL
            Unfreeze specified layers in Esm models.
        '''
        if len(unfreeze_layer_list) != 0:
            for name, param in self.embedding_model.named_parameters():
                if name.startswith('encoder.layer'):
                    layer_index = name.split('.')[2]
                    if layer_index in unfreeze_layer_list:
                        param.requires_grad_(True)
                #if name.startswith('encoder.emb_layer_norm_after'):
                #    param.requires_grad = True
                #if name in ['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight']:
                #    param.requires_grad = True