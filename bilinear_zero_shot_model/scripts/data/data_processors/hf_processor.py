import re
from transformers import AutoTokenizer, T5Tokenizer, BertTokenizer, AlbertTokenizer, EsmTokenizer

from scripts.data.data_processors.base_processor import SequenceProcessor

class HFProcessor(SequenceProcessor):
    def __init__(self, processor_config):
        super().__init__(config=processor_config)
        self.checkpoint_name = self.config['model_name']

        if self.checkpoint_name.startswith('esm'):
            self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{self.checkpoint_name}")
        elif self.checkpoint_name == 'prott5xl':
            self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        elif self.checkpoint_name == 'protbert':
            self.tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')
        elif self.checkpoint_name == 'distilprotbert':
            self.tokenizer = BertTokenizer.from_pretrained('yarongef/DistilProtBert')
        elif self.checkpoint_name == 'saprot':
            self.tokenizer = EsmTokenizer.from_pretrained('Takagi-san/SaProt_650M_AF2')
        elif self.checkpoint_name == 'protalbert':
            raise NotImplementedError # There is a bug in tokenizer I guess
            #self.tokenizer = AlbertTokenizer.from_pretrained('Rostlab/prot_albert')
        elif self.checkpoint_name == 'protgpt2':
            raise NotImplementedError
            #self.tokenizer = AutoTokenizer.from_pretrained('nferruz/ProtGPT2')

    def process_phosphosite_sequence(self, sequences, to_sequence_model=False):
        if self.checkpoint_name.startswith('esm'):
            processed_data = [seq.replace('_', '-').upper() for seq in sequences]
        elif self.checkpoint_name == 'saprot':
            processed_data = sequences
        else:
            processed_data = [" ".join(re.sub(r"[UZOBuzob]", "X", seq).replace('_', '-').upper()) for seq in sequences]
        batch_tokens = self.tokenizer(processed_data, padding=True, return_tensors='pt')['input_ids']
        return batch_tokens

    def process_kinase_sequence(self, sequences):
        if self.checkpoint_name.startswith('esm'):
            processed_data = [seq.replace('_', '-').upper() for seq in sequences]
        elif self.checkpoint_name == 'saprot':
            processed_data = sequences
        else:
            processed_data = [" ".join(re.sub(r"[UZOBuzob]", "X", seq).replace('_', '-').upper()) for seq in sequences]
        batch_tokens = self.tokenizer(processed_data, padding=True, return_tensors='pt')
        return batch_tokens['input_ids'], batch_tokens['attention_mask']
    