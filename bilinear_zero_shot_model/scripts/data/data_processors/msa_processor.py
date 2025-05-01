import torch
from scripts.data.data_processors.base_processor import SequenceProcessor

class MsaProcessor(SequenceProcessor):
    def __init__(self, processor_config):
        super().__init__(config=processor_config)
        _, self.alphabet = torch.hub.load("facebookresearch/esm:main", 'esm_msa1b_t12_100M_UR50S')
        self.esm_batch_converter = self.alphabet.get_batch_converter()

    def process_phosphosite_sequence(self, sequences, to_sequence_model=False):
        pass

    def process_kinase_sequence(self, sequences):
        pass

