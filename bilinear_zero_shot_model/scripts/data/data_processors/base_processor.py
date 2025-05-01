import torch
from abc import ABC, abstractmethod

class SequenceProcessor(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def process_phosphosite_sequence(self, sequences, to_sequence_model=False):
        pass
    
    @abstractmethod
    def process_kinase_sequence(self, sequences):
        pass
