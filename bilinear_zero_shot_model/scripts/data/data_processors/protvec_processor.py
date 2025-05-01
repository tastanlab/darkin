from scripts.data.data_processors.base_processor import SequenceProcessor
import numpy as np
import torch

class ProtVecProcessor(SequenceProcessor):
    def __init__(self, processor_config):
        super().__init__(config=processor_config)
        self.TrigramToVec = {}
        self.load_protvec_vectors(self.config.get('protvec_file_path', 'dataset/new_dataset/protvec_embeddings.txt'))

    def process_phosphosite_sequence(self, sequences, to_sequence_model=False):
        processed_sequences = []
        for seq in sequences:
            vec_mat = self.get_protvec_vectors(seq)
            padded_vec_mat = self.pad_vectors(seq, vec_mat)

            # Reshape the tensor to 1x1300 if to_sequence_model is True
            if not to_sequence_model:
                padded_vec_mat = padded_vec_mat.view(1, -1)

            processed_sequences.append(padded_vec_mat)
        return torch.stack(processed_sequences) if to_sequence_model else torch.cat(processed_sequences, dim=0)

    def process_kinase_sequence(self, sequences):
        return torch.tensor([]), torch.tensor([])

    def load_protvec_vectors(self, protvec_file_path):
        with open(protvec_file_path, 'r') as f:
            for line in f:
                trigram, vec_str = line.strip().split('\t')
                vec = list(map(np.float32, vec_str.split(" , ")))
                self.TrigramToVec[trigram.upper()] = torch.tensor(vec, dtype=torch.float32)

    def get_protvec_vectors(self, sequence):
        trigrams = [sequence[i:i+3] for i in range(len(sequence) - 2)]
        vec_mat = []
        for trigram in trigrams:
            if "_" not in trigram:
                vec_mat.append(self.TrigramToVec[''.join(trigram.upper())])
        return torch.stack(vec_mat)

    def pad_vectors(self, sequence, vec_mat):
        padded_vec_mat = []
        j = 0
        for i in range(len(sequence) - 2):
            if '_' in sequence[i:i+3]:
                padded_vec_mat.append(torch.zeros(100, dtype=torch.float32))
            else:
                padded_vec_mat.append(vec_mat[j])
                j += 1
        return torch.stack(padded_vec_mat)
