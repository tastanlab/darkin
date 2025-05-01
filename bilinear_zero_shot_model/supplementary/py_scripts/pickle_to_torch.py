import torch
import pickle

def convert_pickle_to_torch(embedding_path):
    try:
        with open(embedding_path, 'rb') as file:
            loaded_embeddings_dict = pickle.load(file)
        converted_embeddings_dict = {key: torch.tensor(value, dtype=torch.float32) for key, value in loaded_embeddings_dict.items()}
        torch.save(converted_embeddings_dict, 'proteinbert_kinase.pt')
    except Exception as e:
        print(f"Error reading torch embedding: {e}")
        return None

def read_converted_torch_file(embedding_path):
    try:
        with open(embedding_path, 'rb') as file:
            loaded_embeddings_dict = torch.load(file)
        print('debug')
    except Exception as e:
        print(f"Error reading torch embedding: {e}")
        return None

if __name__ == '__main__':
    embedding_path = '/Users/mpekey/Desktop/ProteinBERT/ProteinBERTKinaseEmbAvg.pickle'
    convert_pickle_to_torch(embedding_path)