import torch
import argparse
import pandas as pd

from scripts.models.esm_models import ESM
from scripts.utils.arguments import load_config
from scripts.utils.data_utils import load_kinase_data
from scripts.data.data_processors.esm_processor import ESMProcessor

def create_kinase_embeddings(model_name):
    kinase_filename = 'dataset/new_dataset/zsl/seed_12345/kinase_properties.csv'
    kinase_data_dict = load_kinase_data(kinase_filename)
    kinase_sequences = []
    for seq in kinase_data_dict.values():
        kinase_sequences.append(seq['domain'])

    model = ESM(
        model_name = model_name,
        embedding_mode = 'sequence'
    )
    model._freeze_embedding_model()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    esm_processor = ESMProcessor({'model_name' : model_name})
    batch_tokens = esm_processor.process_kinase_sequence(kinase_sequences)
    batch_tokens = batch_tokens.to(device)
        
    embeddings_data = {}
    for i in range(len(kinase_sequences)):
      if i % 100 == 0:
        print(f'Step: {i}')
      with torch.no_grad():
        embedding = model(batch_tokens[i].unsqueeze(0))
      embeddings_data[kinase_sequences[i]] = embedding.squeeze().cpu()

    # Save Embeddings
    save_filename = f'embeddings/kinase_{model_name}_emb.pth'
    torch.save(embeddings_data, save_filename)
    print(f"Embeddings saved to {save_filename}")


def create_phosphosite_embeddings(model_name):
    phosphosite_filename_1 = 'dataset/new_dataset/zsl/seed_12345/train_data.csv'
    phosphosite_filename_2 = 'dataset/new_dataset/zsl/seed_12345/validation_data.csv'
    phosphosite_filename_3 = 'dataset/new_dataset/zsl/seed_12345/test_data.csv'

    phosphosite_sequences_1 = pd.read_csv(phosphosite_filename_1)["SITE_+/-7_AA"].tolist()
    phosphosite_sequences_2 = pd.read_csv(phosphosite_filename_2)["SITE_+/-7_AA"].tolist()
    phosphosite_sequences_3 = pd.read_csv(phosphosite_filename_3)["SITE_+/-7_AA"].tolist()
    phosphosite_sequences = list(set(phosphosite_sequences_1 + phosphosite_sequences_2 + phosphosite_sequences_3))
    phosphosite_sequences = [seq for seq in phosphosite_sequences if isinstance(seq, str)]

    model = ESM(
        model_name = model_name,
        embedding_mode = 'sequence'
    )
    model._freeze_embedding_model()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    esm_processor = ESMProcessor({'model_name' : model_name})
    batch_tokens = esm_processor.process_phosphosite_sequence(phosphosite_sequences)
    batch_tokens = batch_tokens.to(device)
    
    embeddings_data = {}
    for i in range(len(phosphosite_sequences)):
      if i % 1000 == 0:
        print(f'Step: {i}')
      with torch.no_grad():
        embedding = model(batch_tokens[i].unsqueeze(0))
      embeddings_data[phosphosite_sequences[i]] = embedding.squeeze().cpu()

    # Save Embeddings
    save_filename = f'embeddings/phosphosite_{model_name}_emb.pth'
    torch.save(embeddings_data, save_filename)
    print(f"Embeddings saved to {save_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Embeddings')
    parser.add_argument('--model_name', default='esm2_t6_8M_UR50D', help='Model Name')
    parser.add_argument('--sequence_type', default='kinase', help='Embedding data type')

    args = parser.parse_args()
    config = load_config(args.config_path)

    if args.sequence_type == 'kinase':
        create_kinase_embeddings(args.model_name)
    else:
        create_phosphosite_embeddings(args.model_name)