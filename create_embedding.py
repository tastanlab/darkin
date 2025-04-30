import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(description="Compute protein sequence embeddings.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_files', nargs='+', help="Paths to input CSV files containing sequences.")
    group.add_argument('--sequence', help="Single protein sequence.")
    parser.add_argument('--column_name', default='sequence', help="Column name for sequences in CSV files.")
    parser.add_argument('--model_id', required=True, help="Hugging Face model ID.")
    parser.add_argument('--embedding_type', choices=['cls','avg','all'], default='all', help="Type of embedding: cls, avg, or all.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for embedding extraction.")
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto', help="Device to use: auto, cpu, or cuda.")
    parser.add_argument('--max_length', type=int, default=None, help="Maximum sequence length (truncate).")
    parser.add_argument('--output_file', default='embeddings.pt', help="Output .pt file name.")
    return parser.parse_args()


def get_device(device_choice):
    """
    Determine the torch device based on user choice.

    Args:
        device_choice (str): 'auto', 'cpu', or 'cuda'.

    Returns:
        torch.device: Selected device ('cuda' if available when 'auto').
    """
    if device_choice == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_choice)


def load_sequences(files, single_sequence, column_name):
    """
    Load protein sequences from CSV(s) or a single string.

    - Reads one or more CSV files and extracts the specified column.
    - Replaces any '_' padding with '-' in all sequences.
    - Deduplicates while preserving the original order.

    Args:
        files (list[str] or None): Paths to CSV files. If None, uses single_sequence.
        single_sequence (str or None): Single sequence string.
        column_name (str): Column name in CSV files containing sequences.

    Returns:
        list[str]: Deduplicated, normalized sequences.
    """
    seqs = []
    if files:
        for f in files:
            df = pd.read_csv(f)
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in {f}")
            
            cleaned = (
                df[column_name]
                .astype(str)
                .str.replace('_', '-', regex=False)
                .tolist()
            )
            seqs.extend(cleaned)
    else:
        seqs.append(single_sequence.replace('_', '-'))
    # deduplicate
    return list(dict.fromkeys(seqs))


def compute_embeddings(sequences, tokenizer, model, device, emb_type, batch_size, max_length):
    """
    Compute embeddings for a list of sequences using a Hugging Face model.

    Args:
        sequences (list[str]): Protein sequences to embed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        model (transformers.PreTrainedModel): Sequence embedding model.
        device (torch.device): Device to run the model on.
        emb_type (str): 'cls', 'avg', or 'all'.
        batch_size (int): Number of sequences per batch.
        max_length (int or None): Maximum token length (for truncation).

    Returns:
        dict[str, torch.Tensor]: Mapping from sequence string to its embedding tensor.
            - If emb_type == 'cls', shape = (hidden_size,)
            - If emb_type == 'avg', shape = (hidden_size,)
            - If emb_type == 'all', shape = (seq_len, hidden_size)
    """
    embeddings = {}
    model.to(device)
    model.eval()
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        enc = tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True,
                        max_length=max_length)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        for j, seq in enumerate(batch_seqs):
            if emb_type == 'cls':
                emb = hidden[j, 0]
            elif emb_type == 'avg':
                mask = attention_mask[j].unsqueeze(-1)
                sum_emb = (hidden[j] * mask).sum(dim=0)
                emb = sum_emb / mask.sum()
            else:  # all
                emb = hidden[j]
            embeddings[seq] = emb.cpu()
    return embeddings


def main():
    args = parse_args()
    device = get_device(args.device)
    sequences = load_sequences(args.input_files, args.sequence, args.column_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModel.from_pretrained(args.model_id)
    embs = compute_embeddings(
        sequences,
        tokenizer,
        model,
        device,
        args.embedding_type,
        args.batch_size,
        args.max_length
    )
    torch.save(embs, args.output_file)
    print(f"Saved embeddings for {len(embs)} sequences to {args.output_file}")


if __name__ == "__main__":
    main()