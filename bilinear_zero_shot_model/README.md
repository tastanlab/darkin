# DeepKinZero

DKZ explanation or abstract

## Getting Started

### Installation

You can install the required libraries using the following pip commands:

```bash
pip install -r requirements.txt
```

### Model Training

To train model or test an already trained one, follow these steps:

1. Clone the project repository:

    ```bash
    git clone https://github.com/mertpekey/DeepKinZero.git
    ```

2. Prepare a config file (or edit one of the example configs)

3. Install and put the model embeddings in a folder in .pt format. Embeddings format should be as follows:

    {'_____MsssEEVsWI' : torch.tensor, ...}

    Data size may be (seq_len, embedding_dim) or (embedding_dim). If 2D embeddings used, then you should decide whether you want to get average of the sequence or cls token embedding by changing embedding_mode (cls or avg) in the config file for both kinase and phosphosite. For 1D embeddings, 'sequence' should be used for embedding_mode.

4. Run the training/testing script with the following command:

    ```bash
    python main.py --mode train --config_path configs/protgpt2_model.yaml --num_of_models 1
    ```

   - `--mode`: train or test
   - `--config_path`: config file path
   - `--num_of_models`: how many independent models you want to train