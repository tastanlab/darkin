# BZSM Method

**BZSM** (Baseline Zero-shot Method) utilizes pre-calculated model embeddings, training only a compatibility matrix `W`. This results in fewer trainable parameters compared to other methods, and we use these results as baselines for comparison.

Below are configuration examples for different model and data types.

---

## Configurations

### Huggingface Models

**Key configurations to edit:**

#### Phosphosite

```yaml
phosphosite:
    dataset:
    processor:
        # Directly read the pre-calculated embedding files
        read_embeddings: true
        phosphosite_embedding_path: /truba/home/mpekey/dkz_models/esm1b_phosphosite.pt

    model:
    # Specify the Huggingface model
    model_name: esm1b_t33_650M_UR50S  # Valid model names are listed below
    embedding_mode: cls  # Options: 'cls' token or 'avg'
```

#### Kinase

```yaml
kinase:
    dataset:
    processor:
        # Directly read the pre-calculated embedding files
        read_embeddings: true
        kinase_embedding_path: /truba/home/mpekey/dkz_models/esm1b_kinase.pt
        
        # Specify which kinase embeddings will be used
        use_family: true
        use_group: true
        use_enzymes: true
        use_domain: true  # 'kinase_domain' embeddings read from the file

    model:
    # Specify the Huggingface model
    model_name: esm1b_t33_650M_UR50S
    embedding_mode: cls  # Options: 'cls' token or 'avg'
```

#### Training

```yaml
training:
  # Normalize phosphosite data for better performance with BZSM
  normalize_phosphosite_data: true
```

#### Valid Models

Here is the list of valid Huggingface models currently supported by the method:

- esm2_t33_650M_UR50D
- esm2_t30_150M_UR50D
- esm2_t12_35M_UR50D
- esm2_t6_8M_UR50D
- esm1v_t33_650M_UR90S_[1-5]
- esm1b_t33_650M_UR50S
- saprot
- prott5xl
- protbert
- distilprotbert
- protalbert
- protgpt2

---

### ProtVec Configurations

#### Phosphosite

```yaml
phosphosite:
    dataset:
    processor:
        processor_type: protvec  # Use ProtVec embedding processor
        model_name: protvec
        read_embeddings: false  # ProtVec embeddings will be read from a file
        protvec_file_path: dataset/new_dataset/protvec_embeddings.txt

    model:
    model_type: protvec
    model_name: protvec
    embedding_mode: sequence  # Use pre-processed ProtVec sequence embeddings
```

#### Kinase Dataset

```yaml
kinase:
    dataset:
    processor:
        processor_type: protvec  # Use ProtVec embedding processor
        model_name: protvec
        read_embeddings: false  # ProtVec embeddings will be read from a file
        protvec_file_path: dataset/new_dataset/protvec_embeddings.txt

        # Specify which kinase embeddings will be used
        use_family: true
        use_group: true
        use_enzymes: true
        use_kin2vec: true  # 'kin2vec' refers to ProtVec embeddings
        use_domain: false  # Domain embeddings are not used

    model:
    model_type: protvec
    model_name: protvec
    embedding_mode: sequence  # Use pre-processed ProtVec sequence embeddings
```

#### Training

```yaml
training:
  # Normalize phosphosite data for better performance with BZSM
  normalize_phosphosite_data: true
```
