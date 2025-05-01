import os

from scripts.models.huggingface_models import HFModel
from scripts.models.bilstm import *
from scripts.models.zero_dkz import Zero_DKZ
from scripts.models.lora import lora
from scripts.evaluation.losses import *

def generate_model(config, data_shapes):
    phosphosite_model = None
    kinase_model = None
    sequence_model = None

    # Create phosphosite model
    if not config['phosphosite']['dataset']['processor']['read_embeddings'] and config['phosphosite']['model']['model_type'] != "protvec":
        if config['phosphosite']['model']['model_type'] == 'hf':
            phosphosite_model = HFModel(
                model_name = config['phosphosite']['model']['model_name'],
                embedding_mode = config['phosphosite']['model']['embedding_mode'],
                is_pretrained=config['phosphosite']['model']['is_pretrained']
            )

            # Remove encoder layers from last. If 0, no layers will be removed.
            phosphosite_model.embedding_model = modify_layers(
                model=phosphosite_model.embedding_model,
                num_layers_to_remove=config['phosphosite']['model'].get('remove_layers', 0),
                encoder_name="encoder",
                layer_name="layer"
            )
            
        # Freeze weights
        if config['phosphosite']['model']['freeze']:
            phosphosite_model._freeze_embedding_model()
            if config['phosphosite']['model']['lora']:
                phosphosite_model = lora(phosphosite_model, config.get('lora_config', None))
            else:
                phosphosite_model._unfreeze_given_encoder_layers(config['helpers']['phosphosite']['unfrozen_layers'])

    # Create kinase model
    if not config['kinase']['dataset']['processor']['read_embeddings'] and config['kinase']['model']['model_type'] != "protvec":
        if config['kinase']['model']['model_type'] == 'hf':
            kinase_model = HFModel(
                model_name = config['kinase']['model']['model_name'],
                embedding_mode = config['kinase']['model']['embedding_mode'],
                is_pretrained=config['kinase']['model']['is_pretrained']
            )
        # Freeze weights
        if config['kinase']['model']['freeze']:
            kinase_model._freeze_embedding_model()
            if config['kinase']['model']['lora']:
                kinase_model = lora(kinase_model, config.get('lora_config', None))
            else:
                kinase_model._unfreeze_given_encoder_layers(config['helpers']['phosphosite']['unfrozen_layers'])

    # Create sequence model
    if config['phosphosite']['sequence_model']['use_sequence_model']:
        if config['phosphosite']['sequence_model']['model_type'] == 'bilstm':
            if config['phosphosite']['model']['model_type'] in ['esm', 'esm_meta', 'huggingface']:
                vocabnum = get_embedding_dim(config['phosphosite']['model']['model_name'])
                seq_lens = data_shapes['phosphosite'][1]
            else:
                vocabnum = data_shapes['phosphosite'][2]
                seq_lens = data_shapes['phosphosite'][1]
            sequence_model = BiLstm(vocabnum, seq_lens)

    # Calculate w shape
    w_dim = get_w_dim(config, data_shapes)
    model = Zero_DKZ(
        w_dim=w_dim,
        phosphosite_model=phosphosite_model,
        kinase_model=kinase_model,
        sequence_model=sequence_model,
        config=config
    )
    return model


def get_w_dim(config, data_shapes):
    # w_1 is the dimension of the phosphosite embeddings
    # If we use a second model to process embeddings, get its output dimension for w_1
    if config['phosphosite']['sequence_model']['use_sequence_model']:
        if config['phosphosite']['sequence_model']['model_type'] == 'bilstm':
            w_1 = config['phosphosite']['sequence_model']['hidden_size'] * 2
        else:
            w_1 = config['phosphosite']['sequence_model']['hidden_size']
    else:
        w_1 = data_shapes['phosphosite'][1]
        if not config['phosphosite']['dataset']['processor']['read_embeddings']:
            # Use embedding model to process phosphosite (They have token ids, so we need to get the embedding size)
            if config['phosphosite']['model']['model_type'] == 'hf':
                w_1 = get_embedding_dim(config['phosphosite']['model']['model_name'])
    
    # w_2 is the dimension of the kinase embeddings
    kinase_sequence_embedding_size = 0
    if config['kinase']['dataset']['processor']['use_domain']:
        if config['kinase']['model']['model_type'] == 'hf':
            kinase_sequence_embedding_size = get_embedding_dim(config['kinase']['model']['model_name'])
        else:
            kinase_sequence_embedding_size = data_shapes['kinase']['sequence'][1] if len(data_shapes['kinase']['sequence']) > 1 else 0

    kinase_property_embedding_size = data_shapes['kinase']['properties'][1] if len(data_shapes['kinase']['properties']) > 1 else 0
    w_2 = kinase_sequence_embedding_size + kinase_property_embedding_size
    print(f'Shape of W: ({w_1 + 1}, {w_2 + 1})')
    return (w_1 + 1, w_2 + 1)


def get_embedding_dim(model_name):
    esm_info = {
        'esm3':1536,
        'esm2_t48_15B_UR50D':5120,
        'esm2_t36_3B_UR50D':2560,
        'esm2_t33_650M_UR50D':1280,
        'esm2_t30_150M_UR50D':640,
        'esm2_t12_35M_UR50D':480,
        'esm2_t6_8M_UR50D':320,
        'esm_if1_gvp4_t16_142M_UR50':512,
        'esm1v_t33_650M_UR90S_[1-5]':1280,
        'esm_msa1b_t12_100M_UR50S':768,
        'esm1b_t33_650M_UR50S':1280,
        'esm1_t34_670M_UR50S':1280,
        'esm1_t34_670M_UR50D':1280,
        'esm1_t34_670M_UR100':1280,
        'esm1_t12_85M_UR50S':768,
        'esm1_t6_43M_UR50S':768,
        'saprot':1280,
        'prott5xl':1024,
        'protbert': 1024,
        'distilprotbert' : 1024,
        'protalbert' : 4096,
        'protgpt2' : 1280
    }
    if model_name.startswith('esm1v_t33_650M_UR90S_'):
        model_name = 'esm1v_t33_650M_UR90S_[1-5]'
    return esm_info[model_name]


def modify_layers(model, num_layers_to_remove, encoder_name="encoder", layer_name="layer"):
    '''
        Removes the last `num_layers_to_remove` layers from the model encoder.
    '''

    if num_layers_to_remove == 0:
        return model

    # Get the encoder
    encoder = getattr(model, encoder_name, None)
    if encoder is None:
        raise AttributeError(f"Model does not have an attribute named '{encoder_name}'")
    
    # Ensure the encoder has the specified layer attribute
    layer = getattr(encoder, layer_name, None)
    if layer is None:
        raise AttributeError(f"Encoder does not have an attribute named '{layer_name}'")
    
    # Ensure the number of layers to remove is valid
    if num_layers_to_remove < 0:
        print('No layers will be removed due to negative value. Returning model.')
        return model
    elif num_layers_to_remove > len(model.encoder.layer):
        print(f"Number of layers to remove exceeds the number of layers in the model. Returning model.")
        return model

    # Remove the last `num_layers_to_remove` layers
    setattr(encoder, layer_name, layer[:-num_layers_to_remove])
    print(f"Removed {num_layers_to_remove} layers from the model encoder.")
    return model


def save_model(config, model_state_dict, optim_state_dict):
    save_filepath = os.path.join(
        config['logging']['local']['saved_model_path'],
        f"{config['logging']['local']['checkpoint_file_name']}",
        f"{config['run_model_id']}.pt"
    )

    state = {
        'state_dict': model_state_dict,
        'optimizer': optim_state_dict
    }
    try:
        directory = '/'.join(save_filepath.split('/')[:-1])
        os.makedirs(directory, exist_ok=True)
        torch.save(state, save_filepath)
        print(f"Model {config['run_model_id']} saved successfully!")
    except Exception as e:
        print(f"Failed to save model {config['run_model_id']}. Error: {e}")


def load_model(config, data_shapes):
    model = generate_model(config, data_shapes)
    # Load the saved state
    load_filepath = os.path.join(
        config['logging']['local']['saved_model_path'],
        f"{config['logging']['local']['checkpoint_file_name']}",
        f"{config['run_model_id']}.pt"
    )
    saved_state = torch.load(load_filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(saved_state['state_dict'])
    return model


def get_optimizer(config, model):
    optimizer = config['hyper_parameters']['optimizer']
    learning_rate = config['hyper_parameters']['learning_rate']
    weight_decay = config['hyper_parameters']['weight_decay']
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def get_scheduler(config, optimizer):
    scheduler = config['hyper_parameters']['scheduler_type']
    gamma = config['hyper_parameters']['gamma']
    if scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)
    elif scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    elif scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    return scheduler


def get_loss_function(config, mode='train'):
    loss_f = config['training'].get('loss_function', 'normalized_cross_entropy')
    
    # Can be commented out if you want to use binary cross entropy for training
    # if mode == 'train' and loss_f == 'binary_cross_entropy':
        # loss_f = 'normalized_cross_entropy'

    if loss_f == 'cross_entropy':
        return cross_entropy
    elif loss_f == 'normalized_cross_entropy':
        return cross_entropy_with_normalization
    elif loss_f == 'focal_loss':
        return focal_loss
    elif loss_f == 'binary_cross_entropy':
        return multilabel_binary_cross_entropy_loss
    elif loss_f == 'cross_entropy_softmax_scaling':
        return cross_entropy_with_softmax_scaling
    elif loss_f == 'feature_cross_entropy_loss':
        return feature_cross_entropy_loss
    else:
        raise ValueError(f"Loss function {loss_f} not found!")


def set_saved_checkpoint_filename(config):
    '''
        This function run if use_config_filename is set to True in the config file.
    '''
    print(f'Best params are : {config["hyper_parameters"]}')

    sequence_model_name = f'{config["phosphosite"]["sequence_model"]["model_type"]}'
    checkpoint_file_name = f'{f"{sequence_model_name}" if config["phosphosite"]["sequence_model"]["use_sequence_model"] else "wo_sequence_model"}'
    checkpoint_file_name += f'_family_{"T" if config["kinase"]["dataset"]["processor"]["use_family"] else "F"}'
    checkpoint_file_name += f'_group_{"T" if config["kinase"]["dataset"]["processor"]["use_group"] else "F"}'
    checkpoint_file_name += f'_EC_{"T" if config["kinase"]["dataset"]["processor"]["use_enzymes"] else "F"}'
    checkpoint_file_name += f'_kinase_domain_{"T" if (config["kinase"]["dataset"]["processor"]["use_domain"] or config["kinase"]["dataset"]["processor"]["use_kin2vec"]) else "F"}'
    checkpoint_file_name += f'_keggPathway_{"T" if config["kinase"]["dataset"]["processor"]["use_pathway"] else "F"}'
    checkpoint_file_name += f'_protvecActiveSite_{"T" if config["kinase"]["dataset"]["processor"]["active_site"]["use_active_site"] else "F"}'
    checkpoint_file_name += f'_pLMActiveSite_{"T" if config["kinase"]["dataset"]["processor"]["active_site"]["use_active_site"] else "F"}'
    checkpoint_file_name += f'_gamma_{config["hyper_parameters"]["gamma"]}'
    checkpoint_file_name += f'_lr_{str(config["hyper_parameters"]["learning_rate"])}'
    checkpoint_file_name += f'_{config["hyper_parameters"]["optimizer"]}'
    checkpoint_file_name += f'_{config["hyper_parameters"]["scheduler_type"]}'
    checkpoint_file_name += f'_weight_decay_{config["hyper_parameters"]["weight_decay"]}'
    #checkpoint_file_name += f'_random_seed_{config["random_seed"]}'
    config["logging"]["local"]["checkpoint_file_name"] = checkpoint_file_name
    print(checkpoint_file_name)
    return config