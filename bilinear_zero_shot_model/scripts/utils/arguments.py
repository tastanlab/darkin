import yaml
# from scripts.utils.training_utils import set_saved_checkpoint_filename

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_helper_arguments(config):
    '''
        Here will create configs of helpers which doesn't necessearily need to be in the config file.
        They are related to other configs so they can be created here with conditions.
    '''
    config['helpers'] = {}
    config['helpers']['phosphosite'] = {}
    config['helpers']['kinase'] = {}
    # 1. if model has token ids, if you want to scale, then scale in forward loop
    config['helpers']['phosphosite']['has_token_id'] = False
    config['helpers']['kinase']['has_token_id'] = False
    if config['phosphosite']['dataset']['processor']['processor_type'] in ['hf'] and not config['phosphosite']['dataset']['processor']['read_embeddings']:
        config['helpers']['phosphosite']['has_token_id'] = True

    if config['kinase']['dataset']['processor']['processor_type'] in ['hf'] and not config['kinase']['dataset']['processor']['read_embeddings']:
        config['helpers']['kinase']['has_token_id'] = True
    
    # 2. unfrozen layer list. A user can give the layer numbers as '4,5' so that they converted into [4,5]
    config['helpers']['phosphosite']['unfrozen_layers'] = [i for i in config['phosphosite']['model']['unfrozen_layers'].split(',') if i!='']
    config['helpers']['kinase']['unfrozen_layers'] = [i for i in config['kinase']['model']['unfrozen_layers'].split(',') if i!='']

    # 3. Decide saved checkpoint filename
    if config['logging']['local']['use_config_filename']:
        config = set_saved_checkpoint_filename(config)
    
    return config

def set_saved_checkpoint_filename(config):
    '''
        This function run if use_config_filename is set to True in the config file.
    '''
    print(f'Best params are : {config["hyper_parameters"]}')

    # sequence_model_name = f'{config["phosphosite"]["sequence_model"]["model_type"]}'
    sequence_model_name = f'{config["phosphosite"]["model"]["model_name"]}'
    checkpoint_file_name = f'{f"{sequence_model_name}" if config["phosphosite"]["sequence_model"]["use_sequence_model"] else "wo_sequence_model"}'
    checkpoint_file_name += f'_family_{"T" if config["kinase"]["dataset"]["processor"]["use_family"] else "F"}'
    checkpoint_file_name += f'_group_{"T" if config["kinase"]["dataset"]["processor"]["use_group"] else "F"}'
    checkpoint_file_name += f'_EC_{"T" if config["kinase"]["dataset"]["processor"]["use_enzymes"] else "F"}'
    checkpoint_file_name += f'_kinase_domain_{"T" if (config["kinase"]["dataset"]["processor"]["use_domain"] or config["kinase"]["dataset"]["processor"]["use_kin2vec"]) else "F"}'
    checkpoint_file_name += f'_keggPathway_{"T" if config["kinase"]["dataset"]["processor"]["use_pathway"] else "F"}'
    checkpoint_file_name += f'_kinsimembed_{"T" if config["kinase"]["dataset"]["processor"].get("use_kinase_similarity_vector", False) else "F"}' if "use_kinase_similarity_vector" in config["kinase"]["dataset"]["processor"] else ""
    checkpoint_file_name += f'{"_kinFineGrainedClust_T" if config["kinase"]["dataset"]["processor"].get("use_fine_grained_clustering_binary_vector", False) else ""}'
    checkpoint_file_name += f'{"_alginedUnlabaledAug_T" if config["training"].get("augment_aligned_unlabaled_sites", False) else ""}'
    checkpoint_file_name += f'_protvecActiveSite_{"T" if config["kinase"]["dataset"]["processor"]["active_site"]["use_active_site"] else "F"}'
    checkpoint_file_name += f'_pLMActiveSite_{"T" if config["kinase"]["dataset"]["processor"]["active_site"]["use_active_site"] else "F"}'
    checkpoint_file_name += f'_gamma_{config["hyper_parameters"]["gamma"]:.4f}'
    checkpoint_file_name += f'_lr_{config["hyper_parameters"]["learning_rate"]:.4f}'
    checkpoint_file_name += f'_{config["hyper_parameters"]["optimizer"]}'
    checkpoint_file_name += f'_{config["hyper_parameters"]["scheduler_type"]}'
    checkpoint_file_name += f'_weight_decay_{config["hyper_parameters"]["weight_decay"]:.4f}'
    checkpoint_file_name += f'_random_seed_{config["random_seed"]}'
    config["logging"]["local"]["checkpoint_file_name"] = checkpoint_file_name
    print(checkpoint_file_name)
    return config