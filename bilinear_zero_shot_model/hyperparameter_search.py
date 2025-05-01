import wandb
import torch
import numpy as np
import random
import argparse

from scripts.run.train import train
from scripts.utils.arguments import load_config, create_helper_arguments

def main():
    config = load_config(args.config_path)
    config = create_helper_arguments(config)

    wandb.init()

    ### Update Config with Wandb Sweep Parameters
    #############################################
    # How to use?
    # Replace the config values with wandb.config.<parameter_name>
    # You only need to replace the values that you defined in the sweep_config.yaml file
    #############################################
    config['training']['train_batch_size'] = wandb.config.batch_size
    config['training']['test_batch_size'] = wandb.config.batch_size
    config['training']['loss_function'] = wandb.config.loss_function
    config['hyper_parameters']['learning_rate'] = wandb.config.learning_rate
    config['hyper_parameters']['gamma'] = wandb.config.gamma
    config['hyper_parameters']['optimizer'] = wandb.config.optimizer
    config['hyper_parameters']['scheduler_type'] = wandb.config.scheduler_type
    config['hyper_parameters']['weight_decay'] = wandb.config.weight_decay
    config['hyper_parameters']['temperature'] = wandb.config.temperature
    config['hyper_parameters']['focal_loss_gamma'] = wandb.config.focal_loss_gamma
    config['hyper_parameters']['use_weighted_loss'] = wandb.config.use_weighted_loss

    # if wandb.config.random_seed not in [0, 42, 87, 112, 12345]:
    #     config['phosphosite']['dataset']['train'] = config['phosphosite']['dataset']['train'].replace('seed_12345', f'random_seed_{str(wandb.config.random_seed)}')
    #     config['phosphosite']['dataset']['validation'] = config['phosphosite']['dataset']['validation'].replace('seed_12345', f'random_seed_{str(wandb.config.random_seed)}')
    #     config['phosphosite']['dataset']['test'] = config['phosphosite']['dataset']['test'].replace('seed_12345', f'random_seed_{str(wandb.config.random_seed)}')

    #     config['kinase']['dataset']['train'] = config['kinase']['dataset']['train'].replace('seed_12345', f'random_seed_{str(wandb.config.random_seed)}')
    #     config['kinase']['dataset']['validation'] = config['kinase']['dataset']['validation'].replace('seed_12345', f'random_seed_{str(wandb.config.random_seed)}')
    #     config['kinase']['dataset']['test'] = config['kinase']['dataset']['test'].replace('seed_12345', f'random_seed_{str(wandb.config.random_seed)}')
    # else:
    #     config['phosphosite']['dataset']['train'] = config['phosphosite']['dataset']['train'].replace('seed_12345', f'seed_{str(wandb.config.random_seed)}')
    #     config['phosphosite']['dataset']['validation'] = config['phosphosite']['dataset']['validation'].replace('seed_12345', f'seed_{str(wandb.config.random_seed)}')
    #     config['phosphosite']['dataset']['test'] = config['phosphosite']['dataset']['test'].replace('seed_12345', f'seed_{str(wandb.config.random_seed)}')

    #     config['kinase']['dataset']['train'] = config['kinase']['dataset']['train'].replace('seed_12345', f'seed_{str(wandb.config.random_seed)}')
    #     config['kinase']['dataset']['validation'] = config['kinase']['dataset']['validation'].replace('seed_12345', f'seed_{str(wandb.config.random_seed)}')
    #     config['kinase']['dataset']['test'] = config['kinase']['dataset']['test'].replace('seed_12345', f'seed_{str(wandb.config.random_seed)}')
    
    model_id = 0
    if config['training']['set_seed']:
        torch.manual_seed(model_id)
        torch.cuda.manual_seed_all(model_id)
        np.random.seed(model_id)
        random.seed(model_id)
    config['run_model_id'] = model_id
    _ = train(config)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='DeepKinZero Sweep Hyperparameter Search')
    parser.add_argument('--config_path', default='configs/example_config.yaml', help='Config yaml file of model', type=str)
    parser.add_argument('--sweep_config_path', default='configs/sweep_config.yaml', help='Sweep Config yaml file path', type=str)
    parser.add_argument('--sweep_project_name', default='dkz_sweep', help='Sweep Project Name in Wandb', type=str)
    parser.add_argument('--sweep_count', default=1, help='Number of Different Hyperparameter Runs', type=int)
    
    global args
    args = parser.parse_args()

    sweep_configuration = load_config(args.sweep_config_path)['sweep']

    if sweep_configuration['method'] == 'grid':
        sweep_count = None
    else:
        sweep_count = args.sweep_count

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.sweep_project_name)
    wandb.agent(sweep_id, function=main, count=sweep_count)
