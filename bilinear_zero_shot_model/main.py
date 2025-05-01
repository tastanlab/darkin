import wandb
import torch
import numpy as np
import random
import argparse

from scripts.run.train import train
from scripts.run.test import test
from scripts.run.train_only import train_val
from scripts.run.predict import predict
from scripts.utils.arguments import load_config, create_helper_arguments
from scripts.utils.evaluation_utils import ensemble_results, average_df_by_id, ensemble_probabilities_aupr
from scripts.evaluation.test_suite import run as run_test_suite

def main():
    parser = argparse.ArgumentParser(description='DeepKinZero_V2')
    #parser.add_argument('--mode', choices=['train', 'test', 'predict'], required=True, help='Mode: train, test, or predict')

    parser.add_argument('--mode', choices=['train', 'test', 'predict', 'train_val'], default='test', help='Mode: train, test, or predict', type=str)
    parser.add_argument('--config_path', default='configs/esm1b_bzsm_config.yaml', help='Config yaml file of model', type=str)
    parser.add_argument('--num_of_models', default=1, help='Number of independent models to train', type=int)

    args = parser.parse_args()
    config = load_config(args.config_path)
    config = create_helper_arguments(config)

    if config['logging']['wandb']['use_wandb']:
        wandb.init(
            project=config['logging']['wandb']['project_name'], 
            name=config['logging']['wandb']['log_name'],
            entity=config['logging']['wandb']['entity_name'],
            config=config
        )
    
    if args.mode == 'train':
        train_results = []
        val_results = []
        all_train_losses, all_val_losses = [], []
        for model_id in range(args.num_of_models):
            if config['training']['set_seed']:
                torch.manual_seed(model_id)
                torch.cuda.manual_seed_all(model_id)
                np.random.seed(model_id)
                random.seed(model_id)
            config['run_model_id'] = model_id
            result = train(config)
            train_results.append(result['train'])
            val_results.append(result['val'])
            all_train_losses.append(result['all_train_loss'])
            all_val_losses.append(result['all_val_loss'])
        
        # Important note!!! : ensemble_probabilities_aupr will only work for the validation results
        # because validation dataloader is not shuffled but the train dataloader is shuffled
        # So for the train results the models will not bre making predictions for the same samples in the
        # same order. So they cannot be ensembled! 
        ensemble_validation_aupr, ensemble_validation_aupr_with_logits = ensemble_probabilities_aupr(val_results)

        mean_train_results, std_dev_train = ensemble_results(train_results)
        mean_val_results, std_dev_val = ensemble_results(val_results)    
        
        if config['logging']['wandb']['use_wandb']:
            wandb.log({f'ensemble_train/{metric}' : value for metric, value in mean_train_results.items()})
            wandb.log({f'ensemble_validation/{metric}' : value for metric, value in mean_val_results.items()})
            wandb.log({f'std_dev_train/{metric}' : value for metric, value in std_dev_train.items()})
            wandb.log({f'std_dev_validation/{metric}' : value for metric, value in std_dev_val.items()})
            wandb.log({f'ensemble_validation_aupr' : ensemble_validation_aupr})
            wandb.log({f'ensemble_validation_aupr_with_logits' : ensemble_validation_aupr_with_logits})

        print(f'Ensembled Train Results:\n{mean_train_results}')
        print(f'Ensembled Validation Results:\n{mean_val_results}')
        print(f'std-dev Train Results:\n{std_dev_train}')
        print(f'std-dev Validation Results:\n{std_dev_val}')
        print(f'Ensembled Probabilities Validation Macro AUPR Results :\n{ensemble_validation_aupr}')
        print(f'Ensembled Probabilities Validation Macro AUPR Results with logits:\n{ensemble_validation_aupr_with_logits}')

        if config['logging']['local'].get('run_test_suite', False):
            config["mode"] = "train"
            config["kinase_similarity_file"] = "dataset/new_dataset/kinase_pairwise_identity_similarity_scores.csv"
            run_test_suite(config, val_results, all_train_losses, all_val_losses)
    
    elif args.mode == 'test':
        test_results_list = []
        class_based_dfs, group_based_dfs, family_based_dfs = [], [], []
        for model_id in range(args.num_of_models):
            if config['training']['set_seed']:
                torch.manual_seed(model_id)
                torch.cuda.manual_seed_all(model_id)
                np.random.seed(model_id)
                random.seed(model_id)
            config['run_model_id'] = model_id
            result, metrics = test(config)
            test_results_list.append(result)
            class_based_dfs.append(metrics.class_aupr_scores)
            group_based_dfs.append(metrics.aupr_per_group)
            family_based_dfs.append(metrics.aupr_per_family)
        
        # ensemble_test_aupr, ensemble_test_aupr_logits = ensemble_probabilities_aupr(test_results_list)
        test_results, std_dev_test = ensemble_results(test_results_list)

        # Save to CSV
        class_df = average_df_by_id(class_based_dfs)
        group_df = average_df_by_id(group_based_dfs)
        family_df = average_df_by_id(family_based_dfs)
        
        class_df.to_csv(f'supplementary/{config["phosphosite"]["model"]["model_name"]}_class_based_aupr.csv', index=False)
        group_df.to_csv(f'supplementary/{config["phosphosite"]["model"]["model_name"]}_group_based_aupr.csv', index=False)
        family_df.to_csv(f'supplementary/{config["phosphosite"]["model"]["model_name"]}_family_based_aupr.csv', index=False)

        if config['logging']['wandb']['use_wandb']:
            wandb.log({f'ensemble_test/{metric}' : value for metric, value in test_results.items()})
            wandb.log({f'std_dev_test/{metric}' : value for metric, value in std_dev_test.items()})
            class_table = wandb.Table(dataframe=class_df)
            group_table = wandb.Table(dataframe=group_df)
            family_table = wandb.Table(dataframe=family_df)
            wandb.log({"ensemble_test/class_based_aupr_table" : class_table, 'ensemble_test/group_based_aupr_table' : group_table, 'ensemble_test/family_based_aupr_table' : family_table})
            # wandb.log({f'ensemble_test_aupr' : ensemble_test_aupr})
            # wandb.log({f'ensemble_test_aupr_with_logits' : ensemble_test_aupr_logits})

        print(f'Ensembled Test Results:\n{test_results}')
        print(f'std-dev Test Results:\n{std_dev_test}')
        # print(f'Ensembled Probabilities Test Macro AUPR Results :\n{ensemble_test_aupr}')
        # print(f'Ensembled Probabilities Test Macro AUPR Results with logits:\n{ensemble_test_aupr_logits}')

        if config['logging']['local'].get('run_test_suite', False):
            config["mode"] = "test"
            config["kinase_similarity_file"] = "dataset/new_dataset/kinase_pairwise_identity_similarity_scores.csv"
            run_test_suite(config, test_results_list, None, None)

    elif args.mode == 'train_val':
        train_results = []
        all_train_losses = []
        for model_id in range(args.num_of_models):
            if config['training']['set_seed']:
                torch.manual_seed(model_id)
                torch.cuda.manual_seed_all(model_id)
                np.random.seed(model_id)
                random.seed(model_id)
            config['run_model_id'] = model_id
            result = train_val(config)
            train_results.append(result['train'])
            all_train_losses.append(result['all_train_loss'])

        mean_train_results, std_dev_train = ensemble_results(train_results) 
        
        if config['logging']['wandb']['use_wandb']:
            wandb.log({f'ensemble_train/{metric}' : value for metric, value in mean_train_results.items()})
            wandb.log({f'std_dev_train/{metric}' : value for metric, value in std_dev_train.items()})

        print(f'Ensembled Train Results:\n{mean_train_results}')
        print(f'std-dev Train Results:\n{std_dev_train}')

    elif args.mode == 'predict':
        results = predict(config)
        print(results.predictions[:5])
        print(results.probabilities[:5])
        print(results.labels[:5])

if __name__ == "__main__":
    main()
