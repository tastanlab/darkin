import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import copy
import os
import statistics

def ensemble_results(results):
    aggregated_results = {
        'epoch_loss': [],
        'epoch_acc': [],
        'epoch_roc_auc': [],
        'epoch_aupr': [],
        'epoch_aupr_with_logits': [],
        'epoch_phosphosite_ap': [],
        'epoch_phosphosite_ap_with_logits': [],
        'epoch_top3_acc': [],
        'epoch_top5_acc': [],
        'epoch_family_macro_ap': [],
        'epoch_group_macro_ap': [],
        'epoch_fine_grained_cluster_macro_ap': [],
        'epoch_family_top1_acc': [],
        'epoch_group_top1_acc': [],
        'epoch_fine_grained_cluster_top1_acc': [],
        'epoch_family_top3_acc': [],
        'epoch_group_top3_acc': [],
        'epoch_fine_grained_cluster_top3_acc': [],
        'epoch_family_top5_acc': [],
        'epoch_group_top5_acc': [],
        'epoch_fine_grained_cluster_top5_acc': [],
        'epoch_group_masked_ap': []
    }
    for result in results:
        for key in aggregated_results:
            aggregated_results[key].append(result[key])
    mean_results = {key: sum(aggregated_results[key]) / len(aggregated_results[key]) for key in aggregated_results}
    std_dev_results = {key: statistics.stdev(aggregated_results[key]) for key in aggregated_results if len(aggregated_results[key]) > 1}
    return mean_results, std_dev_results

def ensemble_probabilities_aupr(results):
    # IMPORTANT NOTE! : This wouldn't work for the train results because the train dataset is shuffled!!!
    # So the model's predictions will not be for the same samples in the same prder, thus they cannot
    # be ensembled.
    all_probabilities = []
    all_logits = []
    for result in results:
        metric = result["metrics"]
        all_probabilities.append(metric.get_probabilities())
        all_logits.append(metric.get_unique_logits())
    # The selected metric is not important at this point since we only need the label_mapping etc, which is the same for all  
    ensemble_probabilities = get_ensemble_probabilities(all_probabilities)
    ensemble_logits = get_ensemble_probabilities(all_logits)
    dummy_metric = copy.deepcopy(results[0]["metrics"])
    dummy_metric.set_probabilities(ensemble_probabilities)
    dummy_metric.set_unique_logits(ensemble_logits)
    ensembled_macro_aupr, _ = dummy_metric.calculate_aupr()
    ensembled_macro_aupr_with_logits, _ = dummy_metric.calculate_aupr(use_logits=True)
    return ensembled_macro_aupr, ensembled_macro_aupr_with_logits

def get_ensemble_probabilities(probabilities):
    ensemble_probabilities = torch.stack(probabilities).mean(dim=0)
    return ensemble_probabilities

def average_df_by_id(dict_results):
    dataframes = [pd.DataFrame(list(dict_res.items()), columns=['Key', 'Value']) for dict_res in dict_results]
    if len(dataframes) < 2:
        return dataframes[0]
    col_names = list(dataframes[0].columns)
    combined_df = pd.concat(dataframes, ignore_index=True)
    result_df = combined_df.groupby(col_names[0])[col_names[1]].mean().reset_index()
    result_df = result_df.sort_values(by='Value', ascending=False)
    return result_df


def plot_avg_residue_att_weights(phosphosite_model, test_dataset, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward_to_model(sequence):
        sequence = sequence.to(device)
        with torch.no_grad():
            attentions = phosphosite_model(sequence.unsqueeze(0), output_attentions=True)[-1]
        return attentions
    
    def get_avg_residue_attention_weights(attentions, tokens, remove_special_tokens=False):
        # Use final layer's attentions
        final_layer_att = attentions[-1]
        avg_attentions = torch.mean(final_layer_att, dim=1).squeeze(0).detach().cpu().numpy()

        key_token_idx = range(len(tokens))
        if remove_special_tokens:
            # Remove CLS (0) and EOS (2) token attentions
            key_token_idx = [i for i, token in enumerate(tokens) if token not in [0, 2]]
        
        avg_residue_att_weight = np.mean(avg_attentions, axis=0)
        avg_residue_att_weight = avg_residue_att_weight[key_token_idx]

        if remove_special_tokens:
            # Normalize to sum to 1
            avg_residue_att_weight /= np.sum(avg_residue_att_weight)

        return avg_residue_att_weight
    
    print('Calculating average residue attention weights...')
    # Get average residue attention weights for all sequences
    all_seq_residue_att_weights = []
    all_seq_residue_att_weights_specials = []
    for phs_seq in test_dataset.phosphosite_data:
        attentions = forward_to_model(phs_seq)
        avg_residue_att_weight = get_avg_residue_attention_weights(attentions, phs_seq, remove_special_tokens=True)
        avg_residue_att_weight_specials = get_avg_residue_attention_weights(attentions, phs_seq, remove_special_tokens=False)
        
        all_seq_residue_att_weights.append(avg_residue_att_weight)
        all_seq_residue_att_weights_specials.append(avg_residue_att_weight_specials)
    
    # Average across all sequences
    all_seq_residue_att_weights = np.mean(all_seq_residue_att_weights, axis=0)
    all_seq_residue_att_weights_specials = np.mean(all_seq_residue_att_weights_specials, axis=0)

    save_folder = f"supplementary/attention_plots/{config['logging']['local']['checkpoint_file_name']}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Plotting w/o special tokens
    positions = np.arange(-7, 8)
    plt.figure()
    plt.bar(positions, all_seq_residue_att_weights, color=(0.55, 0.55, 0.55))
    
    plt.title('Average Attention Weights of the Residue Positions', fontsize=10)
    plt.xlabel('Residue Positions', fontsize=10)
    plt.ylabel('Attention Weights', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(positions)

    plt.tight_layout()
    plt.savefig(f'{save_folder}/avg_residue_att_weights.png')

    # Plotting w/ special tokens
    positions = np.arange(-8, 9)
    plt.figure()
    plt.bar(positions, all_seq_residue_att_weights_specials, color=(0.55, 0.55, 0.55))
    
    plt.title('Average Attention Weights of the Residue Positions (w/ Special Tokens)', fontsize=10)
    plt.xlabel('Residue Positions', fontsize=10)
    plt.ylabel('Attention Weights', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(positions)

    plt.tight_layout()
    plt.savefig(f'{save_folder}/avg_residue_att_weights_w_specials.png')

    if config['logging']['wandb']['use_wandb']:
        wandb.log({
            "avg_residue_att_weights": wandb.Image(f'{save_folder}/avg_residue_att_weights.png'),
            "avg_residue_att_weights_w_specials": wandb.Image(f'{save_folder}/avg_residue_att_weights_w_specials.png')
        })

    print('Attention plot is saved')


def save_test_predictions_to_wandb(metrics, test_dataset, output_path, log_to_wandb=False):
    labels = test_dataset.phosphosite_data_dict['kinase_ids']

    def format_labels(label_list):
        if isinstance(label_list, list):
            return ','.join(label_list)
        return str(label_list)
    
    def get_label_property(label_list, property):
        label_list = label_list.split(',')
        if isinstance(label_list, list):
            return ','.join([test_dataset.kinase_info_dict[label][property] for label in label_list])
        else:
            return test_dataset.kinase_info_dict[label_list][property]
    
    label_families = [get_label_property(label, 'family') for label in labels]
    label_groups = [get_label_property(label, 'group') for label in labels]
    labels = [format_labels(label) for label in labels]
    
    probabilities = metrics.probabilities.cpu().numpy()  # Shape: (num_samples, num_classes)
    family_probabilities = metrics.family_probabilities  # Shape: (num_samples, num_families)
    group_probabilities = metrics.group_probabilities  # Shape: (num_samples, num_groups)
    
    # Get phosphosite sequences from the test dataset
    phosphosite_sequences = test_dataset.phosphosite_data_dict['phosphosite_sequences']

    # List of class probability columns
    class_prob_columns = [f'{test_dataset.label_mapping[i]}' for i in range(probabilities.shape[1])]
    family_prob_columns = [f'{metrics.family_mapping[i]}' for i in range(family_probabilities.shape[1])]
    group_prob_columns = [f'{metrics.group_mapping[i]}' for i in range(group_probabilities.shape[1])]

    # Create the DataFrame
    df = pd.DataFrame({
        'phosphosite_input': phosphosite_sequences,
        'label': labels
    })

    family_df = pd.DataFrame({
        'phosphosite_input': phosphosite_sequences,
        'label': labels,
        'label_families': label_families
    })

    group_df = pd.DataFrame({
        'phosphosite_input': phosphosite_sequences,
        'label': labels,
        'label_groups': label_groups
    })
    
    # Add class probability columns to the DataFrame
    for i, class_prob in enumerate(class_prob_columns):
        df[class_prob] = probabilities[:, i]

    for i, class_prob in enumerate(family_prob_columns):
        family_df[class_prob] = family_probabilities[:, i]

    for i, class_prob in enumerate(group_prob_columns):
        group_df[class_prob] = group_probabilities[:, i]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f'Results saved to {output_path}')

    family_df.to_csv(output_path.replace('.csv', '_family_prob.csv'), index=False)
    print(f'Results saved to {output_path.replace(".csv", "_family.csv")}')

    group_df.to_csv(output_path.replace('.csv', '_group_prob.csv'), index=False)
    print(f'Results saved to {output_path.replace(".csv", "_group.csv")}')

    # Family and Group APs
    family_ap_df = pd.DataFrame(list(metrics.family_ap_dict.items()), columns=['Family', 'AP'])
    group_ap_df = pd.DataFrame(list(metrics.group_ap_dict.items()), columns=['Group', 'AP'])
    # Save the DataFrame to a CSV file
    family_ap_df.to_csv(output_path.replace('.csv', '_family_ap.csv'), index=False)
    group_ap_df.to_csv(output_path.replace('.csv', '_group_ap.csv'), index=False)

    # Optionally log to wandb
    if log_to_wandb:
        # Convert DataFrame to wandb.Table
        table = wandb.Table(dataframe=df)
        # Log the table to wandb
        wandb.log({"test_results_table": table})
        print(f'Results logged to wandb.')