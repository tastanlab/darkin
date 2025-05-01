import torch
from torch.utils.data import DataLoader

import os
import lightning as L

from scripts.utils.training_utils import load_model, get_loss_function
from scripts.utils.evaluation_utils import plot_avg_residue_att_weights, save_test_predictions_to_wandb
from scripts.evaluation.metrics import EvaluationMetrics
from scripts.data.model_dataset import create_zero_shot_dataset


def test_model(
    model,
    dataloader,
    loss_fn,
    fabric,
    hyper_parameters
):
    model.eval()

    metric_results = {
        'epoch_loss' : 0
    }
    metrics = EvaluationMetrics(
        label_mapping = dataloader.dataset.label_mapping,
        kinase_info_dict = dataloader.dataset.kinase_info_dict
    )

    # Unseen kinase set (Buraya en son bakacagim)
    
    for batch_idx, data in enumerate(dataloader):
        ### For'un disina alinca forward icinde degistiginde burada da degisiyor. Daha sonra belki baska yol bulunabilir
        unseen_kinases = {
            'sequences' : dataloader.dataset.unseen_data['sequences'].to(fabric.device),
            'properties' : dataloader.dataset.unseen_data['properties'].to(fabric.device),
            'active_sites' : dataloader.dataset.unseen_data['active_sites'].to(fabric.device),
            'sequence_lengths' : dataloader.dataset.unseen_data['sequence_lengths'].to(fabric.device),
            'active_site_lengths' : dataloader.dataset.unseen_data['active_site_lengths'].to(fabric.device),
            'att_mask_sequences' : dataloader.dataset.unseen_data['att_mask_sequences'].to(fabric.device),
            'att_mask_active_sites' : dataloader.dataset.unseen_data['att_mask_active_sites'].to(fabric.device)
        }

        # Forward pass
        with torch.no_grad():
            outputs = model(
                data['phosphosites'],
                data['kinases'],
                unseen_kinases,
                data['sequence_lengths']
            )

            # Compute loss
            if loss_fn.__name__ == 'focal_loss':
                loss = loss_fn(
                    outputs['kinase_logit'],
                    outputs['unique_logits'],
                    data['kinase_idx'],
                    dataloader.dataset.class_counts if hyper_parameters.get('use_weighted_loss', False) else None,
                    dataloader.dataset.label_mapping if hyper_parameters.get('use_weighted_loss', False) else None,
                    hyper_parameters.get('focal_loss_gamma', 0.0),
                    hyper_parameters['temperature']
                )
            elif loss_fn.__name__ == 'multilabel_binary_cross_entropy_loss':
                loss = loss_fn(outputs['unique_logits'], data['labels'], hyper_parameters['temperature'])
            elif loss_fn.__name__ == 'cross_entropy_with_softmax_scaling':
                loss = loss_fn(outputs['kinase_logit'], outputs['unique_logits'], data['labels'], hyper_parameters['temperature'])
            elif loss_fn.__name__ == 'feature_cross_entropy_loss':
                # feature_cross_entropy_loss(unique_logits, kinase_idx, label_mapping, kinase_info_dict, feature):
                loss = loss_fn(outputs['unique_logits'],
                                data['kinase_idx'],
                                dataloader.dataset.label_mapping,
                                dataloader.dataset.kinase_info_dict,
                                'fine_grained_cluster'   
                        )
            elif loss_fn.__name__ == 'cross_entropy_with_normalization':
                loss = loss_fn(outputs['unique_logits'], data['labels'], hyper_parameters['temperature'])
            elif loss_fn.__name__ == 'cross_entropy':
                loss = loss_fn(
                    outputs['unique_logits'],
                    data['labels'],
                    hyper_parameters.get('temperature', 1.0),
                    hyper_parameters.get('use_soft_probs_ce', False)
                )
            else:
                raise ValueError(f"Loss function {loss_fn.__name__} not implemented")
        
        # Metrics
        if loss_fn.__name__ == 'multilabel_binary_cross_entropy_loss':
            probabilities = torch.sigmoid(outputs['unique_logits'].detach()).cpu()
        else:
            probabilities = torch.nn.functional.softmax(outputs['unique_logits'].detach(), dim=1).cpu()
        predictions = torch.argmax(probabilities, dim=1)

        metric_results['epoch_loss']  += loss.item()
        unique_logits = outputs['unique_logits'].detach().cpu()
        metrics.update_predictions(predictions)
        metrics.update_probabilities(probabilities)
        metrics.update_unique_logits(unique_logits)
        metrics.update_labels(data['labels'].detach().cpu())

    metric_results['epoch_acc'] = metrics.calculate_accuracy()
    metric_results['epoch_top3_acc'] = metrics.calculate_topk_accuracy(k = 3)
    metric_results['epoch_top5_acc'] = metrics.calculate_topk_accuracy(k = 5)
    metric_results['epoch_roc_auc'] = metrics.calculate_roc_auc()
    metric_results['epoch_aupr_with_logits'], metric_results['epoch_phosphosite_ap_with_logits'] = metrics.calculate_aupr(use_logits=True)
    metric_results['epoch_aupr'], metric_results['epoch_phosphosite_ap'] = metrics.calculate_aupr()
    metric_results['epoch_family_macro_ap'], metric_results['epoch_group_macro_ap'], metric_results['epoch_fine_grained_cluster_macro_ap'] = metrics.calculate_family_group_ap()
    metric_results['epoch_family_top1_acc'], metric_results['epoch_group_top1_acc'], metric_results['epoch_fine_grained_cluster_top1_acc'] = metrics.calculate_family_group_acc(k = 1)
    metric_results['epoch_family_top3_acc'], metric_results['epoch_group_top3_acc'], metric_results['epoch_fine_grained_cluster_top3_acc'] = metrics.calculate_family_group_acc(k = 3)
    metric_results['epoch_family_top5_acc'], metric_results['epoch_group_top5_acc'], metric_results['epoch_fine_grained_cluster_top5_acc'] = metrics.calculate_family_group_acc(k = 5)
    metric_results['epoch_group_masked_ap'] = metrics.calculate_group_masked_ap()
    metric_results['metrics'] = metrics
    metric_results['epoch_loss'] /= len(dataloader)
    metrics.calculate_cluster_based_errors('fine_grained_cluster')
    return metric_results, metrics


def test(config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fabric = L.Fabric(
        accelerator = device,
        devices = 1,
        precision = config['training']['precision']
    )

    test_dataset = create_zero_shot_dataset(
        config=config,
        data_type='test'
    )

    if config['training']['normalize_phosphosite_data'] or config['training']['normalize_kinase_data']:
        test_dataset._normalize_data(config, fit_to_data=False)

    test_dataloader = DataLoader(test_dataset, batch_size = config['training']['test_batch_size'], shuffle = False)

    data_shapes = {
        'phosphosite' : test_dataset.phosphosite_data.size(),
        'kinase' : {'sequence' : test_dataset.unseen_data['sequences'].size(), 'properties' : test_dataset.unseen_data['properties'].size()}
    }
    
    # Load Model
    model = load_model(config, data_shapes)
   
    # Loss Function
    loss_fn = get_loss_function(config, mode='eval')

    # Mixed Precision Training using Lighning Fabric
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    model = fabric.setup(model)

    model_results, metrics = test_model(
        model,
        test_dataloader,
        loss_fn,
        fabric,
        config['hyper_parameters']
    )

    # Save Predictions
    if config['logging']['local'].get('save_predictions', False):
        save_pred_dir = f"supplementary/prediction_output_files/{config['logging']['local']['checkpoint_file_name']}"
        if not os.path.exists(save_pred_dir):
            os.makedirs(save_pred_dir)
        prediction_output_filepath = f"{save_pred_dir}/all_preds.csv"
        save_test_predictions_to_wandb(metrics, test_dataset, prediction_output_filepath, log_to_wandb=False)

    # Plotting Key (Residue) Attentions
    if config['phosphosite']['model'].get('plot_residue_attentions', False):
        plot_avg_residue_att_weights(model.phosphosite_model.embedding_model, test_dataset, config)

    ### Logging ###
    log_metrics_keys = [
        'epoch_loss', 'epoch_aupr', 'epoch_aupr_with_logits', 'epoch_phosphosite_ap', 'epoch_phosphosite_ap_with_logits',
        'epoch_roc_auc', 'epoch_acc', 'epoch_top3_acc', 'epoch_top5_acc',
        'epoch_family_macro_ap', 'epoch_group_macro_ap',
        'epoch_family_top1_acc','epoch_family_top3_acc', 'epoch_family_top5_acc',
        'epoch_group_top1_acc', 'epoch_group_top3_acc', 'epoch_group_top5_acc', 'epoch_group_masked_ap'
    ]

    metric_str = ', '.join([
        f"Test {m.replace('epoch_', '').replace('_', ' ').title()}: {model_results[m]:.4f}"
        for m in log_metrics_keys
    ])
    print(metric_str)

    return model_results, metrics