import time
import wandb
import torch
from torch.utils.data import DataLoader

import lightning as L

from scripts.evaluation.metrics import EvaluationMetrics
from scripts.run.test import test_model
from scripts.utils.training_utils import save_model, generate_model, get_optimizer, get_scheduler, get_loss_function
from scripts.data.model_dataset import create_zero_shot_dataset

### --- training ---
def train_one_epoch(model, dataloader, optimizer, loss_fn, fabric, hyper_parameters):
    model.train()

    metric_results = {
        'epoch_loss' : 0
    }
    metrics = EvaluationMetrics(
        label_mapping = dataloader.dataset.label_mapping,
        kinase_info_dict = dataloader.dataset.kinase_info_dict
    )

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
        
        outputs = model(
            data['phosphosites'],
            data['kinases'],
            unseen_kinases,
            data['sequence_lengths']
        )
        
        # Compute loss
        if loss_fn.__name__ == 'focal_loss':
            loss = loss_fn(
                outputs['unique_logits'],
                data['labels'],
                data['kinase_idx'],
                dataloader.dataset.class_counts if hyper_parameters.get('use_weighted_loss', False) else None,
                dataloader.dataset.label_mapping if hyper_parameters.get('use_weighted_loss', False) else None,
                hyper_parameters.get('focal_loss_gamma', 0.0),
                hyper_parameters['temperature']
            )
        elif loss_fn.__name__ == 'multilabel_binary_cross_entropy_loss':
            loss = loss_fn(outputs['unique_logits'], data['labels'], hyper_parameters['temperature'])
        elif loss_fn.__name__ == 'cross_entropy_with_softmax_scaling':
            loss = loss_fn(outputs['unique_logits'], data['labels'], hyper_parameters['temperature'])
        elif loss_fn.__name__ == 'feature_cross_entropy_loss':
            # feature_cross_entropy_loss(unique_logits, kinase_idx, label_mapping, kinase_info_dict, feature):
            loss = loss_fn(
                outputs['unique_logits'],
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
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        #loss.backward()
        fabric.backward(loss)
        optimizer.step()

        # Metrics
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
    return metric_results


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs,
    fabric,
    config
):
    # Training variables
    best_val_macro_aupr = 0
    best_model_results = {}
    training_start_time = time.time()
    all_train_loss = []
    all_val_loss = []
    
    # Training
    for epoch in range(num_epochs):
        train_results = train_one_epoch(model, train_dataloader, optimizer, loss_fn, fabric, config['hyper_parameters'])
        val_results, _ = test_model(model, val_dataloader, get_loss_function(config, mode='eval'), fabric, config['hyper_parameters'])

        ### Logging ###
        log_metrics_keys = [
            'epoch_loss', 'epoch_aupr', 'epoch_aupr_with_logits', 'epoch_phosphosite_ap', 'epoch_phosphosite_ap_with_logits',
            'epoch_roc_auc', 'epoch_acc', 'epoch_top3_acc','epoch_top5_acc',
            'epoch_family_macro_ap', 'epoch_group_macro_ap', 'epoch_fine_grained_cluster_macro_ap',
            'epoch_family_top1_acc', 'epoch_group_top1_acc', 'epoch_fine_grained_cluster_top1_acc',
            'epoch_family_top3_acc', 'epoch_group_top3_acc', 'epoch_fine_grained_cluster_top3_acc',
            'epoch_family_top5_acc', 'epoch_group_top5_acc', 'epoch_fine_grained_cluster_top5_acc',
            'epoch_group_masked_ap'
        ]

        metric_str = ', '.join([
            f"Train {m.replace('epoch_', '').replace('_', ' ').title()}: {train_results[m]:.4f}, Val {m.replace('epoch_', '').replace('_', ' ').title()}: {val_results[m]:.4f}" for m in log_metrics_keys
        ])
        log_message = f"Epoch [{epoch + 1}/{num_epochs}], {metric_str}"
        print(log_message)

        all_train_loss.append(train_results['epoch_loss'])
        all_val_loss.append(val_results['epoch_loss'])

        if config['logging']['wandb']['use_wandb'] and config['run_model_id'] == 0:
            log_dict = {
                'epoch' : epoch,
                'learning_rate' : optimizer.param_groups[0]['lr'],
                **{f'train/{metric}' : value for metric, value in train_results.items() if metric != "metrics"},
                **{f'validation/{metric}' : value for metric, value in val_results.items() if metric != "metrics"}
            }
            wandb.log({**log_dict})

        # Save Model
        if config['training']['save_model'] and val_results["epoch_aupr"] > best_val_macro_aupr:
            best_val_macro_aupr = val_results["epoch_aupr"]
            save_model(
                config = config,
                model_state_dict = model.state_dict(),
                optim_state_dict = optimizer.state_dict()
            )
            best_model_results['train'] = train_results
            best_model_results['val'] = val_results

        # Scheduler Update
        scheduler.step()
    # Training End Logs
    print(f'Training Finished. Average Epoch Time: {(time.time() - training_start_time) / num_epochs} seconds')
    best_model_results["all_train_loss"], best_model_results["all_val_loss"] = all_train_loss, all_val_loss
    return model, best_model_results


def train(config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fabric = L.Fabric(
        accelerator = device,
        devices = 1,
        precision = config['training']['precision']
    )

    # Dataset & Dataloaders
    train_dataset = create_zero_shot_dataset(
        config=config,
        data_type='train'
    )
    val_dataset = create_zero_shot_dataset(
        config=config,
        data_type='validation'
    )

    # Input Normalization
    if config['training']['normalize_phosphosite_data'] or config['training']['normalize_kinase_data']:
        train_dataset._normalize_data(config, fit_to_data=True)
        val_dataset.phosphosite_embed_scaler = train_dataset.phosphosite_embed_scaler
        val_dataset.kinase_embed_scaler = train_dataset.kinase_embed_scaler
        val_dataset._normalize_data(config, fit_to_data=False)

    train_dataloader = DataLoader(train_dataset, batch_size = config['training']['train_batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['training']['test_batch_size'], shuffle = False)

    data_shapes = {
        'phosphosite' : train_dataset.phosphosite_data.size(),
        'kinase' : {'sequence' : train_dataset.unseen_data['sequences'].size(), 'properties' : train_dataset.unseen_data['properties'].size()}
    }

    # Model
    model = generate_model(config, data_shapes)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Percentage of Trainable Parameters: {round(sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()), 4) * 100}%')

    # Optimizer & Scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # Loss Function
    loss_fn = get_loss_function(config, mode='train')

    # Mixed Precision Training using Lightning Fabric
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    model, optimizer = fabric.setup(model, optimizer)
    
    model, model_results = train_model(
        model = model,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        optimizer = optimizer,
        scheduler = scheduler,
        loss_fn = loss_fn,
        num_epochs = config['training']['num_epochs'],
        fabric = fabric,
        config = config
    )
    
    return model_results