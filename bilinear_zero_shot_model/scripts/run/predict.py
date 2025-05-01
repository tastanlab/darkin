import torch
from torch.utils.data import DataLoader

import pandas as pd
import lightning as L

from scripts.utils.training_utils import load_model
from scripts.evaluation.metrics import EvaluationMetrics
from scripts.data.model_dataset import create_zero_shot_dataset


def predict_model(
    model,
    dataloader,
    fabric,
    config
):
    model.eval()
    metrics = EvaluationMetrics(
        label_mapping = dataloader.dataset.label_mapping,
        kinase_info_dict = dataloader.dataset.kinase_info_dict
    )
    
    for batch_idx, data in enumerate(dataloader):
        unseen_kinases = {
            'sequences' : dataloader.dataset.unseen_data['sequences'].to(fabric.device),
            'properties' : dataloader.dataset.unseen_data['properties'].to(fabric.device)
        }

        # Forward pass
        with torch.no_grad():
            outputs = model(
                data['phosphosites'],
                data['kinases'],
                unseen_kinases
            )

        predictions = torch.argmax(outputs['unique_logits'].detach(), dim=1).cpu()
        probabilities = torch.nn.functional.softmax(outputs['unique_logits'].detach(), dim=1).cpu()
        metrics.update_predictions(predictions)
        metrics.update_probabilities(probabilities)
        metrics.update_labels(data['labels'].detach().cpu())

    return metrics


def predict(config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fabric = L.Fabric(
        accelerator = device,
        devices = 1,
        precision = '32-true' #precision_type
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

    # Mixed Precision Training using Lighning Fabric
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    model = fabric.setup(model)

    metrics = predict_model(
        model,
        test_dataloader,
        fabric,
        config
    )

    return metrics