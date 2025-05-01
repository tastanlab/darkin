import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

class EvaluationMetrics:
    def __init__(self, label_mapping, kinase_info_dict):
        self.labels = torch.tensor([], dtype=torch.int8)
        self.predictions = torch.tensor([], dtype=torch.int64)
        self.probabilities = torch.tensor([], dtype=torch.float32)
        self.unique_logits = torch.tensor([], dtype=torch.float32)
        self.label_mapping = label_mapping
        self.kinase_info_dict = kinase_info_dict
        self.selected_kinase_indices = torch.tensor([], dtype=torch.int64)

    def get_probabilities(self):
        return self.probabilities
    
    def set_probabilities(self, probabilities):
        self.probabilities = probabilities

    def get_unique_logits(self):
        return self.unique_logits
    
    def set_unique_logits(self, unique_logits):
        self.unique_logits = unique_logits

    def update_labels(self, labels):
        self.labels = torch.cat([self.labels, labels], dim=0)

    def update_predictions(self, predictions):
        self.predictions = torch.cat([self.predictions, predictions], dim=0)

    def update_probabilities(self, probabilities):
        self.probabilities = torch.cat([self.probabilities, probabilities], dim=0)

    def update_unique_logits(self, unique_logits):
        self.unique_logits = torch.cat([self.unique_logits, unique_logits], dim=0)
    
    def update_selected_kinase_indices(self, kinase_indices):
        self.selected_kinase_indices = torch.cat([self.selected_kinase_indices, kinase_indices], dim=0)

    def calculate_accuracy(self):
        """
            Calculates accuracy even if the labels are multilabel. If one prediction matches, the prediction assumed to be true.
        """
        binary_predictions = torch.zeros_like(self.labels)
        binary_predictions[range(len(self.predictions)), self.predictions] = 1
        correct = (binary_predictions * self.labels).sum(dim=1) > 0
        accuracy = correct.float().mean().item()
        return accuracy
    
    def calculate_topk_accuracy(self, k=5):
        """
            Calculates top-k accuracy.
        """
        num_samples = self.labels.size(0)
        num_correct = 0
        for i in range(num_samples):
            label = self.labels[i]
            topk_indices = torch.argsort(self.probabilities[i], descending=True)[:k]
            # Check if any of the predicted labels match with any of the true labels
            if torch.any(label[topk_indices] == 1):
                num_correct += 1
        accuracy = num_correct / num_samples
        return accuracy

    def calculate_aupr(self, use_logits=False):
        """
            Calculate the mean AUPR across all labels.
        """
        self.class_aupr_scores = {}
        self.row_aupr_scores = []
        self.aupr_per_group = defaultdict(list)
        self.aupr_per_family = defaultdict(list)

        # Choose between logits and probabilities
        scores_to_use = self.unique_logits if use_logits else self.probabilities

        # Get classes using label_mapping ################################## AUPR HESABINDA MULTILABEL OLUR MU ?
        for kin_class_idx, kin_class_name in self.label_mapping.items():
            group = self.kinase_info_dict[kin_class_name]['group']
            family = self.kinase_info_dict[kin_class_name]['family']

            # aupr calculation with auc:
            '''precision, recall, _ = precision_recall_curve(
                self.labels[:, kin_class_idx].detach().cpu().numpy(),
                scores_to_use[:, kin_class_idx].detach().cpu().numpy()
            )
            class_aupr = auc(recall, precision)'''

            # aupr calculation with average_precision_score:
            class_aupr = average_precision_score(
                self.labels[:, kin_class_idx].detach().cpu().numpy(),
                scores_to_use[:, kin_class_idx].detach().cpu().numpy()
            )
            self.class_aupr_scores[kin_class_name] = class_aupr
            self.aupr_per_group[group].append(class_aupr)
            self.aupr_per_family[family].append(class_aupr)
        
        macro_aupr = np.mean(list(self.class_aupr_scores.values()))

        # Row-wise AUPR Calculation (for each sample/row)
        for row_idx in range(self.labels.shape[0]):
            row_aupr = average_precision_score(
                self.labels[row_idx, :].detach().cpu().numpy(),
                scores_to_use[row_idx, :].detach().cpu().numpy()
            )
            self.row_aupr_scores.append(row_aupr)

        # Calculate macro AUPR for rows (row-wise AUPR)
        macro_row_aupr = np.mean(self.row_aupr_scores)

        for group, auprs in self.aupr_per_group.items():
            self.aupr_per_group[group] = np.mean(auprs)
        
        for family, auprs in self.aupr_per_family.items():
            self.aupr_per_family[family] = np.mean(auprs)

        return macro_aupr, macro_row_aupr
    
    def calculate_family_group_ap(self):
        family_macro_ap = self._calculate_property_ap('family')
        group_macro_ap = self._calculate_property_ap('group')
        fine_grained_cluster_macro_ap = self._calculate_property_ap('fine_grained_cluster')
        return family_macro_ap, group_macro_ap, fine_grained_cluster_macro_ap
    
    def _calculate_property_ap(self, property_name):
        '''
            Helper method to calculate attribute based ap scores of kinases such as family, group, ec
        '''
        
        kinase_prop_ap_scores = {}
    
        class_probs = self.probabilities.detach().cpu().numpy()
        class_labels = self.labels.detach().cpu().numpy()

        # Create kinase_prop to index
        class_to_kinase_prop = {class_index: self.kinase_info_dict[class_name][property_name] for class_index, class_name in self.label_mapping.items()}
        unique_kinase_props = list(set(class_to_kinase_prop.values()))
        kinase_prop_to_index = {kinase_prop: idx for idx, kinase_prop in enumerate(unique_kinase_props)}
        
        # Create kinase_prop probabilities and labels matrix
        num_examples = class_probs.shape[0]
        num_kinase_props = len(unique_kinase_props)
        kinase_prop_probabilities = np.zeros((num_examples, num_kinase_props), dtype=np.float32)
        kinase_prop_labels = np.zeros((num_examples, num_kinase_props), dtype=np.int8)

        # Aggregate probabilities to the kinase_prop level
        for class_index in range(class_probs.shape[1]):
            kinase_prop_name = class_to_kinase_prop[class_index]
            kinase_prop_index = kinase_prop_to_index[kinase_prop_name]
            kinase_prop_probabilities[:, kinase_prop_index] += class_probs[:, class_index]
            kinase_prop_labels[:, kinase_prop_index] += class_labels[:, class_index]

        kinase_prop_labels = (kinase_prop_labels >= 1).astype(np.int8)
        
        for kinase_prop_name, kinase_prop_index in kinase_prop_to_index.items():

            kinase_prop_ap = average_precision_score(
                kinase_prop_labels[:, kinase_prop_index],
                kinase_prop_probabilities[:, kinase_prop_index]
            )
            kinase_prop_ap_scores[kinase_prop_name] = kinase_prop_ap
        
        kinase_prop_macro_aupr = np.mean(list(kinase_prop_ap_scores.values()))

        # Save the prediction probabilities for each kinase property in class as attribute
        setattr(self, f'{property_name}_probabilities', kinase_prop_probabilities)
        setattr(self, f'{property_name}_mapping', unique_kinase_props)
        setattr(self, f'{property_name}_ap_dict', kinase_prop_ap_scores)

        return kinase_prop_macro_aupr

    def calculate_cluster_based_errors(self, property_name='fine_grained_cluster'):
        if property_name != 'fine_grained_cluster':
            raise ValueError("This function is designed to work only with the 'fine_grained_cluster' property.")

        class_probs = self.probabilities.detach().cpu().numpy()
        class_labels = self.labels.detach().cpu().numpy()
        class_to_kinase_prop = {class_index: self.kinase_info_dict[class_name][property_name]
                                for class_index, class_name in self.label_mapping.items()}

        # Initialize counters for different types of errors
        incorrect_predictions = 0
        incorrect_same_cluster = 0
        incorrect_different_cluster = 0

        # Analyze each example in the dataset
        for i in range(class_labels.shape[0]):
            true_label_indices = np.where(class_labels[i] == 1)[0]
            predicted_index = np.argmax(class_probs[i])

            # Check if the prediction is incorrect
            if predicted_index not in true_label_indices:
                incorrect_predictions += 1  # Count this as an incorrect prediction
                true_clusters = {class_to_kinase_prop[idx] for idx in true_label_indices}
                predicted_cluster = class_to_kinase_prop[predicted_index]

                # Determine if the incorrect prediction is from the same or a different cluster
                if predicted_cluster in true_clusters:
                    incorrect_same_cluster += 1
                else:
                    incorrect_different_cluster += 1

        print(f"\nIncorrect Predictions: {incorrect_predictions}")
        print(f"Incorrect Same Cluster: {incorrect_same_cluster}")
        print(f"Incorrect Different Cluster: {incorrect_different_cluster}\n")

        return incorrect_predictions, incorrect_same_cluster, incorrect_different_cluster
    
    def calculate_family_group_acc(self, k):
        family_macro_acc = self._calculate_property_topk_accuracy('family', k=k)
        group_macro_acc = self._calculate_property_topk_accuracy('group', k=k)
        fine_grained_cluster_macro_acc = self._calculate_property_topk_accuracy('fine_grained_cluster', k=k)
        return family_macro_acc, group_macro_acc, fine_grained_cluster_macro_acc

    def _calculate_property_topk_accuracy(self, property_name, k=5):
        '''
            Helper method to calculate attribute based topk accuracy scores of kinases such as family, group, ec
        '''
    
        class_probs = self.probabilities.detach().cpu()
        class_labels = self.labels.detach().cpu()

        # Create kinase_prop to index
        class_to_kinase_prop = {class_index: self.kinase_info_dict[class_name][property_name] for class_index, class_name in self.label_mapping.items()}
        unique_kinase_props = list(set(class_to_kinase_prop.values()))
        kinase_prop_to_index = {kinase_prop: idx for idx, kinase_prop in enumerate(unique_kinase_props)}
        
        # Create kinase_prop probabilities and labels matrix
        num_examples = class_probs.shape[0]
        num_kinase_props = len(unique_kinase_props)
        kinase_prop_probabilities = torch.zeros((num_examples, num_kinase_props), dtype=torch.float32)
        kinase_prop_labels = torch.zeros((num_examples, num_kinase_props), dtype=torch.int8)

        # Aggregate probabilities to the kinase_prop level
        for class_index in range(class_probs.shape[1]):
            kinase_prop_name = class_to_kinase_prop[class_index]
            kinase_prop_index = kinase_prop_to_index[kinase_prop_name]
            kinase_prop_probabilities[:, kinase_prop_index] += class_probs[:, class_index]
            kinase_prop_labels[:, kinase_prop_index] += class_labels[:, class_index]

        kinase_prop_labels = (kinase_prop_labels >= 1).to(torch.int8)

        # Get top-k predictions
        topk_indices = torch.argsort(kinase_prop_probabilities, dim=1, descending=True)[:, :k]

        # Create a mask to select top-k labels for all samples at once
        topk_mask = torch.zeros_like(kinase_prop_labels, dtype=torch.bool)
        for i in range(k):
            topk_mask.scatter_(1, topk_indices[:, i].unsqueeze(1), True)

        # Calculate the number of correct matches between predicted top-k and true labels
        correct_predictions = torch.any(kinase_prop_labels.gather(1, topk_indices) == 1, dim=1).float()
        prop_topk_accuracy = correct_predictions.mean().item()

        return prop_topk_accuracy


    def calculate_group_masked_ap(self):
        self.class_masked_ap_scores = {}
        classidx_to_group = {class_index: self.kinase_info_dict[class_name]['group'] for class_index, class_name in self.label_mapping.items()}
        
        # Generate a mask for each observation based on active class groups
        group_masks = np.full(self.unique_logits.shape, -np.inf, dtype=np.float32)
        
        for i, label_row in enumerate(self.labels):
            # Active classes and groups for the current row
            active_classes = np.where(label_row == 1)[0]
            true_groups = {classidx_to_group[cls] for cls in active_classes}
            
            for cls_idx, group in classidx_to_group.items():
                if group in true_groups:
                    group_masks[i, cls_idx] = self.unique_logits[i, cls_idx].detach().cpu().numpy()
        
        # Softmax to renormalize probabilities
        masked_logits_tensor = torch.tensor(group_masks)
        masked_probabilities = torch.nn.functional.softmax(masked_logits_tensor, dim=1).cpu().numpy()

        # Calculate Kinase AP
        for kin_class_idx, kin_class_name in self.label_mapping.items():
            class_aupr = average_precision_score(self.labels[:, kin_class_idx].cpu().numpy(), masked_probabilities[:, kin_class_idx])
            self.class_masked_ap_scores[kin_class_name] = class_aupr
        
        macro_masked_ap = np.mean(list(self.class_masked_ap_scores.values()))
        return macro_masked_ap

    
    def calculate_aupr_per_kinase_class_and_group(self, use_logits=False):
        """
            This is the same as calculate_aupr, I'll only return the class and group aupr's
        """
        self.class_aupr_scores = {}
        self.class_precision_scores = {}
        self.class_reall_scores = {}
        self.aupr_per_group = defaultdict(list)

        # Choose between logits and probabilities
        scores_to_use = self.unique_logits if use_logits else self.probabilities

        # Get classes using label_mapping ################################## AUPR HESABINDA MULTILABEL OLUR MU ?
        for kin_class_idx, kin_class_name in self.label_mapping.items():
            group = self.kinase_info_dict[kin_class_name]['group']

            # aupr calculation with auc:
            precision, recall, _ = precision_recall_curve(
                self.labels[:, kin_class_idx].detach().cpu().numpy(),
                scores_to_use[:, kin_class_idx].detach().cpu().numpy()
            )
            # class_aupr = auc(recall, precision)

            # aupr calculation with average_precision_score:
            class_aupr = average_precision_score(self.labels[:, kin_class_idx].detach().cpu().numpy(),
                                                scores_to_use[:, kin_class_idx].detach().cpu().numpy())
            self.class_aupr_scores[kin_class_name] = class_aupr
            self.class_precision_scores[kin_class_name] = precision
            self.class_reall_scores[kin_class_name] = recall
            
            self.aupr_per_group[group].append(class_aupr)
        
        macro_aupr = np.mean(list(self.class_aupr_scores.values()))

        for group, auprs in self.aupr_per_group.items():
            self.aupr_per_group[group] = np.mean(auprs)

        return self.class_aupr_scores, self.aupr_per_group, self.class_precision_scores, self.class_reall_scores

    def calculate_roc_auc(self):
        """
            Calculate the mean ROC AUC across all labels.
        """
        auroc_scores = {}
        # Get classes using label_mapping ################################## AUROC HESABINDA MULTILABEL OLUR MU ?
        for kin_class_idx, kin_class_name in self.label_mapping.items():
            fpr, tpr, _ = roc_curve(
                self.labels[:, kin_class_idx].detach().cpu().numpy(),
                self.probabilities[:, kin_class_idx].detach().cpu().numpy()
            )
            class_auroc = auc(fpr, tpr)
            auroc_scores[kin_class_idx] = class_auroc
        macro_auroc = np.mean(list(auroc_scores.values()))
        return macro_auroc

    def plot_roc_curve(self):
        """
            Calculate the mean ROC AUC across all labels and plot ROC Curve
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.labels.size(1)):
            class_labels = self.labels[:, i].detach().cpu().numpy()
            class_probabilities = self.probabilities[:, i].detach().cpu().numpy()

            fpr[i], tpr[i], _ = roc_curve(class_labels, class_probabilities)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class (Do this for best 5 class)
        top_classes_indices = sorted(range(self.labels.size(1)), key=lambda i: roc_auc[i], reverse=True)[:5]

        plt.figure(figsize=(8, 8))
        for i in top_classes_indices:
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        # Plot the diagonal line for reference
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for each class')
        plt.legend(loc='lower right')
        plt.show()

    def plot_precision_recall_curve(self):
        """
            Calculate the precision-recall curve for each class
        """
        precision = dict()
        recall = dict()
        aupr = dict()

        for i in range(self.labels.size(1)):
            class_labels = self.labels[:, i].detach().cpu().numpy()
            class_probabilities = self.probabilities[:, i].detach().cpu().numpy()

            precision[i], recall[i], _ = precision_recall_curve(class_labels, class_probabilities)
            aupr[i] = auc(recall[i], precision[i])

        # Plot precision-recall curve for each class
        top_classes_indices = sorted(range(self.labels.size(1)), key=lambda i: aupr[i], reverse=True)[:5]

        plt.figure(figsize=(8, 8))
        for i in top_classes_indices:
            plt.step(recall[i], precision[i], where='post', label=f'Class {i} (AUPR = {aupr[i]:.2f})') # plot or step?

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve for each class')
        plt.legend(loc='upper right')
        plt.show()

    def calculate_aupr_per_class(self):
        """
            Calculate the aupr per class, family and group
        """
        aupr_per_class = {}
        aupr_per_family = defaultdict(list)
        aupr_per_group = defaultdict(list)
        for i in range(self.labels.size(1)):
            # Class info
            uniprot_id = self.label_mapping[i]
            family = self.kinase_info_dict[uniprot_id]['family']
            group = self.kinase_info_dict[uniprot_id]['group']
            # Aupr calculation
            class_labels = self.labels[:, i]
            class_probabilities = self.probabilities[:, i]
            aupr_i = average_precision_score(class_labels.detach().cpu().numpy(), class_probabilities.detach().cpu().numpy())
            # Store AUPR per class
            aupr_per_class[uniprot_id] = aupr_i
            # Store AUPR for family and group
            aupr_per_family[family].append(aupr_i)
            aupr_per_group[group].append(aupr_i)
        
        # Calculate mean AUPR for each family and group
        for family, auprs in aupr_per_family.items():
            aupr_per_family[family] = np.mean(auprs)

        for group, auprs in aupr_per_group.items():
            aupr_per_group[group] = np.mean(auprs)
        
        return aupr_per_class, aupr_per_family, aupr_per_group

if __name__ == '__main__':
    # Example usage:
    labels = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0]])
    outputs = {
        'kinase_logit':torch.randn((labels.size(0)), requires_grad=True),
        'unique_logits':torch.randn((labels.size(0), labels.size(1)), requires_grad=True)
    }

    predictions = torch.argmax(outputs['unique_logits'].detach(), dim=1).cpu() # (b)
    probabilities = torch.nn.functional.softmax(outputs['unique_logits'], dim=1).cpu() # (b, unique_kin_size)

