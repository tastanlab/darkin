import torch

def cross_entropy(unique_logits, labels, temperature=1.0, use_soft_probs_ce=False):
    '''
        unique_logits: Tensor of shape (batch_size, num_classes)
        labels: Tensor of shape (batch_size, num_classes) with 1 for true labels and 0 for others (could be multi-label)
    '''
    unique_logits = unique_logits / temperature

    if use_soft_probs_ce:
        labels = labels / labels.sum(dim=1, keepdim=True)
    else:
        labels = torch.argmax(labels, dim=1)

    loss = torch.nn.functional.cross_entropy(unique_logits, labels, reduction='mean')
    return loss

def cross_entropy_with_normalization(unique_logits, labels, temperature=1.0):
    '''
        kinase_logit: Tensor of shape (batch_size)
        unique_logits: Tensor of shape (batch_size, num_classes)

        p(y_i | x) = exp(F(x, y)) / sum(exp(F(x, y')) where y' is unique classes
    '''

    labels = torch.argmax(labels, dim=1)
    kinase_logit = torch.gather(unique_logits, 1, labels.view(-1, 1)).squeeze()

    maxlogits = torch.max(unique_logits, dim=1, keepdim=True)[0]
    numerator = kinase_logit - maxlogits.squeeze() # Shape (batch_size)
    denominator = torch.sum(torch.exp((unique_logits - maxlogits) / temperature), dim=1) # Shape (batch_size)
    softmax_out = torch.exp(numerator / temperature) / (denominator + 1e-15) # Shape (batch_size)
    P = torch.clamp(softmax_out, min=1e-15, max=1.1)
    loss = torch.mean(-torch.log(P))
    return loss

def focal_loss(unique_logits, labels, batch_kinase_indices, class_counts, label_mapping, gamma=0, temperature=1.0):
    '''
        -alpha * ((1 - pt)^gamma) * log(pt)
    '''

    maxlogits = torch.max(unique_logits, dim=1, keepdim=True)[0]

    labels = torch.argmax(labels, dim=1)
    kinase_logit = torch.gather(unique_logits, 1, labels.view(-1, 1)).squeeze()

    numerator = torch.exp((kinase_logit - maxlogits.squeeze()) / temperature) # (b)
    denominator = torch.sum(
        torch.exp((unique_logits - maxlogits) / temperature),
        dim=1
    ) # (b)

    softmax_out = numerator / (denominator + 1e-15)
    P = torch.clamp(softmax_out, min=1e-15, max=1.1)
    
    ce_loss = -torch.log(P)
    focal_term = (1 - P) ** gamma

    alpha = torch.ones_like(kinase_logit)
    if class_counts is not None:
        alpha = torch.tensor([
            class_counts[i] for i in label_mapping.values()
        ]).to(kinase_logit.device)
        
        alpha = alpha + 1 # To prevent zero division for classes with no similar train examples due to threshold
        alpha = 1.0 / alpha
        alpha = alpha / torch.sum(alpha) # Normalized weights
        alpha = alpha[batch_kinase_indices]

    loss = alpha * focal_term * ce_loss
    loss = torch.sum(loss) / torch.sum(alpha)
    return loss


def multilabel_binary_cross_entropy_loss(unique_logits, true_labels, temperature=1.0):
    '''
        unique_logits: Tensor of shape (batch_size, num_kinases) for all training kinases
        true_labels: Tensor of shape (batch_size, num_kinases) with 1 for true labels and 0 for others

        This will calculate binary cross-entropy loss for each kinase independently.
    '''
    # Apply sigmoid to the unique logits to get independent probabilities for each kinase
    all_kinase_probs = torch.sigmoid(unique_logits / temperature)  # Shape: (batch_size, num_kinases)

    # Calculate binary cross-entropy loss for all training kinases
    loss = torch.nn.functional.binary_cross_entropy(all_kinase_probs, true_labels.float(), reduction='mean')
    return loss


def cross_entropy_with_softmax_scaling(unique_logits, true_labels, temperature=1.0):
    '''
        kinase_logit: Tensor of shape (batch_size) for the given true kinase
        unique_logits: Tensor of shape (batch_size, num_kinases) for all kinases
        true_labels: Tensor of shape (batch_size, num_kinases) with 1 for true labels and 0 for others

        This version creates a mask where true labels are scaled by 1/num_true_labels.
    '''

    maxlogits = torch.max(unique_logits, dim=1, keepdim=True)[0]

    labels = torch.argmax(true_labels, dim=1)
    kinase_logit = torch.gather(unique_logits, 1, labels.view(-1, 1)).squeeze()
    
    # Numerator: Use the given `kinase_logit` for the true label as it is
    numerator = torch.exp((kinase_logit - maxlogits.squeeze()) / temperature)  # Shape: (batch_size)

    # Apply softmax across all logits to get the denominator. Not taking sum before we scale the true labels
    softmax_out = torch.exp((unique_logits - maxlogits) / temperature)  # Shape: (batch_size, num_kinases)
    
    # Mask for true labels (1 for all, except true labels which get 1/num_true_labels)
    true_mask = true_labels == 1  # Shape: (batch_size, num_kinases)
    num_true_labels = torch.sum(true_mask, dim=1).float()  # Shape: (batch_size)
    
    mask = torch.ones_like(unique_logits)
    for i in range(unique_logits.shape[0]):
        mask[i, true_mask[i]] = 1 / num_true_labels[i]
    
    # Apply the mask to the softmax output
    adjusted_softmax_out = softmax_out * mask  # Shape: (batch_size, num_kinases)

    # Calculate Denominator: Sum over all adjusted logits
    denominator = torch.sum(adjusted_softmax_out, dim=1)  # Shape: (batch_size)

    # Calculate final softmax probability for the true label
    final_prob = numerator / (denominator + 1e-15)  # Shape: (batch_size)

    # Calculate cross-entropy loss on the final probability
    P = torch.clamp(final_prob, min=1e-15, max=1.0)  # Ensure stability with clamping
    loss = torch.mean(-torch.log(P))

    return loss

# Mert'in group based loss icin yazdigini genel feature based loss'a cevirdim. Birkac da ufak tefek degisiklik yaptim:
# - list(set(class_to_feature.values())) --> sorted(list(set(class_to_feature.values())))
def feature_cross_entropy_loss(unique_logits, kinase_idx, label_mapping, kinase_info_dict, feature):
    # Map class to group, and create group index mappings
    class_to_feature = {class_index: kinase_info_dict[class_name][feature] for class_index, class_name in label_mapping.items()}
    
    # Extract unique groups
    unique_features = sorted(list(set(class_to_feature.values())))
    feature_to_index = {feature: idx for idx, feature in enumerate(unique_features)}

    num_examples = unique_logits.shape[0]
    num_features = len(unique_features)

    # Initialize logits for the feature
    kinase_feature_logits = torch.zeros((num_examples), device=unique_logits.device)
    feature_logits = torch.zeros((num_examples, num_features), device=unique_logits.device)

    # Aggregate probabilities to the group level (single loop)
    for class_index in range(unique_logits.shape[1]):
        kinase_feature_name = class_to_feature[class_index]
        kinase_feature_index = feature_to_index[kinase_feature_name]
        feature_logits[:, kinase_feature_index] += unique_logits[:, class_index]  # Sum feature probabilities

    # Logit scaling for numerical stability (subtract max logits to prevent overflow in exp)
    feature_logits_max = feature_logits.max(dim=1, keepdim=True)[0]
    feature_logits = feature_logits - feature_logits_max

    # Get feature idx from kinase_idx
    feature_kin_idx = torch.tensor([feature_to_index[kinase_info_dict[label_mapping[idx.item()]][feature]] for idx in kinase_idx], device=unique_logits.device)
    
    kinase_feature_logits = torch.gather(feature_logits, 1, feature_kin_idx.view(-1, 1)).squeeze()
    feature_softmax_out = torch.exp(kinase_feature_logits) / (torch.sum(torch.exp(feature_logits), dim=1) + 1e-15)
    
    feature_P = torch.clamp(feature_softmax_out, min=1e-15, max=1.1)
    feature_loss = torch.mean(-torch.log(feature_P))
    return feature_loss