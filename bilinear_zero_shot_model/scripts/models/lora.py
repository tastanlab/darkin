import torch
from functools import partial


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, dropout=0.05):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.scaling * (self.dropout(x) @ self.A @ self.B)
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, dropout=0.05):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, dropout
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    

def lora(model, lora_config=None):
    
    if lora_config is None:
        # Some defaults that I chose
        print('LoRA config not provided. Using default values.')
        lora_config = {}

    lora_r = lora_config.get("lora_rank", 8)
    lora_alpha = lora_config.get("lora_alpha", 16)
    lora_dropout = lora_config.get("lora_dropout", 0.05)
    lora_query = lora_config.get("lora_query", True)
    lora_key = lora_config.get("lora_key", True)
    lora_value = lora_config.get("lora_value", True)
    lora_output = lora_config.get("lora_output", False)
    lora_intermediate = lora_config.get("lora_intermediate", False)

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    if 'facebook/esm' in model.embedding_model.config._name_or_path:
        for layer in model.embedding_model.encoder.layer:
            if lora_query:
                layer.attention.self.query = assign_lora(layer.attention.self.query)
            if lora_key:
                layer.attention.self.key = assign_lora(layer.attention.self.key)
            if lora_value:
                layer.attention.self.value = assign_lora(layer.attention.self.value)
            if lora_output:
                layer.attention.output.dense = assign_lora(layer.attention.output.dense)
                layer.output.dense = assign_lora(layer.output.dense)
            if lora_intermediate:
                layer.intermediate.dense = assign_lora(layer.intermediate.dense)

    # Make sure LoRA layers are trainable
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    return model

# If one wants to combine LoRA weights with the original weights, they can use the following code snippet:
def combine_lora(layer):
    if isinstance(layer, LinearWithLoRA):
        with torch.no_grad():
            # Combine LoRA weights into the original weight matrix
            layer.linear.weight += layer.lora.A @ layer.lora.B
        return layer.linear  # Return the original linear layer after combining
    return layer

def remove_lora(model):
    layers_to_replace = []  # Collect layers to modify

    # Iterate over the model's named modules to find LoRA layers
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            combined_module = combine_lora(module)
            layers_to_replace.append((name, combined_module))  # Collect layer info

    # Replace LoRA layers with combined layers after iteration
    for name, combined_module in layers_to_replace:
        # Use 'rpartition' to split the layer name into parent and child module
        parent_name, _, child_name = name.rpartition('.')
        
        # If there is a parent module, update the child module in it
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, combined_module)
        else:
            # Directly update the model if no parent
            setattr(model, name, combined_module)
    
    return model