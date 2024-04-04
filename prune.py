import os
import torch.nn.utils.prune as prune

def pruning(model, prune_percentage = 0.1, prune_type = 'l1' ):
    if prune_type == 'l1':
        pruning_method = prune.L1Unstructured
    elif prune_type == 'random':
        pruning_method = prune.RandomUnstructured
    else: raise ValueError

    total_params = sum(p.numel() for p in model.parameters())
    num_params_prune = int(total_params * prune_percentage)
    print(total_params, num_params_prune)

    parameters_to_prune = [(module, 'weight') for module in model.children()]

#     parameters_to_prune = [
#     (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
# ]
    
    print(num_params_prune)
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount=num_params_prune,
    )

    return model


def main():
    pass

if __name__ == '__main__':
    main()