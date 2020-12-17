import numpy as np
import torch


# mixup
def mixup_creater(data, targets, alpha=2):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)

    return data, targets, shuffled_targets, lam
