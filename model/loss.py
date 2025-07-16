import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np



def cross_entropy(pred_dict, target_dict, levels, s_temp=0.2, t_temp=0.09):
    ce_losses = []
    for _, level in enumerate(levels):
        pred = pred_dict[level]
        target = target_dict[level]
        pred = -(pred - 2) / 2
        target = -(target - 2) / 2

        b, h, w = pred.shape

        pred_map_flat = pred.view(b, -1)
        target_map_flat = target.view(b, -1)

        pred_map_softmax = F.softmax(pred_map_flat / s_temp, dim=1)
        target_map_softmax = F.softmax(target_map_flat / t_temp, dim=1)


        pred_map = pred_map_softmax.view(b, h, w)
        target_map = target_map_softmax.view(b, h, w)

        loss = -torch.sum(target_map * torch.log(pred_map)) / b
        ce_losses.append(loss)
    return torch.mean(torch.stack(ce_losses, dim=0).float())



def multi_scale_contrastive_loss(corr_maps, levels):
    '''
    corr_maps: dict, key -- level; value -- corr map with shape of [M, N, H, W]
    '''
    matching_losses = []

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

    return torch.mean(torch.stack(matching_losses, dim=0))



def calculate_errors(true_labels, logits, num_classes=45):
    """
    Calculate the average error and median error
    true_labels: true label tensor, shape is [batch_size]
    predicted_labels: predicted label tensor, shape is [batch_size]
    """
    predicted_labels = torch.argmax(logits, dim=1)
    predicted_labels -= num_classes
    errors = torch.abs(true_labels - predicted_labels)

    mean_error = errors.float().mean()

    median_error = torch.median(errors.float())

    return errors, mean_error, median_error

def generate_soft_labels(true_labels, num_classes, sigma=1.0):
    """
    Generate soft labels based on Gaussian distribution, support batch labels
    true_labels: true labels, should be an integer array (such as [20, 21, 19, ...])
    num_classes: number of categories, assumed to be 90
    sigma: standard deviation of Gaussian distribution, controls the smoothness of soft labels
    """
    batch_size = true_labels.size(0)

    class_indices = torch.arange(num_classes, device=true_labels.device)  # [0, 1, ..., 89]

    soft_labels = []

    for i in range(batch_size):
        true_label = true_labels[i]
        distances = torch.abs(class_indices - true_label)

        soft_label = torch.exp(-distances ** 2 / (2 * sigma ** 2))

        soft_label /= soft_label.sum()

        soft_labels.append(soft_label)

    return torch.stack(soft_labels)

