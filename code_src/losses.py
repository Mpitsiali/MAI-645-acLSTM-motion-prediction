import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_loss(q_pred, q_true):
    q_pred = q_pred / torch.norm(q_pred, dim=-1, keepdim=True)
    q_true = q_true / torch.norm(q_true, dim=-1, keepdim=True)

    dot_product = torch.sum(q_pred * q_true, dim=-1)
    dot_product = torch.clamp(dot_product, -1, 1)

    angle = torch.acos(torch.abs(dot_product))

    loss = 2 * angle


    return torch.mean(loss)



def angle_distance_loss(x, y):
    cos_error = torch.cos(x - y)
    angle_distance = torch.mean(1 - cos_error)
    return angle_distance
