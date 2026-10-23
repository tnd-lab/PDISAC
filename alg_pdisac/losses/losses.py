# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import math
from losses.loss_util import weighted_loss

_reduction_modes = ["none", "mean", "sum"]


@weighted_loss
def l1_loss(pred, target, reduction="mean"):
    return F.l1_loss(pred, target, reduction=reduction)


@weighted_loss
def mse_loss(pred, target, reduction="mean"):
    return F.mse_loss(pred, target, reduction=reduction)


# ------------------------------------------------------------------------
# Regularization Functions
# ------------------------------------------------------------------------


def l1_regularization(model, lambda_l1=1e-4):
    """
    Compute L1 regularization loss (Lasso) over all model parameters.

    Encourages sparsity by penalizing the absolute value of weights.
    Useful when you expect many weights to be zero.

    Args:
        model (nn.Module): The neural network model.
        lambda_l1 (float): Regularization strength. Default: 1e-4.

    Returns:
        torch.Tensor: Scalar L1 regularization loss.
    """
    l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_reg = l1_reg + torch.sum(torch.abs(param))
    return lambda_l1 * l1_reg


def l2_regularization(model, lambda_l2=1e-4):
    """
    Compute L2 regularization loss (Ridge) over all model parameters.

    Penalizes large weights by summing squared values, encouraging
    small and distributed weights. Equivalent to weight decay.

    Args:
        model (nn.Module): The neural network model.
        lambda_l2 (float): Regularization strength. Default: 1e-4.

    Returns:
        torch.Tensor: Scalar L2 regularization loss.
    """
    l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l2_reg = l2_reg + torch.sum(param**2)
    return lambda_l2 * l2_reg


def elastic_regularization(model, lambda_l1=1e-4, lambda_l2=1e-4):
    """
    Compute Elastic Net regularization loss over all model parameters.

    Combines L1 (Lasso) and L2 (Ridge) regularization:
        elastic = lambda_l1 * ||w||_1 + lambda_l2 * ||w||_2^2

    Balances sparsity (L1) with weight smoothness (L2), useful when
    there are correlated features or groups of important weights.

    Args:
        model (nn.Module): The neural network model.
        lambda_l1 (float): L1 regularization strength. Default: 1e-4.
        lambda_l2 (float): L2 regularization strength. Default: 1e-4.

    Returns:
        torch.Tensor: Scalar Elastic Net regularization loss.
    """
    l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_reg = l1_reg + torch.sum(torch.abs(param))
        l2_reg = l2_reg + torch.sum(param**2)
    return (lambda_l1 * l1_reg) + (lambda_l2 * l2_reg)


# ------------------------------------------------------------------------
# Loss Classes
# ------------------------------------------------------------------------


class MAELoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MAELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.loss_weight * l1_loss(pred, target, self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.
    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.loss_weight * mse_loss(pred, target, reduction=self.reduction)


def target_nmse(pred, target, M_tars, eps=1e-8):
    """
    Target-support term of eq_L_rec, ||M_tars * (pred - target)||_F^2, normalized per
    sample by the target energy ||M_tars * target||_F^2 and averaged over the batch.
    The normalisation keeps the ~50 target cells of a 2x512x256 map from being
    negligible next to the global term.  Dimensionless, O(1), like the global term.
    """
    num = (M_tars * (pred - target)).pow(2).flatten(1).sum(1)
    den = (M_tars * target).pow(2).flatten(1).sum(1).clamp_min(eps)
    return (num / den).mean()


def target_rel_mse(pred, target, M_tars, eps=1e-6):
    """
    Target-support term of eq_L_rec normalized PER CELL by the clean power of that cell:
        sum_cells M * |pred - target|^2 / (|target|^2 + eps)  /  sum_cells M ,
    with |.|^2 = I^2 + Q^2 over the two channels and M = M_tars (same on both channels).
    Since M_tars = P / P_peak,m inside the box of target m, each term is about
    |pred - target|^2 / P_peak,m: every target is weighted by its OWN peak, so erasing a
    weak target costs as much as erasing the strongest one.  target_nmse instead divides
    by the summed energy of all targets of the map, which the strongest target dominates.
    Averaged over the batch; dimensionless, O(1).
    """
    P = target.pow(2).sum(1, keepdim=True)              # |Z_prbs|^2 per cell
    err = (pred - target).pow(2).sum(1, keepdim=True)   # |Zhat - Z_prbs|^2 per cell
    M = M_tars[:, :1]
    num = (M * err / (P + eps)).flatten(1).sum(1)
    den = M.flatten(1).sum(1).clamp_min(1.0)
    return (num / den).mean()
