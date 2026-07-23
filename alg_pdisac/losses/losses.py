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
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'sum'.
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
