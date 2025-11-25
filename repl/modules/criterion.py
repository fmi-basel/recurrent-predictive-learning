import numpy as np
from repl.modules.utils import compute_score_bilinear, compute_cosine_similarity, get_subsamples
import torch
import torch.nn as nn


class Criterion(nn.Module):
    """Base class for loss functions

    Args:
        pred_steps (int): Number of prediction steps

    Attributes:
        pred_steps (int): Number of prediction steps
    """

    def __init__(self, pred_steps):
        super(Criterion, self).__init__()
        self.pred_steps = pred_steps


class PredLoss(Criterion):
    """Predictive loss function

    Args:
        pred_steps (int): Number of prediction steps
        discount_factor (float): Discount factor for future predictions
        dense_prediction (bool): Whether to use dense prediction or not
        pred_loss_type (str): Type of prediction loss (cosine or l2)
        no_sg (bool): Whether to use the stop-gradient or not

    Attributes:
        pred_steps (int): Number of prediction steps
        discount_factor (float): Discount factor for future predictions
        dense_prediction (bool): Whether to use dense prediction or not
        pred_loss_type (str): Type of prediction loss
        no_sg (bool): Whether to use the stop-gradient or not

    Notes:
        The `forward` method expects the following inputs:
        - pred_target: Target predictions from the encoder or the context integrator network. Shape is (B, L, C) or (B, L, C, H, W).
        - pred: Predictions from the predictor network (B, L, C*pred_steps)
    """

    def __init__(self, 
                 pred_steps,
                 discount_factor=0.0,
                 dense_prediction=False,
                 pred_loss_type='cosine',
                 no_sg=False):
        super().__init__(pred_steps)

        self.discount_factor = discount_factor
        self.dense_prediction = dense_prediction
        self.pred_loss_type = pred_loss_type
        self.no_sg = no_sg

    def forward(self, pred_target, pred):
        if pred_target.dim() == 5:
            # flatten the last few dimensions of pred_target if it is not already in (B, L, C) format
            pred_target = pred_target.view(pred_target.shape[0], pred_target.shape[1], -1)
            pred = pred.view(pred.shape[0], pred.shape[1], -1)

        batch_size = pred_target.shape[0]
        seq_len = pred_target.shape[1]
        encoded_dim = pred_target.shape[2]
        total_loss = 0

        for k in range(1, self.pred_steps + 1):
            target_k = pred_target[:, k:, :]  # (B, L, C) -> (B, L-k, C) 

            if self.dense_prediction:
                # (B, L, C*pred_steps) -> (B, L-k, C)
                pred_k = pred[:, :-k, (k - 1) * encoded_dim : k * encoded_dim]

                # (B, L, C*pred_steps) -> (B, L-k, C)
                bootstrap_targets = self.discount_factor * pred[:, k:, (k - 1) * encoded_dim : k * encoded_dim]
            else:
                # select the last prediction step only, otherwise skip the current iteration
                if k == self.pred_steps:
                    # (B, L, C) -> (B, L-k, C)
                    pred_k = pred[:, :-k, :]

                    # (B, L, C) -> (B, L-k, C)
                    bootstrap_targets = self.discount_factor * pred[:, k:, :]
                else:
                    continue
            
            targets = target_k + bootstrap_targets.detach()

            if not self.no_sg:  # Apply stop-gradient to targets
                targets = targets.detach()

            if self.pred_loss_type == 'cosine':
                # scores.shape: (B, L, 1)
                scores = compute_cosine_similarity(targets.unsqueeze(3), pred_k)
                total_samples = (seq_len - k) * batch_size
                loss = -scores.sum() / total_samples
                total_loss += loss
            elif self.pred_loss_type == "l2":
                total_loss += torch.nn.functional.mse_loss(pred_k, targets)
            else:
                raise NotImplementedError(f"Unknown pred_loss_type: {self.pred_loss_type}")

        if self.dense_prediction:
            total_loss /= self.pred_steps

        return total_loss


class InvLoss(Criterion):
    """Invariance loss based on Latent Predictive Learning (LPL).

    Args:
        pred_steps (int): Number of prediction steps
        pull_coef (float): Pull loss coefficient
        push_coef (float): Push loss coefficient
        decorr_coef (float): Decorrelation loss coefficient
        discount_factor (float): Discount factor for future predictions
        reg_dim (str): Regularization dimension. One of 'batch', 'batch+time', 'time'
        dense_prediction (bool): Whether to use dense prediction or not
        no_sg (bool): Whether to use the stop-gradient or not

    Attributes:
        pred_steps (int): Number of prediction steps
        pull_coef (float): Pull loss coefficient
        push_coef (float): Push loss coefficient
        decorr_coef (float): Decorrelation loss coefficient
        discount_factor (float): Discount factor for future predictions
        reg_dim (str): Regularization dimension. One of 'batch', 'batch+time', 'time'
        dense_prediction (bool): Whether to use dense prediction or not
        no_sg (bool): Whether to use the stop-gradient or not
        pull_loss_val (float): Pull loss value
        push_loss_val (float): Push loss value
        decorr_loss_val (float): Decorrelation loss value

    Notes:
        The `forward` method expects a representation of size (B, L, C) or (B, L, C, H, W) as input.
    """
    def __init__(
            self, 
            pred_steps, 
            pull_coef=1.0, 
            push_coef=1.0, 
            decorr_coef=10.0,
            discount_factor=0.0,
            reg_dim="batch",
            dense_prediction=False,
            no_sg=False):
        super().__init__(pred_steps)

        self.pull_coef = pull_coef
        self.push_coef = push_coef
        self.decorr_coef = decorr_coef
        self.discount_factor = discount_factor
        self.reg_dim = reg_dim
        self.dense_prediction = dense_prediction
        self.no_sg = no_sg
        
        self.pull_loss_val = 0.0
        self.push_loss_val = 0.0
        self.decorr_loss_val = 0.0

    def pull_loss(self, pred):
        # pred.shape: (B, L, C)
        loss = 0.0
        for k in range(1, self.pred_steps + 1):
            if not self.dense_prediction:
                if k != self.pred_steps:
                    continue
            z = pred[:, k:, :]
            p = pred[:, :-k, :]
            if self.discount_factor > 0:
                target = z.clone()
                # cumulative discounted z
                for i in range(z.shape[1] - 1, 0, -1):
                    target[:, i-1, :] += self.discount_factor * z[:, i, :]
            else:
                target = z

            if self.no_sg:
                loss += torch.nn.functional.mse_loss(p, target)
            else:
                loss += torch.nn.functional.mse_loss(p, target.detach())
        
        if self.dense_prediction:
            loss /= self.pred_steps

        return loss

    def push_loss(self, pred):
        # pred.shape: (B, L, C)
        # flatten the last few dimensions of z if it is not already in (B, L, C) format
        if len(pred.shape) == 5:
            pred = pred.mean(dim=-1).mean(dim=-1)
        epsilon = 1e-5
        if self.reg_dim == "time":
            dim = [1]
            denom = pred.shape[1]
        elif self.reg_dim == "batch":
            dim = [0]
            denom = pred.shape[0]
        else:
            dim = [0,1]
            denom = pred.shape[0] * pred.shape[1]
        c_mean = pred.mean(dim=dim, keepdim=True).detach()
        variance = ((pred - c_mean) ** 2).sum(dim=dim) / (denom - 1)
        loss = -torch.log(variance + epsilon).mean()
        return loss
    
    def decorr_loss(self, pred):
        # pred.shape: (B, L, C)
        # flatten the last few dimensions of z if it is not already in (B, L, C) format
        if len(pred.shape) == 5:
            pred = pred.mean(dim=-1).mean(dim=-1)
        c_mean = pred.mean(dim=[0,1]).detach()
        c_centered = pred - c_mean
        c_centered = c_centered.flatten(0, 1)
        cov = torch.einsum('ij,ik->jk', c_centered, c_centered).fill_diagonal_(0) / (pred.shape[0] * pred.shape[1] - 1)
        loss = torch.sum(cov ** 2) / (cov.shape[0] ** 2 - cov.shape[0])
        return loss
    
    def forward(self, z):
        pull_loss = self.pull_loss(z)
        push_loss = self.push_loss(z)
        decorr_loss = self.decorr_loss(z)
        self.pull_loss_val = pull_loss.item()
        self.push_loss_val = push_loss.item()
        self.decorr_loss_val = decorr_loss.item()
        total_loss = self.pull_coef * pull_loss + \
                     self.push_coef * push_loss + \
                     self.decorr_coef * decorr_loss
        return total_loss
    

class SupervisedLoss(nn.Module):
    """Supervised loss function
    
    Args:
        readout (nn.Module): Readout network
        task (str): Task to perform. One of 'phone', 'speaker'
        full_spatial_readout (bool): Whether to use full spatial readout or not

    Attributes:
        readout (nn.Module): Readout network
        task (str): Task to perform
        loss_fn (nn.CrossEntropyLoss): Cross entropy loss function
        full_spatial_readout (bool): Whether to use full spatial readout or not

    Notes:
        The `forward` method expects the following inputs:
        - data: Encoded representation from the encoder network (B, L, C)
        - labels: Labels of shape (B,)
    """
    def __init__(self, readout, task=None, full_spatial_readout=False):
        super().__init__()
        self.readout = readout
        self.task = task
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.regression_loss_fn = nn.MSELoss()
        self.full_spatial_readout = full_spatial_readout

    def forward(self, data, labels):
        if len(data.shape) == 5:
            if self.full_spatial_readout:
                data = data.view(data.shape[0], data.shape[1], -1)
            else:
                data = data.mean(dim=-1).mean(dim=-1)
        
        if self.task == "multitask":
            seq_labels = labels[0]
            dense_labels = labels[1]
            seq_regr_targets = labels[2]
            dense_regr_targets = labels[3]
            aux_labels = labels[4]
            num_seq_labels = num_dense_labels = 0
            num_seq_regr_targets = num_dense_regr_targets = 0
            loss = 0.0

            num_seq_labels = seq_labels.shape[-1] # will be 0 if seq_labels is empty
            for i in range(num_seq_labels):
                logits = self.readout[i](data)
                loss += self.loss_fn(logits, seq_labels[:, i])
            
            num_dense_labels = dense_labels.shape[-1] # will be 0 if dense_labels is empty
            for i in range(num_dense_labels):
                logits = self.readout[num_seq_labels + i](data)
                loss += self.loss_fn(logits, dense_labels[:, :, i].flatten())

            num_seq_regr_targets = seq_regr_targets.shape[-1] # will be 0 if seq_regr_targets is empty
            for i in range(num_seq_regr_targets):
                pred = self.readout[num_seq_labels + num_dense_labels + i](data)
                loss += self.regression_loss_fn(pred.squeeze(), seq_regr_targets[:, i])

            num_dense_regr_targets = dense_regr_targets.shape[-1]
            for i in range(num_dense_regr_targets):
                pred = self.readout[num_seq_labels + num_dense_labels + num_seq_regr_targets + i](data)
                loss += self.regression_loss_fn(pred.squeeze(), dense_regr_targets[:, :, i].flatten())
            
            return loss
        else:
            labels = labels.view(-1)
            if self.task == "seq2seq":
                logits = self.readout(data[:, -labels.shape[0]:])
            else:
                logits = self.readout(data)
            return self.loss_fn(logits, labels)
