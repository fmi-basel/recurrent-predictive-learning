import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import List
import math
import warnings


def get_subsamples(z, pred, subsample_window):
    """Get subsamples of the input

    Args:
        z (torch.Tensor): z tensor in CPC model of shape (B, L, C)
        pred (torch.Tensor): c tensor in CPC model of shape (B, L, C*pred_steps)
        subsample_window (int): Size of the subsample window

    Returns:
        tuple: Tuple containing subsampled tensors of shape (B, subsample_window, C)
    """
    seq_init = np.random.randint(0, pred.shape[1] - subsample_window)

    # (B, L, C*pred_steps) -> (B, subsample_window, C*pred_steps)
    pred = pred[:, seq_init : seq_init + subsample_window]

    # (B, L, C) -> (B, subsample_window, C)
    z = z[:, seq_init : seq_init + subsample_window]
    return z, pred


def compute_score_bilinear(z, pred):
    """Compute score (f) using bilinear function

    Args:
        z (torch.Tensor): Encoded representation from the encoder network of shape (B, L, C, num_samples)
        pred (torch.Tensor): Predictions from the encoder network of shape (B, L, C)

    Returns:
        torch.Tensor: Scores of shape (B, L, num_samples)
    """
    pred = pred.unsqueeze(3)  # (B, L, C) -> (B, L, C, 1)
    logits = torch.einsum("blci,blcj->blij", z, pred).squeeze(-1)  # (B, L, num_samples)

    return logits

def compute_cosine_similarity(z, pred):
    """Compute score (f) using cosine similarity
    
    Args:
        z (torch.Tensor): Encoded representation from the encoder network of shape (B, L, C, num_samples)
        pred (torch.Tensor): Predictions from the encoder network of shape (B, L, C)

    Returns:
        torch.Tensor: Scores of shape (B, L, num_samples)
    """
    pred = pred.unsqueeze(3)  # (B, L, C) -> (B, L, C, 1)
    z_norm = torch.nn.functional.normalize(z, dim=2)
    pred_norm = torch.nn.functional.normalize(pred, dim=2)

    scores = torch.einsum("blci,blcj->blij", z_norm, pred_norm).squeeze(-1)  # (B, L, num_samples)

    return scores


# Code from https://github.com/Lightning-Universe/lightning-bolts/blob/9ae0581776ddfef4d9ccc2f3709499d00e22be2e/src/pl_bolts/optimizers/lr_scheduler.py
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
