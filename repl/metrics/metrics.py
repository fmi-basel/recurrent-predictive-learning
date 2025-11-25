import torch


def participation_ratio(x):
    """
    Calculates participation ratio of the correlation matrix of the input
    pr = (sum of eigenvalues)^2 / sum of (eigenvalues^2)

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, C)

    Returns:
        torch.Tensor: Participation ratio of the input
    """

    x = x.reshape(-1, x.shape[-1])   # (B, L, C) -> (B*L, C)

    x_c = x - x.mean(dim=0)  # (B*L, C)

    corr_matrix = torch.einsum('ij,ik->jk', x_c, x_c) / x_c.shape[0]  # (C, C)

    eig_vals = torch.linalg.eigvalsh(corr_matrix)  # (C,)
    eig_vals = eig_vals.real  # (C,)

    pr = eig_vals.sum() ** 2 / (eig_vals ** 2).sum()  # (1,)

    return pr


def compute_r2(y_true, y_pred):
    """
    Computes R^2 score

    Args:
        y_true (torch.Tensor): True values
        y_pred (torch.Tensor): Predicted values

    Returns:
        torch.Tensor: R^2 score
    """

    y_true = y_true.detach()
    y_pred = y_pred.detach()

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    r2 = 1 - ss_res / ss_tot

    return r2.item()
