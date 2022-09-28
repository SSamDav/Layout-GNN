"""Reference: VICReg, Bards et al., ICLR 2022 (https://arxiv.org/abs/2105.04906)"""

from typing import Tuple
import torch
import torch.nn as nn


def var_cov_reg_term(data: torch.Tensor, gamma: float = 1, epsilon: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variance and covariance regularization terms of the VICReg loss.

    Computes the covariance matrix of the features (second dimension) in data. The variance regularization term is
    computed from the diagonal of the covariance matrix, while the covariance term uses the off-diagonal elements
    coefficients.
    """
    n, d = data.shape
    if n < 2:
        # We need a batch with at least two samples to compute the covariance matrix
        return torch.tensor(0., device=data.device), torch.tensor(0., device=data.device)

    # Covariance matrix for the `d` channels 
    cov = data.T.cov()
    # Variance term
    std = torch.sqrt(torch.diagonal(cov) + epsilon)  # Regularized std, `epsilon` prevents numerical instabilities.
    var_term = torch.clamp(gamma - std, min=0).mean()
    #Covariance term
    cov_off_diagonal = cov[~torch.eye(d, dtype=bool, device=cov.device)]
    cov_term = cov_off_diagonal.pow(2).sum() / d
    return var_term, cov_term


def vicreg_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    invariance_weight: float = 25.,
    variance_weight: float = 25.,
    covariance_weight: float = 1.,
    regularize_both: bool = True,
    variance_target: float = 1.,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Variance-invariance-covariance regularization loss.

    Args:
        a (torch.Tensor): Representations from the first branch (N x D).
        b (torch.Tensor): Representations from the second branch (N x D).
        invariance_weight (float, optional): Weight of the invariance term. Defaults to 25.
        variance_weight (float, optional): Weight of the variance term. Defaults to 25.
        covariance_weight (float, optional): Weight of the covariance term. Defaults to 1.
        regularize_both (bool, optional): Whether to apply variance and covariance regularization on the
            representations of both branches. If False, these terms are only computed for the first branch (use this
            setting if the second branch has a stop gradient). Defaults to True.
        variance_target (float, optional): The gamma parameter of the variance regularization term. Defaults to 1.
        epsilon (float, optional): Small scalar to prevent numerical instabilities. Defaults to 1e-4.

    Returns:
        torch.Tensor: Scalar value of the loss.
    """

    inv = nn.functional.mse_loss(a, b)
    var, cov = var_cov_reg_term(a, gamma=variance_target, epsilon=epsilon)
    if regularize_both:
        _var, _cov = var_cov_reg_term(b, gamma=variance_target, epsilon=epsilon)
        var += _var
        cov += _cov
    return {
        'invariance_loss': invariance_weight * inv, 
        'variance_loss': variance_weight * var,  
        'covariance_loss': covariance_weight * cov
    }



class VICRegLoss(nn.Module):
    """Variance-invariance-covariance regularization loss."""
    
    def __init__(
        self,
        invariance_weight: float = 25.,
        variance_weight: float = 25.,
        covariance_weight: float = 1.,
        regularize_both: bool = True,
        variance_target: float = 1.,
        epsilon: float = 1e-4,
    ):
        """
        Args:
            invariance_weight (float, optional): Weight of the invariance term. Defaults to 25.
            variance_weight (float, optional): Weight of the variance term. Defaults to 25.
            covariance_weight (float, optional): Weight of the covariance term. Defaults to 1.
            regularize_both (bool, optional): Whether to apply variance and covariance regularization on the
                representations of both branches. If False, these terms are only computed for the first branch (use 
                this setting if the second branch has a stop gradient). Defaults to True.
            variance_target (float, optional): The gamma parameter of the variance regularization term. Defaults to 1.
            epsilon (float, optional): Small scalar to prevent numerical instabilities. Defaults to 1e-4.
        """
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.regularize_both = regularize_both
        self.variance_target = variance_target
        self.epsilon = epsilon

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return vicreg_loss(
            a=a,
            b=b,
            invariance_weight=self.invariance_weight,
            variance_weight=self.variance_weight,
            covariance_weight=self.covariance_weight,
            regularize_both=self.regularize_both,
            variance_target=self.variance_target,
            epsilon=self.epsilon,
        )
