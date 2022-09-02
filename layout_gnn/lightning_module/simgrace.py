"""Reference: SimGRACE, Xia et al., WWW 2022 (https://arxiv.org/abs/2202.03104)"""

from copy import deepcopy
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from layout_gnn.nn.loss import NTXentLoss


def get_perturbation(data: torch.Tensor) -> torch.Tensor:
    return torch.normal(mean=0, std=data.std(), size=data.shape, device=data.device)


class SimGRACE(LightningModule):
    """Uses an encoder and its perturbed version to get two representations of the input and applies the loss (e.g. 
    contrastive) on the pair of batch representations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: nn.Module,
        perturbation_magnitude: float = 1.,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.perturbed_encoder = deepcopy(encoder)
        self.projection_head = projection_head
        self.perturbation_magnitude = perturbation_magnitude

        self.loss_fn = loss_fn if loss_fn is not None else NTXentLoss()
        self.learning_rate = learning_rate

    def training_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="test")

    def forward_loss(
        self,
        inputs: Any,
        update_perturbed_encoder: bool = True,
        log_preffix: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        z, z_ = self(inputs, update_perturbed_encoder=update_perturbed_encoder)
        loss = self.loss_fn(z, z_)
        if log_preffix is not None:
            self.log(f"{log_preffix}_loss", loss, **kwargs)
        return loss

    def forward(self, inputs: Any, update_perturbed_encoder: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if update_perturbed_encoder:
            self.uptade_perturbed_encoder()

        z = self.encoder(inputs)
        if self.projection_head is not None:
            z = self.projection_head(z)
        with torch.no_grad():  # Stop gradient on the perturbed branch
            z_ = self.perturbed_encoder(inputs)
            if self.projection_head is not None:
                z_ = self.projection_head(z_)
        
        return z, z_

    def uptade_perturbed_encoder(self) -> None:
        for src, tgt in zip(self.encoder.parameters(), self.perturbed_encoder.parameters()):
            tgt.data = src.data + self.perturbation_magnitude * get_perturbation(src.data)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    