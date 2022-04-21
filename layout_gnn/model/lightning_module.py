from re import S
from typing import Callable

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class EncoderDecoderWithTripletLoss(LightningModule):
    def __init__(
        self, 
        encoder: nn.Module, 
        decoder: nn.Module = None, 
        triplet_loss_distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        triplet_loss_margin: float = 1.0,
        triplet_loss_swap: bool = False,
        recunstruction_loss_weight: float = 1.0,
        lr: float = 0.001,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=triplet_loss_distance_function, 
            margin=triplet_loss_margin, 
            swap=triplet_loss_swap,
        )

        self.reconstruction_loss = nn.MSELoss() if decoder is not None else None
        self.reconstruction_loss_weight = recunstruction_loss_weight

        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, xp, xn = batch["anchor"], batch["pos"], batch["neg"]
        
        z, zp, zn = self(x), self(xp), self(xn)

        loss = self.triplet_loss(z, zp, zn)
        if self.decoder is not None:
            y = self.decoder(z)
            loss += self.reconstruction_loss_weight * self.reconstruction_loss(y, batch["image"])

        # TODO: Logging
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # TODO: validation_step and test_step
