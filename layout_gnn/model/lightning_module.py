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

    def _step(self, batch, batch_idx):
        x, xp, xn = batch["anchor"], batch["pos"], batch["neg"]
        
        z, zp, zn = self(x), self(xp), self(xn)

        triplet_loss = self.triplet_loss(z, zp, zn)
        if self.decoder is not None:
            reconstruction_loss = self.reconstruction_loss(self.decoder(z), batch["image"])
            loss = triplet_loss + self.reconstruction_loss_weight * reconstruction_loss
            return loss, triplet_loss, reconstruction_loss
        else:
            return triplet_loss, None, None


    def training_step(self, batch, batch_idx):
        loss, triplet_loss, reconstruction_loss = self._step(batch, batch_idx)

        if triplet_loss is not None:
            self.log("train_triplet_loss", triplet_loss)
        if reconstruction_loss is not None:
            self.log("train_reconstruction_loss", reconstruction_loss)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, triplet_loss, reconstruction_loss = self._step(batch, batch_idx)

        if triplet_loss is not None:
            self.log("val_triplet_loss", triplet_loss)
        if reconstruction_loss is not None:
            self.log("val_reconstruction_loss", reconstruction_loss)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, triplet_loss, reconstruction_loss = self._step(batch, batch_idx)

        if triplet_loss is not None:
            self.log("test_triplet_loss", triplet_loss)
        if reconstruction_loss is not None:
            self.log("test_reconstruction_loss", reconstruction_loss)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
