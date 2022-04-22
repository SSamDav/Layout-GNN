from re import S
from typing import Callable

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class EncoderDecoderWithTripletLoss(LightningModule):
    """LightningModule to train the graph layout encoder with a combination of triplet and reconstruction loss."""
    def __init__(
        self, 
        encoder: nn.Module, 
        decoder: nn.Module = None, 
        triplet_loss_distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        triplet_loss_margin: float = 1.0,
        triplet_loss_swap: bool = False,
        reconstruction_loss_weight: float = 1.0,
        lr: float = 0.001,
    ):
        """
        Args:
            encoder (nn.Module): Model that receives the layout graphs and returns the embeddings.
            decoder (nn.Module, optional): Model that resonstructs the layout image from the embedding. If not
                provided, the reconstruction loss is not considered.
            triplet_loss_distance_function (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): the 
                distance metric to be used in the triplet loss. If not provided, euclidean distance is used.
            triplet_loss_margin (float, optional): Margin of the triplet loss. Defaults to 1.0.
            triplet_loss_swap (bool, optional): If True, and if the positive example is closer to the negative example
                than the anchor is, swaps the positive example and the anchor in the loss computation (see "Learning 
                shallow convolutional feature descriptors with triplet losses" by V. Balntas et al). Defaults to False.
            reconstruction_loss_weight (float, optional): Weight of the reconstruction loss relative to the triplet 
                loss. Defaults to 1.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=triplet_loss_distance_function, 
            margin=triplet_loss_margin, 
            swap=triplet_loss_swap,
        )

        self.reconstruction_loss = nn.MSELoss() if decoder is not None else None
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def _step(self, batch, batch_idx):
        x, xp, xn = batch["anchor"], batch["pos"], batch["neg"]
        
        z, zp, zn = self(x), self(xp), self(xn)

        triplet_loss = self.triplet_loss(z, zp, zn)
        if self.decoder is not None:
            y_pred, y_true = self.decoder(z), batch["image"]
            reconstruction_loss = self.reconstruction_loss_weight * self.reconstruction_loss(y_pred, y_true)
            loss = triplet_loss + reconstruction_loss
            return loss, triplet_loss, reconstruction_loss
        else:
            return triplet_loss, None, None


    def training_step(self, batch, batch_idx):
        loss, triplet_loss, reconstruction_loss = self._step(batch, batch_idx)

        if triplet_loss is not None:
            self.log("train_triplet_loss", triplet_loss, on_epoch=True)
        if reconstruction_loss is not None:
            self.log("train_reconstruction_loss", reconstruction_loss, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

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
