from typing import Callable, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from aim import Run, Image
from layout_gnn.nn.model import LayoutGraphModel
from layout_gnn.nn.neural_rasterizer import CNNNeuralRasterizer
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torch_geometric.nn import GCN, global_mean_pool


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

        self.reconstruction_loss = nn.MSELoss(reduction='sum') if decoder is not None else None
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def _step(self, batch, batch_idx, return_samples: bool = False):
        x, xp, xn = batch["anchor"], batch["pos"], batch["neg"]
        
        z, zp, zn = self(x), self(xp), self(xn)

        triplet_loss = self.triplet_loss(z, zp, zn)
        if self.decoder is not None:
            # We need to add to dimensions to z: the height and width of the image
            y_pred = self.decoder(z.unsqueeze(-1).unsqueeze(-1))
            # The target images are in the format [batch, size, size, dim], but we need [batch, dim, size, size]
            y_true = batch["image"].transpose(1, -1)
            reconstruction_loss = self.reconstruction_loss_weight * self.reconstruction_loss(y_pred, y_true)
            loss = triplet_loss + reconstruction_loss
            samples = None
            if return_samples:
                samples = {
                    'pred': y_pred[0, :, :, :].transpose(-2, -1),
                    'gold': y_true[0, :, :, :].transpose(-2, -1),
                }
                
            return loss, triplet_loss, reconstruction_loss, samples
        else:
            return triplet_loss, None, None, None


    def training_step(self, batch, batch_idx):
        return_samples = batch_idx % 1000 == 0
        loss, triplet_loss, reconstruction_loss, samples = self._step(batch, batch_idx, return_samples=return_samples)

        if triplet_loss is not None:
            self.log("train_triplet_loss", triplet_loss, on_epoch=True)
        if reconstruction_loss is not None:
            self.log("train_reconstruction_loss", reconstruction_loss, on_epoch=True)
            
        if samples:
            
            # If experiment is Aim experiment
            if type(self.logger.experiment) == Run:
                self.logger.experiment.track(Image(samples['pred']), name='train_pred_image')
                self.logger.experiment.track(Image(samples['gold']), name='train_gold_image')    

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, triplet_loss, reconstruction_loss, samples = self._step(batch, batch_idx, return_samples=True)

        if triplet_loss is not None:
            self.log("val_triplet_loss", triplet_loss)
        if reconstruction_loss is not None:
            self.log("val_reconstruction_loss", reconstruction_loss)
        if samples:
            # If experiment is Aim experiment
            if type(self.logger.experiment) == Run:
                self.logger.experiment.track(Image(samples['pred']), name='val_pred_image')
                self.logger.experiment.track(Image(samples['gold']), name='val_gold_image')
                
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, triplet_loss, reconstruction_loss, samples = self._step(batch, batch_idx)

        if triplet_loss is not None:
            self.log("test_triplet_loss", triplet_loss)
        if reconstruction_loss is not None:
            self.log("test_reconstruction_loss", reconstruction_loss)
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LayoutGraphModelCNNNeuralRasterizer(EncoderDecoderWithTripletLoss):
    """Extends `EncoderDecoderWithTripletLoss` to instantiate the encoder and decoder models and log hparams."""
    # TODO: Review defaults
    def __init__(
        self,
        num_labels: int,
        label_embedding_dim: int = 32,
        bbox_embedding_layer_dims: Union[Sequence[int], int] = 32,
        gnn_hidden_channels: int = 128,
        gnn_num_layers: int = 3,
        gnn_out_channels: Optional[int] = None,
        gnn_model_cls: Type[nn.Module] = GCN,
        use_edge_attr: bool = False,
        num_edge_labels: Optional[int] = None,
        edge_label_embedding_dim: Optional[int] = None,
        readout: Optional[Callable[[torch.Tensor, Data], torch.Tensor]] = None,
        cnn_output_dim: Optional[int] = None,
        cnn_hidden_dim: int = 8,
        cnn_output_size: int = None,
        triplet_loss_distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        triplet_loss_margin: float = 1,
        triplet_loss_swap: bool = False,
        reconstruction_loss_weight: float = 1,
        lr: float = 0.001,
        **kwargs
    ):
        """
        Args:
            num_labels (int): Number of classes in the node label attribute.
            label_embedding_dim (int): dimension of the label embeddings.
            bbox_embedding_layer_dims (Union[Sequence[int], int]): Layer dimensions of the MLP that embeds the
                bounding box. If a single dimension is provided, the bounding box is embedded with a single linear
                layer.
            gnn_hidden_channels (int): Dimension of the hidden node representation in the GNN.
            gnn_num_layers (int): Number of GNN layers.
            gnn_out_channels (Optional[int], optional): Dimension of the output node representation of the GNN. If not
                provided, same as gnn_hidden_channels.
            gnn_model_cls (Type[BasicGNN], optional): Class of the GNN model, must follow the torch_geometric BasicGNN
                format. Defaults to GCN.
            use_edge_attr (bool, optional): If True, the edge label will be embedded and used as edge attribute. 
                Defaults to False.
            num_edge_labels (Optional[int], optional): Number of classes in the edge label attribute. Defaults to None.
            edge_label_embedding_dim (Optional[int], optional): _description_. Defaults to None.
            readout (Optional[Callable[[torch.Tensor, data.Data], torch.Tensor]]): Callable that receives the tensor of
                node embeddings and the input graph/batch and returns the graph embeddings. If None, the tensor of node
                embeddings is returned.
            cnn_output_dim (Optional[int]): Number of channels of the decoded image. If not provided, the decoder is
                not used.
            cnn_hidden_dim (int): Hidden dimension of the decoder CNN.
            cnn_output_size (Optional[int]): Size of the decoded image. If not provided, the decoder is not used.
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
        if readout is None:
            # The readout is mandatory in this setup
            readout = lambda x, inputs: global_mean_pool(x, batch=inputs.batch)

        encoder = LayoutGraphModel(
            num_labels=num_labels,
            label_embedding_dim=label_embedding_dim,
            bbox_embedding_layer_dims=bbox_embedding_layer_dims,
            gnn_hidden_channels=gnn_hidden_channels,
            gnn_num_layers=gnn_num_layers,
            gnn_out_channels=gnn_out_channels,
            gnn_model_cls=gnn_model_cls,
            use_edge_attr=use_edge_attr,
            num_edge_labels=num_edge_labels,
            edge_label_embedding_dim=edge_label_embedding_dim,
            readout=readout,
            **kwargs,
        )
        if cnn_output_dim is not None and cnn_output_size is not None:
            decoder = CNNNeuralRasterizer(
                input_dim=gnn_out_channels if gnn_out_channels is not None else gnn_hidden_channels,
                output_dim=cnn_output_dim,
                hidden_dim=cnn_hidden_dim,
                output_size=cnn_output_size,
            )
        else:
            decoder = None

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            triplet_loss_distance_function=triplet_loss_distance_function,
            triplet_loss_margin=triplet_loss_margin,
            triplet_loss_swap=triplet_loss_swap,
            reconstruction_loss_weight=reconstruction_loss_weight,
            lr=lr
        )
        
        self.save_hyperparameters()
