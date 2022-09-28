"""Reference: SimGRACE, Xia et al., WWW 2022 (https://arxiv.org/abs/2202.03104)"""

from copy import deepcopy
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch_geometric.data import Data
from torch_geometric.nn import GIN, global_mean_pool

from layout_gnn.nn.loss import NTXentLoss, VICRegLoss
from layout_gnn.nn.model import LayoutGraphModel
from layout_gnn.nn.utils import get_mlp


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
        lr: float = 0.01,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.perturbed_encoder = deepcopy(encoder)
        self.projection_head = projection_head
        self.perturbation_magnitude = perturbation_magnitude

        self.loss_fn = loss_fn if loss_fn is not None else NTXentLoss()
        self.learning_rate = lr

    def training_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="train", on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="val", on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="test", on_epoch=True)

    def forward_loss(
        self,
        inputs: Any,
        update_perturbed_encoder: bool = True,
        log_preffix: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        z, z_ = self(inputs, update_perturbed_encoder=update_perturbed_encoder)
        loss = self.loss_fn(z, z_)
        
        if not isinstance(loss, dict):
            if log_preffix is not None:
                self.log(f"{log_preffix}_loss", loss, **kwargs)
        else:
            total_loss = 0
            for k, v in loss.items():
                if log_preffix is not None:
                    self.log(f"{log_preffix}_{k}", v, **kwargs)
                total_loss += v
            
            if log_preffix is not None:   
                self.log(f"{log_preffix}_loss", total_loss, **kwargs)
            loss = total_loss
                
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


class LayoutGNNSimGRACE(SimGRACE):
    """Extends `SimGRACE` to instantiate the encoder and projection head."""

    # TODO: Review defaults
    def __init__(
        self,
        num_labels: int,
        label_embedding_dim: int = 64,
        bbox_embedding_layer_dims: Union[Sequence[int], int] = 64,
        gnn_hidden_channels: int = 300,
        gnn_num_layers: int = 5,
        gnn_out_channels: Optional[int] = None,
        gnn_model_cls: Type[nn.Module] = GIN,
        use_edge_attr: bool = False,
        num_edge_labels: Optional[int] = None,
        edge_label_embedding_dim: Optional[int] = None,
        readout: Callable[[torch.Tensor, Data], torch.Tensor] = global_mean_pool,
        projection_head_dims: Optional[Union[int, Sequence[int]]] = None,
        perturbation_magnitude: float = 1.,
        loss_fn: Optional[nn.Module] = None,
        lr: float = 0.01,
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
                format. Defaults to GIN.
            use_edge_attr (bool, optional): If True, the edge label will be embedded and used as edge attribute. 
                Defaults to False.
            num_edge_labels (Optional[int], optional): Number of classes in the edge label attribute. Defaults to None.
            edge_label_embedding_dim (Optional[int], optional): _description_. Defaults to None.
            readout (Optional[Callable[[torch.Tensor, data.Data], torch.Tensor]]): Callable that receives the tensor of
                node embeddings and the input graph/batch and returns the graph embeddings. If None, the tensor of node
                embeddings is returned.
            projection_head_dims (Optional[Union[int, Sequence[int]]]): Layer sises of the projection head MLP.
            perturbation_magnitude (float): Magnitude of the perturbations for the perturbed branch in SimGRACE.
            loss_fn (Optional[nn.Module]): Loss function used to train the encoder. If not provided, NTXent is used.
            lr (float, optional): Learning rate. Defaults to 0.001.
        """

        # Build the encoder
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

        # Build the projection head
        in_features = gnn_out_channels or gnn_hidden_channels  # encoder output size
        if projection_head_dims is None:
            # By default, we use a two layer MLP with the same layer dimensions as the encoder output
            projection_head = get_mlp(in_features, num_layers=2)
        elif isinstance(projection_head_dims, int):
            # If it is an int, we use a single linear layer with that dimension
            projection_head = get_mlp(in_features, projection_head_dims)
        else:
            projection_head = get_mlp(in_features, *projection_head_dims)

        super().__init__(
            encoder=encoder,
            projection_head=projection_head,
            perturbation_magnitude=perturbation_magnitude,
            loss_fn=loss_fn,
            lr=lr,
        )


class LayoutGNNSimGRACENTXent(LayoutGNNSimGRACE):
    """Extends `LayoutGNNSimGRACE` to instantiate the NTXent loss and log hparams."""
    def __init__(
        self,
        num_labels: int,
        label_embedding_dim: int = 64,
        bbox_embedding_layer_dims: Union[Sequence[int], int] = 64,
        gnn_hidden_channels: int = 300,
        gnn_num_layers: int = 5,
        gnn_out_channels: Optional[int] = None,
        gnn_model_cls: Type[nn.Module] = GIN,
        use_edge_attr: bool = False,
        num_edge_labels: Optional[int] = None,
        edge_label_embedding_dim: Optional[int] = None,
        readout: Callable[[torch.Tensor, Data], torch.Tensor] = global_mean_pool,
        projection_head_dims: Optional[Union[int, Sequence[int]]] = None,
        perturbation_magnitude: float = 1,
        lr: float = 0.01,
        # NTXentLoss parameter
        temperature: float = 0.5,
        **kwargs):
        super().__init__(
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
            projection_head_dims=projection_head_dims,
            perturbation_magnitude=perturbation_magnitude,
            loss_fn=NTXentLoss(temperature=temperature),
            lr=lr,
            **kwargs
        )
        self.save_hyperparameters()


class LayoutGNNSimGRACEVICReg(LayoutGNNSimGRACE):
    """Extends `LayoutGNNSimGRACE` to instantiate the VICReg loss and log hparams."""
    def __init__(
        self,
        num_labels: int,
        label_embedding_dim: int = 64,
        bbox_embedding_layer_dims: Union[Sequence[int], int] = 64,
        gnn_hidden_channels: int = 300,
        gnn_num_layers: int = 5,
        gnn_out_channels: Optional[int] = None,
        gnn_model_cls: Type[nn.Module] = GIN,
        use_edge_attr: bool = False,
        num_edge_labels: Optional[int] = None,
        edge_label_embedding_dim: Optional[int] = None,
        readout: Callable[[torch.Tensor, Data], torch.Tensor] = global_mean_pool,
        projection_head_dims: Optional[Union[int, Sequence[int]]] = None,
        perturbation_magnitude: float = 1,
        lr: float = 0.01,
        weight_decay: float = 1e-6,
        # VICRegLoss parameters
        invariance_weight: float = 25.,
        variance_weight: float = 25.,
        covariance_weight: float = 1.,
        variance_target: float = 1.,
        epsilon: float = 1e-4,
        **kwargs):
        super().__init__(
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
            projection_head_dims=projection_head_dims,
            perturbation_magnitude=perturbation_magnitude,
            loss_fn=VICRegLoss(
                invariance_weight=invariance_weight,
                variance_weight=variance_weight,
                covariance_weight=covariance_weight,
                regularize_both=False,  # Set to false because we have a stop gradient on the second branch
                variance_target=variance_target,
                epsilon=epsilon,
            ),
            lr=lr,
            **kwargs
        )
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
