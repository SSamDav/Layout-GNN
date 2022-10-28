from contextlib import nullcontext
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import LightningModule
from torch_geometric.nn import GIN, global_mean_pool

from layout_gnn.nn.loss.simclr import NTXentLoss
from layout_gnn.nn.loss.vicreg import VICRegLoss
from layout_gnn.nn.model import LayoutGraphModel
from layout_gnn.nn.utils import get_mlp


class JointEmbedding(LightningModule):
    """Core class to pre-train models using a joint embedding predictive architecture.
    By default (i.e. if only the `first_branch` argument is provided) it is equivalent to SimCLR.
    """

    def __init__(
        self,
        first_branch: nn.Module,
        second_branch: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        stop_grad_on_second_branch: bool = False,
        lr: float = 0.001,
    ) -> None:
        super().__init__()

        self.first_branch = first_branch
        self.second_branch = second_branch if second_branch is not None else first_branch
        self.loss_fn = loss_fn or NTXentLoss()
        self.stop_grad_on_second_branch = stop_grad_on_second_branch
        self.lr = lr

    def training_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="train", on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="val", on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        return self.forward_loss(batch, log_preffix="test", on_epoch=True)

    def forward_loss(
        self,
        inputs: Tuple[Any, Any],
        log_preffix: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        z, z_ = self(inputs)
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

    def forward(self, inputs: Tuple[Any, Any]) -> Tuple[Any, Any]:
        x, x_ = inputs

        z = self.first_branch(x)

        context = torch.no_grad() if self.stop_grad_on_second_branch else nullcontext()
        with context:
            z_ = self.second_branch(x_)

        return z, z_

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LayoutGNNMultimodal(JointEmbedding):
    """Extends `JointEmbedding` to instantiate the two branches."""

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
        readout: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = global_mean_pool,
        cnn: Union[str, nn.Module] = "resnet18",
        freeze_cnn: bool = False,
        projection_head_dims: Optional[Union[int, Sequence[int]]] = None,
        loss_fn: Optional[nn.Module] = None,
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
                format. Defaults to GIN.
            use_edge_attr (bool, optional): If True, the edge label will be embedded and used as edge attribute. 
                Defaults to False.
            num_edge_labels (Optional[int], optional): Number of classes in the edge label attribute. Defaults to None.
            edge_label_embedding_dim (Optional[int], optional): _description_. Defaults to None.
            readout (Optional[Callable[[torch.Tensor, torch.Tensot], torch.Tensor]]): Callable that receives the tensor
                of node embeddings and the tensor of batch indexes and returns the graph embeddings. If None, the
                tensor of node embeddings is returned.
            cnn (Union[str, nn.Module]): CNN model for the second branch. Can be a str with the name of a pretrained
                model from torchvision.models, or an instantiated nn.Module.
            projection_head_dims (Optional[Union[int, Sequence[int]]]): Layer sizes of the projection head MLP.
            loss_fn (Optional[nn.Module]): Loss function used to train the encoder. If not provided, NTXent is used.
            lr (float, optional): Learning rate. Defaults to 0.001.
        """

        # Build the encoders
        gnn = LayoutGraphModel(
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

        if isinstance(cnn, str):
            cnn = getattr(torchvision.models, cnn)(weights="DEFAULT")
        if freeze_cnn:
            for param in cnn.parameters():
                param.requires_grad = False

        # Build the projection heads
        in_features_gnn = gnn_out_channels or gnn_hidden_channels  # encoder output size
        # TODO: this is assuming that the last module in `cnn.modules()` is the final layer (is this always the case?)
        #  and that it is a Linear layer (otherwise, instead of `out_features` we would need to get some other
        #  property).
        last_cnn_layer = None
        for last_cnn_layer in cnn.modules():
            pass
        assert last_cnn_layer is not None, "Empty CNN module provided"
        in_features_cnn = last_cnn_layer.out_features

        if projection_head_dims is None:
            # By default, we use a two layer MLP with the same layer dimensions as the encoder output
            projection_head_gnn = get_mlp(in_features_gnn, num_layers=2)
            projection_head_cnn = get_mlp(in_features_cnn, in_features_gnn, num_layers=2)
        elif isinstance(projection_head_dims, int):
            # If it is an int, we use a single linear layer with that dimension
            projection_head_gnn = get_mlp(in_features_gnn, projection_head_dims)
            projection_head_cnn = get_mlp(in_features_cnn, projection_head_dims)
        else:
            projection_head_gnn = get_mlp(in_features_gnn, *projection_head_dims)
            projection_head_cnn = get_mlp(in_features_cnn, *projection_head_dims)

        gnn = nn.Sequential(gnn, projection_head_gnn)
        cnn = nn.Sequential(cnn, projection_head_cnn)

        super().__init__(
            first_branch=gnn,
            second_branch=cnn,
            loss_fn=loss_fn,
            lr=lr,
        )


class LayoutGNNMultimodalNTXent(JointEmbedding):
    """Extends `LayoutGNNMultimodal` to instantiate the NTXent loss and log hparams."""

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
        readout: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = global_mean_pool,
        cnn: Union[str, nn.Module] = "resnet18",
        freeze_cnn: bool = False,
        projection_head_dims: Optional[Union[int, Sequence[int]]] = None,
        lr: float = 0.001,
        # NTXentLoss parameter
        temperature: float = 0.5,        
        **kwargs
    ):
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
            cnn=cnn,
            freeze_cnn=freeze_cnn,
            loss_fn=NTXentLoss(temperature=temperature),
            lr=lr,
            **kwargs
        )
        self.save_hyperparameters()


class LayoutGNNMultimodalVICReg(JointEmbedding):
    """Extends `LayoutGNNSimGRACE` to instantiate the VICReg loss and log hparams."""

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
        readout: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = global_mean_pool,
        cnn: Union[str, nn.Module] = "resnet18",
        freeze_cnn: bool = False,
        projection_head_dims: Optional[Union[int, Sequence[int]]] = None,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        # VICRegLoss parameters
        invariance_weight: float = 25.,
        variance_weight: float = 25.,
        covariance_weight: float = 1.,
        variance_target: float = 1.,
        epsilon: float = 1e-4,
        **kwargs
    ):
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
            cnn=cnn,
            freeze_cnn=freeze_cnn,
            loss_fn=VICRegLoss(
                invariance_weight=invariance_weight,
                variance_weight=variance_weight,
                covariance_weight=covariance_weight,
                regularize_both=True,
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