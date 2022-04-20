from typing import Callable, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from torch_geometric import data
from torch_geometric.nn import GCN, MLP
from torch_geometric.nn.models.basic_gnn import BasicGNN


class LayoutGraphModel(nn.Module):
    def __init__(
        self,
        num_labels: int,
        label_embedding_dim: int,
        bbox_embedding_layer_dims: Union[Sequence[int], int],
        gnn_hidden_channels: int,
        gnn_num_layers: int,
        gnn_out_channels: Optional[int] = None,
        gnn_model_cls: Type[BasicGNN] = GCN,
        use_edge_attr: bool = False,
        num_edge_labels: Optional[int] = None,
        edge_label_embedding_dim: Optional[int] = None,
        readout: Optional[Callable[[torch.Tensor, data.Data], torch.Tensor]] = None,
        **kwargs,
    ) -> None:
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
        Raises:
            TypeError: If `use_edge_attr` is True (and the dimensions are provided) but the given `gnn_model_cls` does
                not support edge attributes.
        """

        super().__init__()
        self.label_embedding = nn.Embedding(num_embeddings=num_labels, embedding_dim=label_embedding_dim)

        if isinstance(bbox_embedding_layer_dims, int):
            bbox_embedding_layer_dims = (4, bbox_embedding_layer_dims)
        else:
            bbox_embedding_layer_dims = (4, *bbox_embedding_layer_dims)
        self.bbox_embedding = MLP(bbox_embedding_layer_dims)

        if use_edge_attr and num_edge_labels is not None and edge_label_embedding_dim is not None:
            self.edge_label_embedding = nn.Embedding(num_embeddings=num_edge_labels, embedding_dim=edge_label_embedding_dim)
            kwargs["edge_dim"] = edge_label_embedding_dim
        else:
            self.edge_label_embedding = None

        self.gnn = gnn_model_cls(
            in_channels=label_embedding_dim + bbox_embedding_layer_dims[-1],
            hidden_channels=gnn_hidden_channels,
            num_layers=gnn_num_layers,
            out_channels=gnn_out_channels,
            **kwargs
        )

        self.readout = readout

    def forward(self, inputs: data.Data) -> torch.Tensor:
        """
        Args:
            inputs (data.Data): Graph or batch of graphs to be embedded

        Returns:
            torch.Tensor: Tensor with node embeddings
        """
        label = self.label_embedding(inputs["label"])
        bbox = self.bbox_embedding(inputs["bbox"])
        x = torch.cat((label, bbox), dim=-1)
        if self.edge_label_embedding is None:
            x = self.gnn(x=x, edge_index=inputs.edge_index)
        else:
            edge_attr = self.edge_label_embedding(inputs["edge_label"])
            x = self.gnn(x=x, edge_index=inputs.edge_index, edge_attr=edge_attr)

        if self.readout is not None:
            x = self.readout(x, inputs)

        return x
