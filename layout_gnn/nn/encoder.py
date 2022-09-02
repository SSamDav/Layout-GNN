from typing import Callable, Optional, Type

import torch
import torch.nn as nn
from torch_geometric import data
from torch_geometric.nn import GCN
from torch_geometric.nn.models.basic_gnn import BasicGNN


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_node_types: int,
        node_embedding_dim: int,
        gnn_num_layers: int,
        gnn_hidden_channels: Optional[int] = None,
        gnn_out_channels: Optional[int] = None,
        gnn_model_cls: Type[BasicGNN] = GCN,
        num_edge_types: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None,
        readout: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            num_node_types (int): Number of classes in the node attribute.
            node_embedding_dim (int): dimension of the node input embeddings.
            gnn_num_layers (int): Number of GNN layers.
            gnn_hidden_channels (int): Dimension of the hidden node representation in the GNN. If not provided,
                node_embedding_dim is used.
            gnn_out_channels (Optional[int], optional): If provided, the will apply a final linear transformation to
                convert hidden node embeddings from the GNN to this output size.
            gnn_model_cls (Type[BasicGNN], optional): Class of the GNN model, must follow the torch_geometric BasicGNN
                format. Defaults to GCN.
            num_edge_types (Optional[int], optional): Number of classes in the edge attribute. If not provided, the 
                edge attribute is not used.
            edge_embedding_dim (Optional[int], optional): dimension of the edge input embeddings. If not provided, the 
                edge attribute is not used.
            readout (Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]): Callable that receives the tensor
                of node embeddings and the tensor of batch indexes, and returns the graph level embeddings. If None,
                the tensor of node embeddings is returned.
        Raises:
            TypeError: If the edge embedding dimensions are provided but the given `gnn_model_cls` does not support
                edge attributes.
        """

        super().__init__()
        self.node_embedding = nn.Embedding(num_embeddings=num_node_types, embedding_dim=node_embedding_dim)

        if num_edge_types is not None and edge_embedding_dim is not None:
            self.edge_embedding = nn.Embedding(num_embeddings=num_edge_types, embedding_dim=edge_embedding_dim)
            kwargs["edge_dim"] = edge_embedding_dim
        else:
            self.edge_embedding = None

        self.gnn = gnn_model_cls(
            in_channels=node_embedding_dim,
            hidden_channels=gnn_hidden_channels or node_embedding_dim,
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
        x = self.node_embedding(inputs.x)
        if self.edge_embedding is None:
            x = self.gnn(x=x, edge_index=inputs.edge_index)
        else:
            edge_attr = self.edge_embedding(inputs.edge_attr)
            x = self.gnn(x=x, edge_index=inputs.edge_index, edge_attr=edge_attr)

        if self.readout is not None:
            x = self.readout(x, inputs.batch)

        return x
