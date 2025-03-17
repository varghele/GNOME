import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer, MLP
try:
    from torch_scatter import scatter_mean, scatter_add
except ImportError:
    from utils.alternative_torch_scatter import scatter_mean
from typing import Optional, Union, Callable, Literal, List


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class EdgeModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_edge_mlp_layers, act, norm, dropout):
        super().__init__()
        self.edge_mlp = MLP(
            in_channels=2 * node_dim + edge_dim, # no global_dim
            hidden_channels=hidden_dim,
            out_channels=edge_dim,
            num_layers=num_edge_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_node_mlp_layers, act, norm, dropout):
        super().__init__()
        self.node_mlp_1 = MLP(
            in_channels=node_dim + edge_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_node_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        )
        self.node_mlp_2 = MLP(
            in_channels=node_dim + hidden_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=node_dim,
            num_layers=num_node_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)  # Removed u[batch]
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, num_global_mlp_layers, act, norm, dropout):
        super().__init__()
        self.global_mlp = MLP(
            in_channels=node_dim + edge_dim,  # No global_dim
            hidden_channels=hidden_dim,
            out_channels=global_dim,
            num_layers=num_global_mlp_layers,
            act=act,
            norm="BatchNorm",#norm,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([
            scatter_mean(x, batch, dim=0),
            scatter_mean(edge_attr, batch[edge_index[0]], dim=0)
        ], dim=1)
        return self.global_mlp(out)


class MPGNN(torch.nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            global_dim: int,
            hidden_dim: int,
            num_layers: int,
            num_encoder_layers: int,
            num_edge_mlp_layers: int,
            num_node_mlp_layers: int,
            num_global_mlp_layers: int,
            shift_predictor_hidden_dim: Union[int, List[int]],
            shift_predictor_layers: int,
            embedding_type: Literal["node", "global", "combined"],
            act: Union[str, Callable] = "relu",
            norm: Optional[str] = "batch_norm",
            dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_type = embedding_type

        # Store all necessary attributes
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_edge_mlp_layers = num_edge_mlp_layers
        self.num_node_mlp_layers = num_node_mlp_layers
        self.num_global_mlp_layers = num_global_mlp_layers
        self.act = act
        self.norm = norm
        self.dropout = dropout

        # Initialize node and edge encoders as None (will be initialized in forward)
        self.node_encoder = None
        self.edge_encoder = None

        # Message passing layers
        self.layers = torch.nn.ModuleList([
            MetaLayer(
                edge_model=EdgeModel(node_dim, edge_dim, hidden_dim, num_edge_mlp_layers, act, norm, dropout),
                node_model=NodeModel(node_dim, edge_dim, hidden_dim, num_node_mlp_layers, act, norm, dropout),
                global_model=GlobalModel(node_dim, edge_dim, global_dim, hidden_dim, num_global_mlp_layers, act, norm, dropout)
            ) for _ in range(num_layers)
        ])

        # Determine input dimension for shift predictor
        if embedding_type == "node":
            shift_predictor_in_dim = node_dim
        elif embedding_type == "global":
            shift_predictor_in_dim = global_dim
        else:  # combined
            shift_predictor_in_dim = node_dim + global_dim

        # Create channel list for shift predictor MLP
        if isinstance(shift_predictor_hidden_dim, int):
            channel_list = [shift_predictor_in_dim] + [shift_predictor_hidden_dim] * (shift_predictor_layers - 1) + [1]
        else:
            channel_list = [shift_predictor_in_dim] + list(shift_predictor_hidden_dim) + [1]

        # Shift predictor MLP
        self.shift_predictor = MLP(
            channel_list=channel_list,
            act=act,
            norm=None,#norm,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initialize node encoder if not already initialized
        if self.node_encoder is None:
            node_in_channels = x.size(1)  # Infer input dimension from node features
            self.node_encoder = MLP(
                in_channels=node_in_channels,
                hidden_channels=self.hidden_dim,
                out_channels=self.node_dim,
                num_layers=self.num_encoder_layers,
                act=self.act,
                norm="BatchNorm",#self.norm,
                dropout=self.dropout
            ).apply(init_weights).to(x.device)  # Move node_encoder to the same device as x

        # Initialize edge encoder if not already initialized
        if self.edge_encoder is None:
            edge_in_channels = edge_attr.size(1)  # Infer input dimension from edge features
            self.edge_encoder = MLP(
                in_channels=edge_in_channels,
                hidden_channels=self.hidden_dim,
                out_channels=self.edge_dim,
                num_layers=self.num_encoder_layers,
                act=self.act,
                norm="BatchNorm",#self.norm,
                dropout=self.dropout
            ).apply(init_weights).to(x.device)  # Move edge_encoder to the same device as x


        # Encode node and edge features
        x = self.node_encoder(x)  # Encode node features
        edge_attr = self.edge_encoder(edge_attr)  # Encode edge features

        # Initialize random global attribute u with the same size as node_dim
        u = torch.randn(batch.max() + 1, x.size(1), device=x.device)  # Random u with size (num_graphs, node_dim)

        # Message passing
        for layer in self.layers:
            x, edge_attr, u = layer(x, edge_index, edge_attr, u, batch)

        # Prepare embeddings for shift prediction
        if self.embedding_type == "node":
            shift_input = x
        elif self.embedding_type == "global":
            # Broadcast global features to all nodes
            shift_input = u[batch]
        else:  # combined
            shift_input = torch.cat([x, u[batch]], dim=1)

        # Predict shifts
        shifts = self.shift_predictor(shift_input)

        return shifts, (x, edge_attr, u)  # Return both shifts and embeddings


# Example usage:
if __name__ == "__main__":
    # Model parameters
    model = MPGNN(
        node_dim=32,
        edge_dim=16,
        global_dim=8,
        hidden_dim=64,
        num_layers=3,
        num_encoder_layers=2,
        num_edge_mlp_layers=2,
        num_node_mlp_layers=2,
        num_global_mlp_layers=2,
        shift_predictor_hidden_dim=[128, 64, 32],  # Custom architecture for shift predictor
        shift_predictor_layers=4,
        embedding_type="combined",  # Use both node and global features
        act="relu",
        norm="batch_norm",
        dropout=0.1
    )

    # Example data
    num_nodes = 10
    num_edges = 15
    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 16)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[5:] = 1  # Second half of nodes belong to second graph

    # Forward pass
    shifts, (node_embeddings, edge_embeddings, global_embeddings) = model(x, edge_index, edge_attr, batch)

    print(f"Predicted shifts shape: {shifts.shape}")  # [num_nodes, 1]
