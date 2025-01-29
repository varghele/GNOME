import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MLP
from typing import Optional, Union, Callable, List

class GAT(nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            global_dim: int,
            hidden_dim: int,
            num_layers: int,
            heads: int = 4,
            dropout: float = 0.1,
            shift_predictor_hidden_dim: Union[int, List[int]] = 64,
            shift_predictor_layers: int = 3,
            embedding_type: str = "combined",
            act: str = "relu",
            norm: str = "batch_norm"
    ):
        super().__init__()
        self.embedding_type = embedding_type

        # Ensure hidden_dim is divisible by heads
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        # Initial node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        # First layer
        self.gat_layers.append(GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim
        ))

        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim
            ))

        # Last GAT layer
        self.gat_layers.append(GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False  # Average the attention heads
        ))

        # Global feature processing
        self.global_mlp = MLP(
            in_channels=global_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            dropout=dropout,
            act=act,
            norm=norm
        )

        # Determine input dimension for shift predictor
        if embedding_type == "node":
            shift_predictor_in_dim = hidden_dim
        elif embedding_type == "global":
            shift_predictor_in_dim = hidden_dim
        else:  # combined
            shift_predictor_in_dim = 2 * hidden_dim

        # Create hidden channels list for shift predictor
        if isinstance(shift_predictor_hidden_dim, int):
            hidden_channels = shift_predictor_hidden_dim
        else:
            hidden_channels = shift_predictor_hidden_dim[-1]  # Use the last dimension

        # Shift predictor using MLP
        self.shift_predictor = MLP(
            in_channels=shift_predictor_in_dim,
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=shift_predictor_layers,
            dropout=dropout,
            act=act,
            norm=norm
        )

    def forward(self, x, edge_index, edge_attr, u, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial node embedding
        x = self.node_embedding(x)

        # Process global features
        u_processed = self.global_mlp(u)

        # GAT layers
        for gat_layer in self.gat_layers:
            x = F.elu(gat_layer(x, edge_index, edge_attr))

        # Prepare input for shift prediction
        if self.embedding_type == "node":
            shift_input = x
        elif self.embedding_type == "global":
            shift_input = u_processed[batch]
        else:  # combined
            shift_input = torch.cat([x, u_processed[batch]], dim=1)

        # Predict shifts
        shifts = self.shift_predictor(shift_input)

        return shifts, (x, edge_attr, u_processed)


if __name__ == "__main__":
    # Test the model
    model = GAT(
        node_dim=32,
        edge_dim=16,
        global_dim=8,
        hidden_dim=64,
        num_layers=3,
        heads=4,
        dropout=0.1,
        shift_predictor_hidden_dim=64,  # Single integer for hidden dimension
        shift_predictor_layers=3,
        embedding_type="combined",
        act="relu",
        norm="batch_norm"
    )

    # Create example data
    num_nodes = 10
    num_edges = 15

    # Node features (batch_size x node_dim)
    x = torch.randn(num_nodes, 32)

    # Edge indices (2 x num_edges)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Edge features (num_edges x edge_dim)
    edge_attr = torch.randn(num_edges, 16)

    # Global features (num_graphs x global_dim)
    u = torch.randn(2, 8)  # 2 graphs

    # Batch assignment (num_nodes)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[5:] = 1  # Second half of nodes belong to second graph

    # Forward pass
    with torch.no_grad():
        shifts, (node_embeddings, edge_embeddings, global_embeddings) = model(x, edge_index, edge_attr, u, batch)

    # Print shapes
    print("\nInput shapes:")
    print(f"x: {x.shape}")
    print(f"edge_index: {edge_index.shape}")
    print(f"edge_attr: {edge_attr.shape}")
    print(f"u: {u.shape}")
    print(f"batch: {batch.shape}")

    print("\nOutput shapes:")
    print(f"shifts: {shifts.shape}")
    print(f"node_embeddings: {node_embeddings.shape}")
    print(f"edge_embeddings: {edge_embeddings.shape}")
    print(f"global_embeddings: {global_embeddings.shape}")
