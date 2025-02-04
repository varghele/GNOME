# models/schnet.py
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter
from torch_geometric.nn import MLP
from typing import Optional, Union, Callable, List


class SchNetInteraction(MessagePassing):
    def __init__(self, hidden_dim, num_filters, num_gaussians, cutoff):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff

        # Filter network
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, num_filters)
        )

        # Dense layers for atom updates
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.filter_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dense1.weight)
        self.dense1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dense2.weight)
        self.dense2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, edge_length):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_length=edge_length)

    def message(self, x_j, edge_attr, edge_length):
        W = self.filter_net(edge_attr)
        return x_j * W

    def update(self, aggr_out, x):
        out = self.dense1(aggr_out)
        out = nn.ReLU()(out)
        out = self.dense2(out)
        return x + out


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SchNet(nn.Module):
    def __init__(
            self,
            global_dim: int,
            num_global_mlp_layers: int,
            num_encoder_layers: int,
            hidden_dim=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            max_num_neighbors=32,
            node_dim=32,  # Atomic number embedding dimension
            shift_predictor_hidden_dim: Union[int, List[int]] = 64,
            shift_predictor_layers: int = 3,
            embedding_type: str = "combined",
            act: str = "relu",
            norm: str = "batch_norm",
            dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.embedding_type = embedding_type
        self.act=act
        self.norm=norm
        self.dropout=dropout
        self.node_dim = node_dim
        self.global_dim = global_dim
        self.num_encoder_layers = num_encoder_layers

        # Initialize node encoder as None (will be initialized in forward)
        self.node_encoder = None

        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            SchNetInteraction(
                hidden_dim=hidden_dim,
                num_filters=num_filters,
                num_gaussians=num_gaussians,
                cutoff=cutoff
            ) for _ in range(num_interactions)
        ])

        # Global feature processing
        self.global_mlp = MLP(
            in_channels=global_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_global_mlp_layers,
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

        # Create channel list for shift predictor MLP
        if isinstance(shift_predictor_hidden_dim, int):
            channel_list = [shift_predictor_in_dim] + [shift_predictor_hidden_dim] * (shift_predictor_layers - 1) + [1]
        else:
            channel_list = [shift_predictor_in_dim] + list(shift_predictor_hidden_dim) + [1]

        # Shift predictor MLP
        self.shift_predictor = MLP(
            channel_list=channel_list,
            act=act,
            norm=norm,
            dropout=dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        for interaction in self.interactions:
            interaction.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initialize node encoder if not already initialized
        if self.node_encoder is None:
            node_in_channels = x.size(1)  # Infer input dimension from node features
            self.node_encoder = MLP(
                in_channels=node_in_channels,
                hidden_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_layers=self.num_encoder_layers,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout
            ).to(x.device)  # Move node_encoder to the same device as x

        # Convert atomic numbers to embeddings
        h = self.node_encoder(x)

        # Compute edge lengths
        row, col = edge_index
        edge_length = torch.norm(edge_attr, dim=1)

        # Expand distances
        edge_attr = self.distance_expansion(edge_length)

        # Interaction blocks
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_attr, edge_length)

        # Initialize random global attribute u with the same size as hidden_dim
        u = torch.randn(batch.max() + 1, self.global_dim, device=x.device)  # Random u with size (num_graphs, hidden_dim)

        # Process global features
        u_processed = self.global_mlp(u)

        # Prepare input for shift prediction
        if self.embedding_type == "node":
            shift_input = h
        elif self.embedding_type == "global":
            shift_input = u_processed[batch]
        else:  # combined
            shift_input = torch.cat([h, u_processed[batch]], dim=1)

        # Predict shifts
        shifts = self.shift_predictor(shift_input)

        return shifts, (h, edge_attr, u_processed)


if __name__ == "__main__":
    # Test the model
    model = SchNet(
        hidden_dim=128,
        global_dim=32,
        num_global_mlp_layers=1,
        num_encoder_layers=2,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        max_num_neighbors=32,
        node_dim=32,
        shift_predictor_hidden_dim=64,  # Single integer for hidden dimension
        shift_predictor_layers=3,
        embedding_type="combined",
        act="relu",
        norm="batch_norm",
        dropout=0.1
    )

    # Create example data
    num_nodes = 10
    x = torch.randn(num_nodes, 32)  # Node features with 32 dimensions
    edge_index = torch.randint(0, num_nodes, (2, 15))  # Edge indices
    edge_attr = torch.randn(15, 3)  # Edge features with 3 dimensions

    # Forward pass
    shifts, (node_embeddings, edge_embeddings, global_embeddings) = model(x, edge_index, edge_attr)

    # Print shapes
    print("\nInput shapes:")
    print(f"x: {x.shape}")
    print(f"edge_index: {edge_index.shape}")
    print(f"edge_attr: {edge_attr.shape}")

    print("\nOutput shapes:")
    print(f"shifts: {shifts.shape}")
    print(f"node_embeddings: {node_embeddings.shape}")
    print(f"edge_embeddings: {edge_embeddings.shape}")
    print(f"global_embeddings: {global_embeddings.shape}")
