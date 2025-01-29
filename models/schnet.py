# models/schnet.py
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter


class SchNetInteraction(MessagePassing):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super().__init__(aggr='add')
        self.hidden_channels = hidden_channels
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
        self.dense1 = nn.Linear(hidden_channels, hidden_channels)
        self.dense2 = nn.Linear(hidden_channels, hidden_channels)

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
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            max_num_neighbors=32,
            node_dim=32,  # Atomic number embedding dimension
            readout_hidden_dim=128
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        # Embedding layer for atomic numbers
        self.embedding = nn.Embedding(node_dim, hidden_channels)

        # Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            SchNetInteraction(
                hidden_channels=hidden_channels,
                num_filters=num_filters,
                num_gaussians=num_gaussians,
                cutoff=cutoff
            ) for _ in range(num_interactions)
        ])

        # Output network for chemical shift prediction
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, readout_hidden_dim),
            nn.ReLU(),
            nn.Linear(readout_hidden_dim, readout_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(readout_hidden_dim // 2, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        for layer in self.readout:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, u, batch=None):
        # Convert atomic numbers to embeddings
        h = self.embedding(x.squeeze(-1).long())

        # Compute edge lengths
        row, col = edge_index
        edge_length = torch.norm(edge_attr, dim=1)

        # Expand distances
        edge_attr = self.distance_expansion(edge_length)

        # Interaction blocks
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_attr, edge_length)

        # Predict chemical shifts
        shifts = self.readout(h)

        return shifts, (h, edge_attr, u)


if __name__ == "__main__":
    # Test the model
    model = SchNet()
    num_nodes = 10
    num_edges = 20

    x = torch.randint(0, 10, (num_nodes, 1))  # Atomic numbers
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)  # 3D coordinates
    u = torch.randn(2, 8)  # Global features (batch size 2)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[5:] = 1

    shifts, (node_emb, edge_emb, global_emb) = model(x, edge_index, edge_attr, u, batch)
    print(f"Shifts shape: {shifts.shape}")
    print(f"Node embeddings shape: {node_emb.shape}")
