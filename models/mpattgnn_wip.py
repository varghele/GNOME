import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer


class EdgeModel(nn.Module):
    def __init__(self, edge_dim, node_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention mechanism for edges
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # Concatenate source, destination node features and edge features
        edge_features = torch.cat([src, dest, edge_attr], dim=1)

        # Compute edge updates
        edge_updates = self.edge_mlp(edge_features)

        # Compute attention weights
        attention_weights = self.attention(edge_updates)

        # Apply attention
        edge_updates = edge_updates * attention_weights

        return edge_updates


class NodeModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp_2 = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # Node-level attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # Aggregate edge features for each node
        row, col = edge_index
        edge_features_aggregated = scatter_mean(edge_attr, row, dim=0, dim_size=x.size(0))

        # First MLP
        node_features = torch.cat([x, edge_features_aggregated], dim=1)
        node_features = self.node_mlp_1(node_features)

        # Compute attention weights
        attention_weights = self.attention(node_features)

        # Apply attention
        node_features = node_features * attention_weights

        # Second MLP
        node_features = torch.cat([x, node_features], dim=1)
        node_updates = self.node_mlp_2(node_features)

        return node_updates


class GlobalModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_dim)
        )

        # Global attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # Aggregate node and edge features
        node_features_mean = scatter_mean(x, batch, dim=0)
        edge_features_mean = scatter_mean(edge_attr, batch[edge_index[0]], dim=0)

        # Concatenate with global features
        global_features = torch.cat([u, node_features_mean, edge_features_mean], dim=1)

        # Update global features
        global_features = self.global_mlp(global_features)

        # Compute and apply attention
        attention_weights = self.attention(global_features)
        global_updates = global_features * attention_weights

        return global_updates


class AttentionMessagePassingGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim):
        super().__init__()

        self.message_passing = MetaLayer(
            edge_model=EdgeModel(edge_dim, node_dim, hidden_dim),
            node_model=NodeModel(node_dim, edge_dim, hidden_dim),
            global_model=GlobalModel(node_dim, edge_dim, global_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.message_passing(x, edge_index, edge_attr, u, batch)


# Example usage:
"""
model = AttentionMessagePassingGNN(
    node_dim=32,
    edge_dim=16,
    global_dim=64,
    hidden_dim=128
)

# Forward pass
x_updated, edge_attr_updated, u_updated = model(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_features,
    u=global_features,
    batch=batch_index
)
"""
