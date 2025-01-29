import pytest
import torch
from torch_geometric.data import Batch, Data
from models.schnet import *


@pytest.fixture
def model_params():
    return {
        'hidden_channels': 128,
        'num_filters': 128,
        'num_interactions': 6,
        'num_gaussians': 50,
        'cutoff': 10.0,
        'max_num_neighbors': 32,
        'node_dim': 32,
        'readout_hidden_dim': 128
    }


@pytest.fixture
def sample_data():
    # Create two molecules
    num_nodes = [5, 3]  # First molecule has 5 atoms, second has 3
    num_edges = [8, 4]  # First molecule has 8 edges, second has 4

    # First molecule
    x1 = torch.randint(0, 10, (num_nodes[0], 1))  # Atomic numbers
    edge_index1 = torch.randint(0, num_nodes[0], (2, num_edges[0]))
    edge_attr1 = torch.randn(num_edges[0], 3)  # 3D coordinates
    u1 = torch.randn(1, 8)  # Global features

    # Second molecule
    x2 = torch.randint(0, 10, (num_nodes[1], 1))
    edge_index2 = torch.randint(0, num_nodes[1], (2, num_edges[1]))
    edge_attr2 = torch.randn(num_edges[1], 3)
    u2 = torch.randn(1, 8)

    # Create PyG Data objects
    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1, u=u1)
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2, u=u2)

    return Batch.from_data_list([data1, data2])


def test_schnet_initialization(model_params):
    model = SchNet(**model_params)
    assert isinstance(model, torch.nn.Module)
    assert len(model.interactions) == model_params['num_interactions']
    assert model.hidden_channels == model_params['hidden_channels']


def test_gaussian_smearing():
    smearing = GaussianSmearing(0.0, 5.0, 50)
    distances = torch.tensor([0.0, 2.5, 5.0])
    expanded = smearing(distances)
    assert expanded.shape == (3, 50)
    assert torch.all(expanded >= 0)  # Gaussian values should be positive
    assert torch.all(expanded <= 1)  # Gaussian values should be normalized


def test_schnet_forward(model_params, sample_data):
    model = SchNet(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    u = torch.cat([g.u for g in sample_data.to_data_list()], dim=0)
    batch = sample_data.batch

    shifts, (node_emb, edge_emb, global_emb) = model(x, edge_index, edge_attr, u, batch)

    # Test output shapes
    assert shifts.shape == (8, 1)  # Total atoms: 5 + 3 = 8
    assert node_emb.shape == (8, model_params['hidden_channels'])
    assert edge_emb.shape == (12, model_params['num_gaussians'])  # Total edges: 8 + 4 = 12


def test_schnet_interaction_block(model_params):
    interaction = SchNetInteraction(
        hidden_channels=model_params['hidden_channels'],
        num_filters=model_params['num_filters'],
        num_gaussians=model_params['num_gaussians'],
        cutoff=model_params['cutoff']
    )

    num_nodes = 5
    num_edges = 8
    x = torch.randn(num_nodes, model_params['hidden_channels'])
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, model_params['num_gaussians'])
    edge_length = torch.randn(num_edges)

    out = interaction(x, edge_index, edge_attr, edge_length)
    assert out.shape == (num_nodes, model_params['hidden_channels'])


def test_gradient_flow(model_params, sample_data):
    model = SchNet(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    u = torch.cat([g.u for g in sample_data.to_data_list()], dim=0)
    batch = sample_data.batch

    # Forward pass
    shifts, _ = model(x, edge_index, edge_attr, u, batch)

    # Create dummy target
    target = torch.randn_like(shifts)

    # Compute loss and backpropagate
    loss = torch.nn.functional.mse_loss(shifts, target)
    loss.backward()

    # Check if gradients exist
    has_gradient = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_gradient = True
            break
    assert has_gradient, "No gradients found in any parameter"


def test_cutoff_behavior(model_params):
    model = SchNet(**model_params)

    # Create a simple molecule with atoms at different distances
    num_nodes = 3
    x = torch.tensor([[1], [1], [1]])  # Three identical atoms
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])  # Three edges
    edge_attr = torch.tensor([
        [1.0, 0.0, 0.0],  # Distance = 1
        [15.0, 0.0, 0.0],  # Distance > cutoff
        [5.0, 0.0, 0.0]  # Distance = 5
    ])
    u = torch.randn(1, 8)

    shifts, _ = model(x, edge_index, edge_attr, u)
    assert shifts.shape == (3, 1)


if __name__ == "__main__":
    pytest.main([__file__])
