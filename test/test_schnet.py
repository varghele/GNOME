import pytest
import torch
from torch_geometric.data import Batch, Data
from models.schnet import SchNet


@pytest.fixture
def model_params():
    return {
        'global_dim': 8,
        'num_global_mlp_layers': 2,
        'num_encoder_layers': 2,
        'hidden_dim': 128,
        'num_filters': 128,
        'num_interactions': 6,
        'num_gaussians': 50,
        'cutoff': 10.0,
        'max_num_neighbors': 32,
        'node_dim': 32,  # Atomic number embedding dimension
        'shift_predictor_hidden_dim': [128, 64, 32],
        'shift_predictor_layers': 4,
        'embedding_type': "combined",
        'act': "relu",
        'norm': "batch_norm",
        'dropout': 0.1
    }


@pytest.fixture
def sample_data():
    # Create two molecules
    num_nodes = [5, 3]  # First molecule has 5 atoms, second has 3
    num_edges = [8, 4]  # First molecule has 8 edges, second has 4

    # First molecule
    x1 = torch.randn(num_nodes[0], 32)
    edge_index1 = torch.randint(0, num_nodes[0], (2, num_edges[0]))
    edge_attr1 = torch.randn(num_edges[0], 3)  # Edge features with 3 dimensions

    # Second molecule
    x2 = torch.randn(num_nodes[1], 32)
    edge_index2 = torch.randint(0, num_nodes[1], (2, num_edges[1]))
    edge_attr2 = torch.randn(num_edges[1], 3)  # Edge features with 3 dimensions

    # Create PyG Data objects
    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)

    return Batch.from_data_list([data1, data2])


def test_schnet_initialization(model_params):
    model = SchNet(**model_params)
    assert isinstance(model, torch.nn.Module)
    assert len(model.interactions) == model_params['num_interactions']
    assert hasattr(model, 'shift_predictor')


def test_schnet_forward(model_params, sample_data):
    model = SchNet(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    batch = sample_data.batch

    shifts, (node_emb, edge_emb, global_emb) = model(x, edge_index, edge_attr, batch)

    # Test output shapes
    assert shifts.shape == (8, 1)  # Total nodes: 5 + 3 = 8
    assert node_emb.shape == (8, model_params['hidden_dim'])
    assert edge_emb.shape == (12, model_params['num_gaussians'])  # Edge embeddings after GaussianSmearing
    assert global_emb.shape == (2, model_params['hidden_dim'])  # Global embeddings now have size (num_graphs, hidden_dim)



@pytest.mark.parametrize("embedding_type", ["node", "global", "combined"])
def test_different_embedding_types(model_params, sample_data, embedding_type):
    model_params['embedding_type'] = embedding_type
    model = SchNet(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    batch = sample_data.batch

    shifts, _ = model(x, edge_index, edge_attr, batch)
    assert shifts.shape == (8, 1)


def test_attention_mechanism(model_params):
    model = SchNet(**model_params)
    model.eval()  # Set model to evaluation mode to disable BatchNorm

    # Create a simple graph where attention should matter
    num_nodes = 4
    x = torch.randn(num_nodes, model_params['node_dim'])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 3)  # Edge features with 3 dimensions

    # Forward pass
    shifts, (node_emb, _, _) = model(x, edge_index, edge_attr)

    # Check that node embeddings are different
    # (attention should produce different representations)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            assert not torch.allclose(node_emb[i], node_emb[j])


def test_gradient_flow(model_params, sample_data):
    model = SchNet(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    batch = sample_data.batch

    # Forward pass
    shifts, _ = model(x, edge_index, edge_attr, batch)

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


def test_batch_none_handling(model_params):
    model = SchNet(**model_params)
    model.eval()  # Set to eval mode to avoid batch norm issues

    # Single graph data (no batch)
    num_nodes = 5
    x = torch.randn(num_nodes, model_params['node_dim'])
    edge_index = torch.randint(0, num_nodes, (2, 8))
    edge_attr = torch.randn(8, 3)  # Edge features with 3 dimensions

    # Should work without providing batch
    with torch.no_grad():
        shifts, _ = model(x, edge_index, edge_attr)
    assert shifts.shape == (num_nodes, 1)


if __name__ == "__main__":
    pytest.main([__file__])
