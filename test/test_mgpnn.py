import pytest
import torch
from torch_geometric.data import Batch, Data
from models.mpgnn import MPGNN


@pytest.fixture
def model_params():
    return {
        'node_dim': 32,
        'edge_dim': 16,
        'global_dim': 32,  # Set global_dim to match node_dim
        'hidden_dim': 64,
        'num_layers': 3,
        'num_encoder_layers': 2,
        'num_edge_mlp_layers': 2,
        'num_node_mlp_layers': 2,
        'num_global_mlp_layers': 2,
        'shift_predictor_hidden_dim': [128, 64, 32],
        'shift_predictor_layers': 4,
        'embedding_type': "combined",
        'act': "relu",
        'norm': "batch_norm",
        'dropout': 0.1
    }


@pytest.fixture
def sample_data():
    # Create two graphs with varying input dimensions
    num_nodes = [5, 3]
    num_edges = [8, 4]

    # First graph
    x1 = torch.randn(num_nodes[0], 7)  # Node features with 7 dimensions
    edge_index1 = torch.randint(0, num_nodes[0], (2, num_edges[0]))
    edge_attr1 = torch.randn(num_edges[0], 4)  # Edge features with 4 dimensions

    # Second graph
    x2 = torch.randn(num_nodes[1], 7)  # Node features with 7 dimensions
    edge_index2 = torch.randint(0, num_nodes[1], (2, num_edges[1]))
    edge_attr2 = torch.randn(num_edges[1], 4)  # Edge features with 4 dimensions

    # Create PyG Data objects
    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)

    return Batch.from_data_list([data1, data2])


def test_model_initialization(model_params):
    model = MPGNN(**model_params)
    assert isinstance(model, torch.nn.Module)
    assert len(model.layers) == model_params['num_layers']
    assert hasattr(model, 'shift_predictor')


def test_model_forward(model_params, sample_data):
    model = MPGNN(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    batch = sample_data.batch

    shifts, (node_emb, edge_emb, global_emb) = model(x, edge_index, edge_attr, batch)

    # Test output shapes
    assert shifts.shape == (8, 1)  # Total nodes: 5 + 3 = 8
    assert node_emb.shape == (8, model_params['node_dim'])
    assert edge_emb.shape == (12, model_params['edge_dim'])  # Total edges: 8 + 4 = 12
    assert global_emb.shape == (2, model_params['node_dim'])  # Global embeddings now have size (num_graphs, node_dim)


@pytest.mark.parametrize("embedding_type", ["node", "global", "combined"])
def test_different_embedding_types(model_params, sample_data, embedding_type):
    model_params['embedding_type'] = embedding_type
    model = MPGNN(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    batch = sample_data.batch

    shifts, _ = model(x, edge_index, edge_attr, batch)
    assert shifts.shape == (8, 1)  # Should work for all embedding types


def test_shift_predictor_architectures(model_params, sample_data):
    # Test with integer hidden dim
    model_params['shift_predictor_hidden_dim'] = 64
    model_params['shift_predictor_layers'] = 3
    model1 = MPGNN(**model_params)

    # Test with list hidden dim
    model_params['shift_predictor_hidden_dim'] = [128, 64, 32]
    model_params['shift_predictor_layers'] = 4
    model2 = MPGNN(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    batch = sample_data.batch

    # Both models should work
    shifts1, _ = model1(x, edge_index, edge_attr, batch)
    shifts2, _ = model2(x, edge_index, edge_attr, batch)

    assert shifts1.shape == (8, 1)
    assert shifts2.shape == (8, 1)


def test_gradient_flow(model_params, sample_data):
    model = MPGNN(**model_params)

    x = sample_data.x
    edge_index = sample_data.edge_index
    edge_attr = sample_data.edge_attr
    batch = sample_data.batch

    # Forward pass
    shifts, (node_emb, edge_emb, global_emb) = model(x, edge_index, edge_attr, batch)

    # Create dummy targets
    shift_target = torch.randn_like(shifts)

    # Compute loss and backpropagate
    loss = torch.nn.functional.mse_loss(shifts, shift_target)
    loss.backward()

    # Check if gradients exist
    has_gradient = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_gradient = True
            break
    assert has_gradient, "No gradients found in any parameter"


def test_batch_none_handling(model_params):
    model = MPGNN(**model_params)

    # Set model to eval mode to avoid BatchNorm issues
    model.eval()

    # Single graph data (no batch)
    num_nodes = 5
    num_edges = 8
    x = torch.randn(num_nodes, 7)  # Node features with 7 dimensions
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)  # Edge features with 4 dimensions

    # Should work without providing batch
    with torch.no_grad():  # Disable gradient computation for inference
        shifts, _ = model(x, edge_index, edge_attr)

    assert shifts.shape == (num_nodes, 1)


# Alternative version using layer_norm instead of batch_norm
def test_batch_none_handling_layer_norm(model_params):
    # Modify normalization to layer_norm
    model_params['norm'] = 'layer_norm'
    model = MPGNN(**model_params)

    # Single graph data (no batch)
    num_nodes = 5
    num_edges = 8
    x = torch.randn(num_nodes, 7)  # Node features with 7 dimensions
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 4)  # Edge features with 4 dimensions

    # Should work without providing batch
    shifts, _ = model(x, edge_index, edge_attr)

    assert shifts.shape == (num_nodes, 1)


if __name__ == "__main__":
    pytest.main([__file__])
