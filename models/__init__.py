# models/__init__.py
from .mpgnn import MPGNN
from .schnet import SchNet  # You'll need to implement this
from .gat import GAT

def get_model(model_type, args):
    if model_type == 'MPGNN':
        return MPGNN(
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            global_dim=args.global_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_encoder_layers=args.num_encoder_layers,
            num_edge_mlp_layers=args.num_edge_mlp_layers,
            num_node_mlp_layers=args.num_node_mlp_layers,
            num_global_mlp_layers=args.num_global_mlp_layers,
            shift_predictor_hidden_dim=args.shift_predictor_hidden_dim,
            shift_predictor_layers=args.shift_predictor_layers,
            embedding_type=args.embedding_type,
            act=args.activation,
            norm=args.normalization,
            dropout=args.dropout
        )
    elif model_type == 'SCHNET':
        return SchNet(
            global_dim=args.global_dim,
            num_global_mlp_layers=args.num_global_mlp_layers,
            num_encoder_layers=args.num_encoder_layers,
            hidden_dim=args.num_filters,
            num_filters=args.num_filters,
            num_interactions=args.num_interactions,
            num_gaussians=args.num_interactions,
            cutoff=args.cutoff,
            max_num_neighbors =args.max_num_neighbors,
            node_dim=args.node_dim,  # Atomic number embedding dimension
            shift_predictor_hidden_dim=args.shift_predictor_hidden_dim,
            shift_predictor_layers=args.shift_predictor_layers,
            embedding_type=args.embedding_type,
            act=args.activation,
            norm=args.normalization,
            dropout=args.dropout
        )
    elif model_type == 'GAT':
        return GAT(
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            global_dim=args.global_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_encoder_layers=args.num_encoder_layers,
            num_global_mlp_layers=args.num_global_mlp_layers,
            heads=args.heads,
            shift_predictor_hidden_dim=args.shift_predictor_hidden_dim,
            shift_predictor_layers=args.shift_predictor_layers,
            embedding_type=args.embedding_type,
            act=args.activation,
            norm=args.normalization,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
