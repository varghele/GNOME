# models/__init__.py
from .mpgnn import MPGNN
#from .schnet import SchNet  # You'll need to implement this

def get_model(model_type, args):
    if model_type == 'MPGNN':
        return MPGNN(
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            global_dim=args.global_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            shift_predictor_hidden_dim=args.shift_predictor_hidden_dim,
            shift_predictor_layers=args.shift_predictor_layers,
            embedding_type=args.embedding_type,
            act=args.activation,
            norm=args.normalization,
            dropout=args.dropout
        )
    elif model_type == 'SCHNET':
        return SchNet(
            # Add SchNet-specific parameters here
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
