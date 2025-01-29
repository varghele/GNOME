# args.py
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Neural Network Training/Inference')

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'inference'],
                        help='Mode of operation')

    # Model selection and architecture
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['MPGNN', 'SCHNET'],
                        help='Type of model to use')
    parser.add_argument('--node_dim', type=int, default=32,
                        help='Dimension of node features')
    parser.add_argument('--edge_dim', type=int, default=16,
                        help='Dimension of edge features')
    parser.add_argument('--global_dim', type=int, default=8,
                        help='Dimension of global features')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for message passing')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of message passing layers')

    # Shift predictor architecture
    parser.add_argument('--shift_predictor_hidden_dim', type=int, nargs='+', default=[128, 64, 32],
                        help='Hidden dimensions for shift predictor')
    parser.add_argument('--shift_predictor_layers', type=int, default=4,
                        help='Number of layers in shift predictor')
    parser.add_argument('--embedding_type', type=str, default='combined',
                        choices=['node', 'global', 'combined'],
                        help='Type of embeddings to use for shift prediction')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Model components
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'leaky_relu', 'elu', 'tanh'],
                        help='Activation function')
    parser.add_argument('--normalization', type=str, default='batch_norm',
                        choices=['batch_norm', 'layer_norm', 'none'],
                        help='Normalization layer type')

    # Paths and logistics
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save/load model checkpoints')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--model_path', type=str,
                        help='Path to saved model for inference')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')

    return parser.parse_args()