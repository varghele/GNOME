import torch
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from models import get_model
from models.mpgnn import MPGNN
from dataset import OrganometallicDataset, MoleculeDataset
import ast


def filter_model_args(args):
    """
    Filter the relevant arguments for the model from the args namespace.

    Args:
        args (Namespace): The full namespace of arguments.

    Returns:
        dict: A dictionary containing only the arguments needed by the model.
    """
    model_args = {
        'node_dim': args.node_dim,
        'edge_dim': args.edge_dim,
        'global_dim': args.global_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_encoder_layers': args.num_encoder_layers,
        'num_edge_mlp_layers': args.num_edge_mlp_layers,
        'num_node_mlp_layers': args.num_node_mlp_layers,
        'num_global_mlp_layers': args.num_global_mlp_layers,
        'shift_predictor_hidden_dim': args.shift_predictor_hidden_dim,
        'shift_predictor_layers': args.shift_predictor_layers,
        'embedding_type': args.embedding_type,
        'activation': args.activation,
        'normalization': args.normalization,
        'dropout': args.dropout,
    }
    return model_args


# Step 1: Load the Best Model
def load_best_model(model_path, model_type):
    """
    Load the best model using the get_model function.

    Args:
        model_path (str): Path to the saved model checkpoint.
        model_type (str): Type of model to load (e.g., 'MPGNN', 'SCHNET', 'GAT').

    Returns:
        model: The loaded model in evaluation mode.
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path)

    # Check if the checkpoint contains a 'model_state_dict' key
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        # If no 'model_state_dict' key, assume the checkpoint is the state_dict itself
        model_state_dict = checkpoint

    # Initialize the model (assuming MPGNN is the model class)
    model = MPGNN(
        node_dim=checkpoint["model_params"]["node_dim"],
        edge_dim=checkpoint["model_params"]["edge_dim"],
        global_dim=checkpoint["model_params"]["global_dim"],
        hidden_dim=checkpoint["model_params"]["hidden_dim"],
        num_layers=checkpoint["model_params"]["num_layers"],
        num_encoder_layers=checkpoint["model_params"]["num_encoder_layers"],
        num_edge_mlp_layers=checkpoint["model_params"]["num_edge_mlp_layers"],
        num_node_mlp_layers=checkpoint["model_params"]["num_node_mlp_layers"],
        num_global_mlp_layers=checkpoint["model_params"]["num_global_mlp_layers"],
        shift_predictor_hidden_dim=ast.literal_eval(checkpoint["model_params"]["shift_predictor_hidden_dim"]),
        shift_predictor_layers=checkpoint["model_params"]["shift_predictor_layers"],
        embedding_type=checkpoint["model_params"]["embedding_type"],
        act=checkpoint["model_params"]["activation"],
        norm=checkpoint["model_params"]["normalization"],
        dropout=checkpoint["model_params"]["dropout"],
    )

    # Load the model's state dict
    model.load_state_dict(model_state_dict, strict=False)

    model.eval()  # Set the model to evaluation mode

    return model


# Step 2: Load the Organometallic Test Set
def load_test_dataset():
    """
    Load the organometallic test set using the OrganometallicDataset class.
    """
    # Create the organometallic dataset
    test_dataset = MoleculeDataset(root='data') #OrganometallicDataset()

    # Create the DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # Adjust batch size as needed
        shuffle=False,
        num_workers=4  # Adjust number of workers as needed
    )
    return test_loader


# Step 3: Make Predictions on the Test Set
def make_predictions(model, test_loader, device):
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in test_loader:
            # Move batch to the appropriate device (e.g., GPU if available)
            batch = batch.to(device)

            # Make predictions
            predictions = model(batch.x, batch.edge_index, batch.edge_attr)

            # Get the target values
            targets = batch.y.cpu().numpy()

            # Create a mask to ignore NaN values in the targets
            mask = ~np.isnan(targets)

            # Apply the mask to both predictions and targets
            predictions_masked = predictions[0].squeeze().cpu().numpy()[mask]
            targets_masked = targets[mask]

            # Store predictions and ground truth (only non-NaN values)
            all_predictions.append(predictions_masked)
            all_targets.append(targets_masked)

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_predictions, all_targets


# Step 4: Evaluate the Model's Performance
def evaluate_model(all_targets, all_predictions):
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")


# Step 5: Visualize the Predictions (Optional)
def visualize_predictions(all_targets, all_predictions):
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], color='red',
             linestyle='--')  # Diagonal line
    plt.xlabel("Ground Truth Shifts")
    plt.ylabel("Predicted Shifts")
    plt.title("Predicted vs Ground Truth Shifts")
    plt.show()


# Main Function
def main():
    # Paths
    model_path = "checkpoints/best_model_20250215-122212_cv0.pt"  # Replace with the path to your saved model
    model_type = "MPGNN"  # Replace with the type of model you want to load (e.g., 'MPGNN', 'SCHNET', 'GAT')

    # Device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load the Best Model
    model = load_best_model(model_path, model_type)
    model.to(device)

    # Step 2: Load the Organometallic Test Set
    test_loader = load_test_dataset()

    # Step 3: Make Predictions on the Test Set
    all_predictions, all_targets = make_predictions(model, test_loader, device)

    # Step 4: Evaluate the Model's Performance
    evaluate_model(all_targets, all_predictions)

    # Step 5: Visualize the Predictions (Optional)
    visualize_predictions(all_targets, all_predictions)


# Run the script
if __name__ == "__main__":
    main()
