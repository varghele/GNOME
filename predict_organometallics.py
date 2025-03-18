import torch
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from models import get_model
from models.mpgnn import MPGNN
from models.gnnml3 import GNNML3
from dataset import OrganometallicDataset, MoleculeDataset, FilteredDataset
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


# Step 2: Load the Test Set
def load_test_dataset():
    """
    Load the test dataset.
    """
    # Create the test dataset
    test_dataset = FilteredDataset()  # MoleculeDataset(root='data') #OrganometallicDataset()

    # Create the DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # Adjust batch size as needed
        shuffle=False,
        num_workers=4  # Adjust number of workers as needed
    )
    return test_loader, test_dataset


# Step 3: Make Predictions on the Test Set
def make_predictions(model, test_loader, device):
    all_predictions = []
    all_targets = []
    all_environment_types = []

    # Define environment types
    environment_types = ["non-carbon", "aliphatic", "alkoxy/amino", "aromatic", "olefinic", "carbonyl/carboxyl"]

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in test_loader:
            # Move batch to the appropriate device
            batch = batch.to(device)

            # Make predictions
            predictions = model(batch.x, batch.edge_index, batch.edge_attr)

            # Get the target values
            targets = batch.y.cpu().numpy()

            # Extract environment types from one-hot encoding
            for i in range(len(batch.x)):
                # Get the environment one-hot encoding (last 6 features)
                env_encoding = batch.x[i][-6:].cpu().numpy()
                if np.max(env_encoding) > 0:  # Check if any environment is encoded
                    env_index = np.argmax(env_encoding)
                    all_environment_types.append(environment_types[env_index])
                else:
                    all_environment_types.append("unknown")

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

    # Cut down environment types to match the length after masking
    # This is a simplified approach - in a real-world script,
    # you'd need to track which indices were masked
    all_environment_types = all_environment_types[:len(all_predictions)]

    return all_predictions, all_targets, all_environment_types


# Step 4: Evaluate the Model's Performance
def evaluate_model(all_targets, all_predictions):
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Calculate error metrics by range buckets
    ranges = [(0, 50), (50, 90), (90, 150), (150, 220)]
    range_names = ["aliphatic", "alkoxy/amino", "aromatic/olefinic", "carbonyl/carboxyl"]

    print("\nPerformance by Chemical Shift Range:")
    for (start, end), name in zip(ranges, range_names):
        mask = (all_targets >= start) & (all_targets <= end)
        if np.sum(mask) > 0:
            range_mae = mean_absolute_error(all_targets[mask], all_predictions[mask])
            range_rmse = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
            range_r2 = r2_score(all_targets[mask], all_predictions[mask])
            print(
                f"  {name} ({start}-{end} ppm): MAE = {range_mae:.4f}, RMSE = {range_rmse:.4f}, R² = {range_r2:.4f}, Count = {np.sum(mask)}")

    return mae, rmse, r2


# Function to create histograms using matplotlib
def create_histogram(ax, data, bins=50, color='blue', alpha=0.7, title=None, xlabel=None, ylabel=None):
    n, bins, patches = ax.hist(data, bins=bins, color=color, alpha=alpha, density=True)

    # Add KDE-like smooth curve
    try:
        from scipy.ndimage import gaussian_filter1d
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist, sigma=1)
        ax.plot(bin_centers, smoothed, color='black', linewidth=1.5)
    except ImportError:
        # Fallback if scipy is not available
        pass

    if title:
        ax.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(True, alpha=0.3)


# Step 5: Enhanced Visualization
def visualize_predictions_with_distributions(all_targets, all_predictions, all_environment_types=None):
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)

    # Create a scatter plot of predictions vs targets
    ax_scatter = fig.add_subplot(gs[0:2, 0:2])

    # Add color by environment if available
    if all_environment_types is not None and len(all_environment_types) == len(all_targets):
        unique_environments = sorted(list(set([env for env in all_environment_types
                                               if env != "non-carbon" and env != "unknown"])))

        # Create a colormap
        cmap = plt.cm.viridis
        colors = [cmap(i / len(unique_environments)) for i in range(len(unique_environments))]
        color_dict = dict(zip(unique_environments, colors))

        # Create scatter points by environment type
        for env in unique_environments:
            mask = np.array(all_environment_types) == env
            ax_scatter.scatter(all_targets[mask], all_predictions[mask],
                               alpha=0.5, s=10, color=color_dict[env], label=env)

        ax_scatter.legend(loc='upper left', title="Carbon Environment")
    else:
        # Simple scatter plot without colors
        ax_scatter.scatter(all_targets, all_predictions, alpha=0.5, s=10, color='blue')

    # Add diagonal line
    ax_scatter.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)],
                    color='red', linestyle='--')

    # Add horizontal and vertical lines for expected ranges
    ranges = [(0, 50), (50, 90), (90, 150), (150, 220)]
    range_colors = ['skyblue', 'lightgreen', 'salmon', 'plum']

    for (start, end), color in zip(ranges, range_colors):
        # Vertical lines for target ranges
        ax_scatter.axvspan(start, end, alpha=0.1, color=color)
        # Horizontal lines for prediction ranges
        ax_scatter.axhspan(start, end, alpha=0.1, color=color)

    ax_scatter.set_xlabel("Ground Truth Shifts (ppm)", fontsize=12)
    ax_scatter.set_ylabel("Predicted Shifts (ppm)", fontsize=12)
    ax_scatter.set_title("Predicted vs Ground Truth Shifts", fontsize=16)
    ax_scatter.grid(True, alpha=0.3)

    # Add text with performance metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)

    text_str = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax_scatter.text(0.05, 0.95, text_str, transform=ax_scatter.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

    # Create histograms for targets and predictions
    ax_hist_target = fig.add_subplot(gs[0, 2])
    ax_hist_prediction = fig.add_subplot(gs[1, 2])

    # Target histogram using matplotlib
    create_histogram(ax_hist_target, all_targets, bins=50, color='blue', alpha=0.7,
                     title="Distribution of Target Shifts", xlabel="Chemical Shift (ppm)", ylabel="Density")

    # Add vertical lines for expected ranges
    for (start, end), color in zip(ranges, range_colors):
        ax_hist_target.axvspan(start, end, alpha=0.2, color=color)

    # Prediction histogram using matplotlib
    create_histogram(ax_hist_prediction, all_predictions, bins=100, color='green', alpha=0.7,
                     title="Distribution of Predicted Shifts", xlabel="Chemical Shift (ppm)", ylabel="Density")

    # Add vertical lines for expected ranges
    for (start, end), color in zip(ranges, range_colors):
        ax_hist_prediction.axvspan(start, end, alpha=0.2, color=color)

    # Create a Q-Q plot to compare distributions
    ax_qq = fig.add_subplot(gs[2, 0])
    from scipy import stats
    stats.probplot(all_targets, dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q Plot of Target Values", fontsize=14)

    # Add a residuals plot
    ax_residuals = fig.add_subplot(gs[2, 1])
    residuals = all_predictions - all_targets
    ax_residuals.scatter(all_targets, residuals, alpha=0.5, s=10)
    ax_residuals.axhline(y=0, color='r', linestyle='--')
    ax_residuals.set_title("Residuals Plot", fontsize=14)
    ax_residuals.set_xlabel("Ground Truth Shifts (ppm)", fontsize=12)
    ax_residuals.set_ylabel("Residuals (Predicted - Ground Truth)", fontsize=12)
    ax_residuals.grid(True, alpha=0.3)

    # Add a histogram of residuals
    ax_res_hist = fig.add_subplot(gs[2, 2])
    create_histogram(ax_res_hist, residuals, bins=50, color='purple', alpha=0.7,
                     title="Distribution of Residuals", xlabel="Residual Value (ppm)", ylabel="Density")

    # Add text with residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    median_residual = np.median(residuals)

    text_str = f"Mean: {mean_residual:.4f}\nMedian: {median_residual:.4f}\nStd Dev: {std_residual:.4f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax_res_hist.text(0.05, 0.95, text_str, transform=ax_res_hist.transAxes, fontsize=12,
                     verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig("prediction_analysis.png", dpi=300)
    plt.show()

    # Create a separate figure to show error by shift value range
    plt.figure(figsize=(14, 7))

    # Define shift ranges
    ranges = [(0, 50), (50, 90), (90, 150), (150, 220)]
    range_names = ["aliphatic", "alkoxy/amino", "aromatic/olefinic", "carbonyl/carboxyl"]

    # Calculate MAE, RMSE for each range
    mae_by_range = []
    rmse_by_range = []
    counts_by_range = []

    for start, end in ranges:
        mask = (all_targets >= start) & (all_targets <= end)
        if np.sum(mask) > 0:
            mae_range = mean_absolute_error(all_targets[mask], all_predictions[mask])
            rmse_range = np.sqrt(mean_squared_error(all_targets[mask], all_predictions[mask]))
            mae_by_range.append(mae_range)
            rmse_by_range.append(rmse_range)
            counts_by_range.append(np.sum(mask))
        else:
            mae_by_range.append(0)
            rmse_by_range.append(0)
            counts_by_range.append(0)

    # Create bar plot
    x = np.arange(len(range_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, mae_by_range, width, label='MAE', color='cornflowerblue')
    bars2 = ax1.bar(x + width / 2, rmse_by_range, width, label='RMSE', color='lightcoral')

    # Plot count line
    ax2.plot(x, counts_by_range, 'go-', linewidth=2, markersize=8, label='Count')

    ax1.set_xlabel('Chemical Shift Range', fontsize=14)
    ax1.set_ylabel('Error (ppm)', fontsize=14)
    ax2.set_ylabel('Number of Data Points', fontsize=14, color='green')
    ax1.set_title('Model Performance by Chemical Shift Range', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(range_names)
    ax2.tick_params(axis='y', labelcolor='green')

    # Add count values as text
    for i, count in enumerate(counts_by_range):
        ax2.annotate(f'{count}',
                     xy=(i, count),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     color='green')

    # Add the values on top of the bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig("performance_by_range.png", dpi=300)
    plt.show()


# Main Function
def main():
    # Paths
    model_path = "checkpoints/best_model_20250318-142549_cv0.pt"  # Replace with the path to your saved model
    model_type = "MPGNN"  # Replace with the type of model

    # Device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load the Best Model
    print(f"Loading model from {model_path}...")
    model = load_best_model(model_path, model_type)
    model.to(device)

    # Step 2: Load the Test Set
    print("Loading test dataset...")
    test_loader, test_dataset = load_test_dataset()
    print(f"Test dataset contains {len(test_dataset)} molecules")

    # Step 3: Make Predictions on the Test Set
    print("Making predictions...")
    all_predictions, all_targets, all_environment_types = make_predictions(model, test_loader, device)
    print(f"Generated {len(all_predictions)} predictions for non-NaN targets")

    # Step 4: Evaluate the Model's Performance
    print("\nEvaluating model performance:")
    mae, rmse, r2 = evaluate_model(all_targets, all_predictions)

    # Step 5: Visualize the Predictions with Distributions
    print("\nGenerating visualizations...")
    visualize_predictions_with_distributions(all_targets, all_predictions, all_environment_types)

    print("\nAnalysis complete!")


# Run the script
if __name__ == "__main__":
    main()
