import subprocess
import time

# List of experiments (each experiment is a list of arguments to pass to main.py)
# List of experiments (each experiment is a list of arguments to pass to main.py)
experiments = [
    # Experiment 1: SchNet
    [
        "--model_type", "SCHNET",
        "--epochs", "300",
        "--activation", "leaky_relu",
        "--normalization", "layer_norm",
        "--num_interactions", "6",  # SchNet-specific argument
        "--num_filters", "128",     # SchNet-specific argument
        "--num_gaussians", "50",     # SchNet-specific argument
        "--cutoff", "10.0",         # SchNet-specific argument
        "--max_num_neighbors", "32" # SchNet-specific argument
    ],
    # Experiment 2: GAT
    [
        "--model_type", "GAT",
        "--epochs", "300",
        "--activation", "leaky_relu",
        "--normalization", "layer_norm",
        "--heads", "8"  # GAT-specific argument
    ],
    # Experiment 3: MPGNN
    [
        "--model_type", "MPGNN",
        "--epochs", "300",
        "--activation", "leaky_relu",
        "--normalization", "layer_norm",
        "--num_edge_mlp_layers", "3",  # MPGNN-specific argument
        "--num_node_mlp_layers", "3",   # MPGNN-specific argument
        "--num_global_mlp_layers", "3" # MPGNN-specific argument
    ]
]

experiments = [
    [
        "--model_type", "SCHNET",
        "--epochs", "300",
        "--activation", "leaky_relu",
        "--normalization", "layer_norm",
        "--num_interactions", "6",  # SchNet-specific argument
        "--num_filters", "128",  # SchNet-specific argument
        "--num_gaussians", "50",  # SchNet-specific argument
        "--cutoff", "10.0",  # SchNet-specific argument
        "--max_num_neighbors", "32"  # SchNet-specific argument
    ]
]


# Function to run a single experiment
def run_experiment(args):
    print(f"Starting experiment with args: {args}")
    start_time = time.time()

    # Run the experiment by calling main.py with the given arguments
    result = subprocess.run(["python", "main.py"] + args, capture_output=True, text=True)

    # Print the output and error (if any)
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")

    end_time = time.time()
    print(f"Experiment finished in {end_time - start_time:.2f} seconds\n")


# Run all experiments sequentially
for i, args in enumerate(experiments):
    print(f"Running experiment {i + 1}/{len(experiments)}")
    run_experiment(args)

print("All experiments completed!")
