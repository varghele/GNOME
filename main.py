import os
import torch
import random
from torch_geometric.loader import DataLoader
from args import get_args
from models import get_model
from trainer import Trainer
from dataset import get_dataset


def k_fold_split(dataset, k, test_size=0.1, seed=42):
    """
    Manually splits the dataset into k folds for cross-validation, including a test set.
    If k is 1, performs a standard 80-15-5 split (train, validation, test).

    Args:
        dataset (list): The full dataset to split.
        k (int): Number of folds. If k is 1, performs a standard 80-15-5 split.
        test_size (float): Proportion of the dataset to include in the test split (default: 0.1).
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of k tuples, where each tuple contains (train_indices, val_indices, test_indices).
              If k is 1, returns a single tuple with the 80-15-5 split.
    """
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)  # Shuffle the indices to ensure randomness

    # If k is 1, perform a standard 80-15-5 split
    if k == 1:
        train_size = int(len(dataset) * 0.8)
        val_size = int(len(dataset) * 0.15)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return [(train_indices, val_indices, test_indices)]

    # Otherwise, perform k-fold cross-validation with a test set
    test_split = int(len(dataset) * test_size)
    test_indices = indices[:test_split]
    train_val_indices = indices[test_split:]

    # Now split the train_val_indices into k folds
    fold_size = len(train_val_indices) // k
    folds = []

    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else len(train_val_indices)
        val_indices = train_val_indices[val_start:val_end]
        train_indices = train_val_indices[:val_start] + train_val_indices[val_end:]
        folds.append((train_indices, val_indices, test_indices))

    return folds


def main():
    # Get arguments
    args = get_args()

    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    if args.mode == 'inference':
        os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Training on:{device}")

    # Initialize model
    model = get_model(args.model_type, args)
    model = model.to(device)

    if args.mode == 'train':
        # Get the dataset (full or debug)
        if args.debug == "True":
            print(f"Debug: {args.debug}")
            dataset = get_dataset(args.data_dir, split='debug', debug_fraction=args.debug_fraction)
        else:
            print(f"Full")
            dataset = get_dataset(args.data_dir, split='full')

        # Perform k-fold cross-validation
        folds = k_fold_split(dataset, args.num_splits, seed=args.seed)

        # Loop through each fold
        for fold, (train_idx, val_idx, test_idx) in enumerate(folds):
            print(f"Fold {fold + 1}/{args.num_splits}")

            # Split the dataset into train and validation sets
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            test_dataset = torch.utils.data.Subset(dataset, test_idx)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,  # Enable pinned memory for faster GPU transfer
                prefetch_factor = 2,  # Prefetch batches asynchronously
                persistent_workers=True # Keeps workers alive to speed up loading
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )

            # Initialize trainer
            trainer = Trainer(model, args)

            # Train model on this fold
            trainer.train(train_loader, val_loader, test_loader, fold)

    else:  # Inference mode
        if not args.model_path:
            raise ValueError("Model path must be provided for inference mode")

        # Load model weights
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Get test dataset
        test_dataset = get_dataset(args.data_dir, split='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # Run inference
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                shifts, _ = model(batch.x, batch.edge_index,
                                  batch.edge_attr, batch.u, batch.batch)
                predictions.append(shifts.cpu())

        # Save predictions
        predictions = torch.cat(predictions, dim=0)
        torch.save(predictions, os.path.join(args.output_dir, 'predictions.pt'))


if __name__ == "__main__":
    main()
