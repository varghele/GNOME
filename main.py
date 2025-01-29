# main.py
import os
import torch
from torch_geometric.loader import DataLoader
#import wandb

from args import get_args
from models import get_model
from trainer import Trainer
from dataset import get_dataset


def main():
    # Get arguments
    args = get_args()

    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    if args.mode == 'inference':
        os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)

    # Initialize model
    model = get_model(args.model_type, args)
    model = model.to(device)

    if args.mode == 'train':
        # Get datasets
        train_dataset = get_dataset(args.data_dir, split='train')
        val_dataset = get_dataset(args.data_dir, split='val')

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # Initialize trainer
        trainer = Trainer(model, args)

        # Train model
        trainer.train(train_loader, val_loader)

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