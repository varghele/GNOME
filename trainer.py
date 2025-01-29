# trainer.py
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
#import wandb


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device(args.device)

        # Initialize optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc='Training'):
            batch = batch.to(self.device)

            # Forward pass
            shifts, _ = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.u,
                batch.batch
            )

            # Compute loss
            loss = torch.nn.functional.mse_loss(shifts, batch.shifts)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        for batch in val_loader:
            batch = batch.to(self.device)
            shifts, _ = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.u,
                batch.batch
            )
            loss = torch.nn.functional.mse_loss(shifts, batch.shifts)
            total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')

        # Initialize wandb
        wandb.init(
            project=f"{self.args.model_type}-training",
            config=vars(self.args)
        )

        for epoch in range(self.args.epochs):
            # Training
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Log metrics
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(self.args.save_dir, 'best_model.pt'))
