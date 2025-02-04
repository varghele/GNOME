import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging
import csv  # For CSV logging


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

        # Initialize TensorBoard writer (optional)
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

        # Initialize CSV logging (optional)
        self.csv_file = os.path.join(args.save_dir, 'training_logs.csv')
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc='Processing Step'):
            batch = batch.to(self.device)

            # Forward pass
            shifts, _ = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            # Compute loss
            loss = torch.nn.functional.mse_loss(shifts, torch.unsqueeze(batch.y,dim=-1))

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
                batch.batch
            )
            loss = torch.nn.functional.mse_loss(shifts, torch.unsqueeze(batch.y,dim=-1))
            total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, cv_split=None):
        """
        Trains the model and logs detailed training results, including all supplied arguments.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            cv_split (int, optional): Cross-validation split index (default: None).
        """
        best_val_loss = float('inf')

        # Log all arguments supplied to the training function
        model_params = {
            'cv_split': cv_split  # Cross-validation split index
        }

        # Add all arguments from self.args to the model_params dictionary
        for arg in vars(self.args):
            model_params[arg] = getattr(self.args, arg)

        # Log model parameters to CSV
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Model Parameters'])
            writer.writerow(['Parameter', 'Value'])
            for key, value in model_params.items():
                writer.writerow([key, value])
            writer.writerow([])  # Add a blank row for separation

        for epoch in tqdm(range(self.args.epochs), desc="Training", unit="epoch"):
            # Training
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Log metrics to TensorBoard (optional)
            self.writer.add_scalar('Train Loss', train_loss, epoch)
            self.writer.add_scalar('Validation Loss', val_loss, epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Log metrics to CSV
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr']])

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'model_params': model_params,  # Save all model parameters with the best model
                    'cv_split': cv_split  # Save cross-validation split index
                }, os.path.join(self.args.save_dir, 'best_model.pt'))

        # Close TensorBoard writer
        self.writer.close()

