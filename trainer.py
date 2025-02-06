import os
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging
import csv  # For CSV logging
import datetime


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device(args.device)

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
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

    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()
        total_loss = 0

        for batch in test_loader:
            batch = batch.to(self.device)
            shifts, _ = self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )
            loss = torch.nn.functional.mse_loss(shifts, torch.unsqueeze(batch.y, dim=-1))
            total_loss += loss.item()

        return total_loss / len(test_loader)



    def train(self, train_loader, val_loader, test_loader, cv_split=None):
        """
        Trains the model and logs detailed training results using TensorBoard.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            cv_split (int, optional): Cross-validation split index (default: None).
        """
        best_val_loss = float('inf')
        last_train_loss = float('inf')
        test_loss = float('inf')

        # Create unique log directory with timestamp and CV split info
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        if cv_split is not None:
            log_dir = os.path.join(self.args.save_dir, "logs", f"run_{current_time}",f"cv{cv_split}")
        else:
            log_dir = os.path.join(self.args.save_dir, "logs", f"run_{current_time}")

        # Check if the directory exists, and if not, create it
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=log_dir)

        # Create a unique CSV file for this CV split
        csv_file = os.path.join(log_dir, 'training_logs.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])

        # Log all arguments supplied to the training function
        model_params = {}

        # Add all arguments from self.args to the model_params dictionary
        for arg in vars(self.args):
            value = getattr(self.args, arg)
            # Ensure the value is one of the allowed types (int, float, str, bool, torch.Tensor)
            if isinstance(value, (int, float, str, bool, torch.Tensor)):
                model_params[arg] = value
            else:
                # Convert other types to string (optional, or you can skip them)
                model_params[arg] = str(value)

        # Log hyperparameters as scalar values in the same folder
        for param_name, param_value in model_params.items():
            if isinstance(param_value, (int, float)):
                writer.add_scalar(f'hyperparameters/{param_name}', param_value, 0)
            elif isinstance(param_value, str):
                writer.add_text(f'hyperparameters/{param_name}', str(param_value), 0)

        # Log model parameters to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(['=== Run Configuration ==='])
            for key, value in model_params.items():
                writer_csv.writerow([key, value])
            writer_csv.writerow([])  # Add a blank row for separation

        for epoch in tqdm(range(self.args.epochs), desc="Training", unit="epoch"):
            # Training
            train_loss = self.train_epoch(train_loader)
            last_train_loss = train_loss

            # Validation
            val_loss = self.validate(val_loader)

            # Test set
            test_loss = self.test(test_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Log metrics to TensorBoard
            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Test Loss', test_loss, epoch)
            writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Log metrics to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer_csv = csv.writer(file)
                writer_csv.writerow([epoch, train_loss, val_loss, self.optimizer.param_groups[0]['lr']])

            # Save best model with enhanced information
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Create unique model filename using the same timestamp and CV split info
                model_filename = f'best_model_{current_time}'
                if cv_split is not None:
                    model_filename += f'_cv{cv_split}'
                model_filename += '.pt'

                # Save model with all relevant information
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'model_params': model_params,
                    'cv_split': cv_split,
                    'timestamp': current_time,
                    'log_dir': log_dir  # Save the corresponding log directory for reference
                }, os.path.join(self.args.save_dir, model_filename))

        # Log final metrics for this CV split
        writer.add_hparams(
            hparam_dict=model_params,  # Dictionary of hyperparameters
            metric_dict={
                'Last Training Loss': last_train_loss,
                'Last Test Loss': test_loss,
                'Best Validation Loss': best_val_loss,  # Log the best validation loss
            },
        )

        # Close TensorBoard writer
        writer.close()

