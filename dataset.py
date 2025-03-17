import os
import torch
import requests
import argparse
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from utils.dlutils import create_molecule_data, is_organometallic
from tqdm import tqdm
import h5py  # For HDF5 storage

import warnings

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

class MoleculeDataset(Dataset):
    def __init__(self, root, split='full', transform=None):
        super().__init__(root, transform)
        self.split = split  # 'full', 'train', 'val', 'test'

    @property
    def raw_file_names(self):
        return ['nmrshiftdb2withsignals.sd']  # Raw file name (can be modified)

    @property
    def processed_file_names(self):
        return ['processed_data.h5']  # Processed file name in HDF5 format

    def download(self):
        # Ensure the raw data directory exists
        os.makedirs(self.raw_dir, exist_ok=True)

        # Check if the file already exists
        file_name = self.raw_file_names[0]
        save_path = os.path.join(self.raw_dir, file_name)

        if os.path.exists(save_path):
            print(f"File already exists: {save_path}. Skipping download.")
            return

        # Download the nmrshiftdb2.sd file from SourceForge
        project_name = "nmrshiftdb2"
        url = f"https://sourceforge.net/projects/{project_name}/files/{file_name}/download"

        # Send a GET request to download the file
        response = requests.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the file to the specified path
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"File downloaded successfully: {save_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

    def process(self):
        # Process raw data into PyG Data objects
        raw_data_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        # Load the .sd file using RDKit
        supplier = Chem.SDMolSupplier(raw_data_path)

        # Create an HDF5 file to store the processed data
        h5_file_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        organometallic_h5_file_path = os.path.join(self.processed_dir, "organometallic_" + self.processed_file_names[0])
        with h5py.File(h5_file_path, 'w') as h5_file, h5py.File(organometallic_h5_file_path,
                                                                'w') as organometallic_h5_file:
            # Counter for valid molecules
            valid_molecule_count = 0
            organometallic_molecule_count = 0

            for idx, mol in enumerate(tqdm(supplier, desc="Processing molecules", unit="mol")):
                # Skip invalid molecules
                if mol is None:
                    print(f"Skipping molecule at index {idx} because it is invalid.")
                    continue

                # Skip molecules with no atoms or bonds
                if mol.GetNumAtoms() == 0 or mol.GetNumBonds() == 0:
                    print(f"Skipping molecule at index {idx} because it has no atoms or bonds.")
                    continue

                # Sanitize the molecule
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    print(f"Skipping molecule at index {idx} because it failed sanitization: {e}")
                    continue

                # Create molecule data using the create_molecule_data function
                data = create_molecule_data(mol)
                if data is None:
                    print(f"Skipping molecule at index {idx} because create_molecule_data returned None.")
                    continue

                # Check if all values in data.y are NaN (i.e., the molecule has no NMR shifts)
                if torch.isnan(data.y).all():
                    print(f"Skipping molecule at index {idx} because all NMR shifts are NaN.")
                    continue

                # Check if the molecule is organometallic
                if is_organometallic(mol):
                    # Save organometallic molecule in the organometallic HDF5 file
                    group = organometallic_h5_file.create_group(f'molecule_{organometallic_molecule_count}')
                    organometallic_molecule_count += 1
                else:
                    # Save normal molecule in the normal HDF5 file
                    group = h5_file.create_group(f'molecule_{valid_molecule_count}')
                    valid_molecule_count += 1

                # Save each molecule as a group in the HDF5 file
                group.create_dataset('x', data=data.x.numpy())
                group.create_dataset('edge_index', data=data.edge_index.numpy())
                group.create_dataset('edge_attr', data=data.edge_attr.numpy())
                group.create_dataset('y', data=data.y.numpy())

        print(f"Processed data saved to: {h5_file_path}")
        print(f"Organometallic processed data saved to: {organometallic_h5_file_path}")

    def len(self):
        # Get the number of molecules in the HDF5 file
        with h5py.File(os.path.join(self.processed_dir, self.processed_file_names[0]), 'r') as h5_file:
            return len(h5_file.keys())

    def get(self, idx):
        # Load a molecule from the HDF5 file
        with h5py.File(os.path.join(self.processed_dir, self.processed_file_names[0]), 'r') as h5_file:
            # Takes care of corrupted/missing files
            if f'molecule_{idx}' not in h5_file:
                return None
            group = h5_file[f'molecule_{idx}']

            # Debug: Print the keys and attributes of the group
            # print(f"Group 'molecule_{idx}' contains the following keys: {list(group.keys())}")
            # print(f"Attributes of group 'molecule_{idx}': {dict(group.attrs)}")

            # Load the datasets
            x = torch.tensor(group['x'][:])
            edge_index = torch.tensor(group['edge_index'][:])
            edge_attr = torch.tensor(group['edge_attr'][:])

            # Check if 'y' exists in the group
            if 'y' in group:
                y = torch.tensor(group['y'][:])
            else:
                # If 'y' is missing, assign a placeholder (e.g., zeros)
                y = torch.zeros(1)  # Placeholder
                print(f"Warning: 'y' is missing in group 'molecule_{idx}'. Using placeholder.")

            # Create a PyG Data object
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class OrganometallicDataset(Dataset):
    def __init__(self):
        """
        Initialize the OrganometallicDataset class with internal properties.
        """
        super(OrganometallicDataset, self).__init__()
        self._processed_dir = "data/processed/"
        self._processed_file_name = "organometallic_processed_data.h5"

        # Open the HDF5 file to get the number of molecules
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            self.molecule_keys = list(h5_file.keys())

    @property
    def processed_dir(self) -> str:
        """
        Get the processed directory.
        """
        return self._processed_dir

    @property
    def processed_file_name(self) -> str:
        """
        Get the processed file name.
        """
        return self._processed_file_name

    @property
    def h5_file_path(self) -> str:
        """
        Get the full path to the HDF5 file.
        """
        return os.path.join(self.processed_dir, self.processed_file_name)

    def len(self):
        """
        Returns the number of molecules in the dataset.
        """
        return len(self.molecule_keys)

    def get(self, idx):
        """
        Load a molecule from the HDF5 file and convert it into a PyG Data object.

        Args:
            idx (int): Index of the molecule to retrieve.

        Returns:
            data (torch_geometric.data.Data): A PyG Data object.
        """
        # Load the molecule from the HDF5 file
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            # Check if the molecule exists in the HDF5 file
            molecule_key = f'molecule_{idx}'
            if molecule_key not in h5_file:
                print(f"Warning: Molecule {molecule_key} not found in the HDF5 file.")
                return None

            group = h5_file[molecule_key]

            # Load the datasets
            x = torch.tensor(group['x'][:], dtype=torch.float)  # Node features
            edge_index = torch.tensor(group['edge_index'][:], dtype=torch.long)  # Edge indices
            edge_attr = torch.tensor(group['edge_attr'][:], dtype=torch.float)  # Edge features

            # Check if 'y' exists in the group
            if 'y' in group:
                y = torch.tensor(group['y'][:], dtype=torch.float)  # Target (e.g., chemical shifts)
            else:
                # If 'y' is missing, assign a placeholder (e.g., zeros)
                y = torch.zeros(1, dtype=torch.float)  # Placeholder
                print(f"Warning: 'y' is missing in group {molecule_key}. Using placeholder.")

            # Create a PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        return data


def get_dataset(data_dir, split='full', debug_fraction=0.05):
    """
    Loads the dataset and returns the appropriate split for cross-validation.

    Args:
        data_dir (str): Root directory where the dataset is stored.
        split (str): The split to return ('full', 'train', 'val', 'test', 'debug').
        debug_fraction (float): Fraction of the dataset to load for the 'debug' split (default: 0.1).

    Returns:
        Dataset: The dataset split.
    """
    # Load the full dataset (already filtered for None values)
    full_dataset = MoleculeDataset(root=data_dir, split='full')

    if split == 'full':
        return full_dataset

    # Define train/val/test splits (e.g., 80/10/10)
    num_samples = len(full_dataset)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size

    # Split the dataset
    indices = torch.randperm(num_samples).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Return the appropriate split
    if split == 'train':
        return torch.utils.data.Subset(full_dataset, train_indices)
    elif split == 'val':
        return torch.utils.data.Subset(full_dataset, val_indices)
    elif split == 'test':
        return torch.utils.data.Subset(full_dataset, test_indices)
    elif split == 'debug':
        # Load a small fraction of the dataset for debugging
        debug_size = int(debug_fraction * num_samples)
        debug_indices = indices[:debug_size]
        return torch.utils.data.Subset(full_dataset, debug_indices)
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'full', 'train', 'val', 'test', or 'debug'.")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Download and process NMR data from nmrshiftdb2.")
    parser.add_argument('--data_dir', type=str, default='data', help='Root directory for the dataset')
    parser.add_argument('--download', action='store_true', help='Download the raw data')
    parser.add_argument('--process', action='store_true', help='Process the raw data into PyG format')
    args = parser.parse_args()

    # Create the dataset
    dataset = MoleculeDataset(root=args.data_dir)

    # Download the data if requested
    if args.download:
        dataset.download()

    # Process the data if requested
    if args.process:
        dataset.process()

    # Print dataset information
    if not args.download and not args.process:
        print(f"Number of molecules in the dataset: {len(dataset)}")
        print(f"First molecule in the dataset: {dataset[0]}")
        data_0 = dataset[0]
        print(data_0.y)


if __name__ == "__main__":
    main()