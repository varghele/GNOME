import os
import torch
import requests
import argparse
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from utils.dlutils import create_molecule_data
from tqdm import tqdm

class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    @property
    def raw_file_names(self):
        return ['nmrshiftdb2withsignals.sd']  # Raw file name (can be modified)

    @property
    def processed_file_names(self):
        return ['processed_data.pt']  # Processed file name

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

        data_list = []
        for mol in tqdm(supplier, desc="Processing molecules", unit="mol"):
            if mol is None:
                continue

            # Create molecule data using the create_molecule_data function
            data = create_molecule_data(mol)
            if data is not None:
                data_list.append(data)

        # Save the processed data
        torch.save(data_list, os.path.join(self.processed_dir, self.processed_file_names[0]))
        print(f"Processed data saved to: {os.path.join(self.processed_dir, self.processed_file_names[0])}")

    def len(self):
        return len(torch.load(os.path.join(self.processed_dir, self.processed_file_names[0])))

    def get(self, idx):
        data_list = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]))
        return data_list[idx]


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


if __name__ == "__main__":
    main()
