import os
import torch
import requests
import argparse
from torch_geometric.data import Dataset, Data


class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    @property
    def raw_file_names(self):
        return ['nmrshiftdb2.sd']  # Raw file name (can be modified)

    @property
    def processed_file_names(self):
        return ['processed_data.pt']  # Processed file name

    def download(self):
        # Ensure the raw data directory exists
        os.makedirs(self.raw_dir, exist_ok=True)

        # Download the nmrshiftdb2.sd file from SourceForge
        project_name = "nmrshiftdb2"
        file_name = "nmrshiftdb2.sd"  # Replace with the desired file name
        save_path = os.path.join(self.raw_dir, file_name)

        # Construct the download URL
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
        # This is a placeholder - implement according to your data format
        raw_data_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        # Load the raw data (assuming it's a list of dictionaries)
        # You need to implement this based on the actual format of your data
        raw_data = torch.load(raw_data_path)  # Placeholder for loading raw data

        data_list = []
        for item in raw_data:
            data = Data(
                x=item['node_features'],  # Replace with actual node features
                edge_index=item['edge_index'],  # Replace with actual edge indices
                edge_attr=item['edge_features'],  # Replace with actual edge features
                u=item['global_features'],  # Replace with actual global features
                shifts=item['shifts']  # Replace with actual NMR shifts
            )
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
