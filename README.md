# GNOME
GNOME — Graph-based Neural Organometallic Magnetic (Shift) Estimator

## Downloading and Processing Data with `datasets.py`

This guide will walk you through the steps to download and process the NMR dataset using the `datasets.py` script.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11**
- **PyTorch** (with CUDA if available)
- **PyTorch Geometric**
- **RDKit** (for molecular data processing)
- **torch-scatter** (for data processing)

Either use the provided `environment.yml`: 

```bash
conda env create -f environment.yml
conda activate GNOME
```

Or run the following (you have to adjust for your CUDA version:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg -c pyg
pip install torch-scatter
conda install conda-forge::rdkit
conda install conda-forge::tensorboard
conda install anaconda::h5py 
```
If you have trouble installing `torch-scatter`, don't worry, there is a backup solution in place for the `MPGNN` model.
### Step 1: Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/varghele/GNOME.git
cd GNOME
```
### Step 2: Download the Raw Data
To download the raw NMR dataset, run the following command:
```bash
python dataset.py --data_dir data --download
```
This will download the `nmrshiftdb2withsignals.sd` file from SourceForge and save it in the `data/raw` directory.

### Step 3: Process the Data
Once the raw data is downloaded, you can process it into PyTorch Geometric format by running:
```bash
python dataset.py --data_dir data --process
```

This step takes roughly **5 minutes** and will:

1. **Load the `.sd` file using RDKit**: The script uses RDKit to parse the `.sd` file and extract molecular structures.
2. **Extract atom features, bond features, and NMR shifts**: For each molecule, the script extracts:
   - **Atom features**: Atomic number, degree, formal charge, number of hydrogens, aromaticity, and hybridization.
   - **Bond features**: Bond type, conjugation, ring membership, and bond length.
   - **NMR shifts**: 13C NMR shifts for each atom (if available).
3. **Create ghost bonds**: The script adds ghost bonds between atoms that are not connected by real bonds. These bonds are labeled with a bond type of `4` and include the distance between the atoms as a feature.
4. **Save the processed data**: The processed data is saved as `processed_data.pt` in the `data/processed` directory.

### Step 4: Verify the Dataset
To verify that the dataset was processed correctly, you can check the number of molecules and inspect the first molecule:
```bash
python dataset.py --data_dir data
```
This will print:

    The number of molecules in the dataset.
    The first molecule in the dataset (as a PyTorch Geometric Data object).

### Step 5: Run training
Take a look at args.py what arguments you can pass to the pipeline. Be aware however, that not all models share all arguments, as some are model specific.
```bash
python main.py [**kwargs]
```

### Step 6: Monitor with Tensorboard
During training, model performance is logged for tracking and hyperparameter tuning. Logging is available via tensorboard. In the main directory, run:
```bash
python tensorboard --logdir checkpoints
```
