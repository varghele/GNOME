import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem import rdchem
import numpy as np
from tqdm import tqdm


# Function to create atom features
def get_atom_features(atom):
    # Atomic number
    atomic_num = atom.GetAtomicNum()

    # Degree (number of bonds)
    degree = atom.GetDegree()

    # Formal charge
    formal_charge = atom.GetFormalCharge()

    # Number of implicit hydrogens
    num_hydrogens = atom.GetTotalNumHs()

    # Aromaticity
    is_aromatic = int(atom.GetIsAromatic())

    # Hybridization (sp, sp2, sp3, etc.)
    hybridization = atom.GetHybridization()
    hybridization_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
        rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4,
        rdchem.HybridizationType.UNSPECIFIED: 5,
    }
    hybridization_idx = hybridization_map.get(hybridization, 5)

    # Combine features into a single vector
    atom_features = torch.tensor([
        atomic_num, degree, formal_charge, num_hydrogens, is_aromatic, hybridization_idx
    ], dtype=torch.float)

    return atom_features


# Function to calculate bond length
def get_bond_length(bond, mol):
    conf = mol.GetConformer()
    start_atom_idx = bond.GetBeginAtomIdx()
    end_atom_idx = bond.GetEndAtomIdx()
    start_pos = np.array(conf.GetAtomPosition(start_atom_idx))
    end_pos = np.array(conf.GetAtomPosition(end_atom_idx))
    bond_length = np.linalg.norm(start_pos - end_pos)
    return bond_length


# Function to create bond features
def get_bond_features(bond, mol):
    # Bond type (single, double, triple, aromatic)
    bond_type = bond.GetBondType()
    bond_type_map = {
        rdchem.BondType.SINGLE: 0,
        rdchem.BondType.DOUBLE: 1,
        rdchem.BondType.TRIPLE: 2,
        rdchem.BondType.AROMATIC: 3,
    }
    bond_type_idx = bond_type_map.get(bond_type, 0)

    # Conjugation (whether the bond is conjugated)
    is_conjugated = int(bond.GetIsConjugated())

    # Ring membership (whether the bond is part of a ring)
    is_in_ring = int(bond.IsInRing())

    # Bond length
    bond_length = get_bond_length(bond, mol)

    # Combine features into a single vector
    bond_features = torch.tensor([
        bond_type_idx, is_conjugated, is_in_ring, bond_length
    ], dtype=torch.float)

    return bond_features


# Function to extract 13C NMR shifts from the molecule properties
def extract_nmr_shifts(mol):
    if not mol.HasProp('Spectrum 13C 0'):
        return {}

    # Parse the 13C NMR shifts
    nmr_shift_data = mol.GetProp('Spectrum 13C 0')
    nmr_shifts = {}
    for shift_entry in nmr_shift_data.split('|'):
        if not shift_entry:
            continue
        shift, _, atom_idx = shift_entry.split(';')
        atom_idx = int(atom_idx)
        nmr_shifts[atom_idx] = float(shift)
    return nmr_shifts


def create_molecule_data(mol):
    if mol is None:
        return None

    # Extract 13C NMR shifts
    nmr_shifts = extract_nmr_shifts(mol)

    # Node features: atom features + NMR shift
    node_features = []
    for atom in tqdm(mol.GetAtoms(), desc="Processing atoms", leave=False):  # Add tqdm for atoms
        atom_features = get_atom_features(atom)

        # Get the NMR shift for the atom (if available)
        nmr_shift = nmr_shifts.get(atom.GetIdx(), 0.0)  # Default to 0.0 if no shift is available

        # Combine atom features and NMR shift into a single feature vector
        node_feature = torch.cat([atom_features, torch.tensor([nmr_shift])])
        node_features.append(node_feature)

    # Edge features: bond features
    edge_indices = []
    edge_attrs = []
    num_atoms = mol.GetNumAtoms()

    # Add real bonds
    for bond in tqdm(mol.GetBonds(), desc="Processing bonds", leave=False):  # Add tqdm for bonds
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_features = get_bond_features(bond, mol)

        # Add edge (both directions for undirected graph)
        edge_indices.append([start_atom, end_atom])
        edge_indices.append([end_atom, start_atom])
        edge_attrs.append(bond_features)
        edge_attrs.append(bond_features)

    # Add ghost bonds between all pairs of atoms that are not connected by real bonds
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if not mol.GetBondBetweenAtoms(i, j):
                # Calculate distance between atoms i and j
                conf = mol.GetConformer()
                pos_i = np.array(conf.GetAtomPosition(i))
                pos_j = np.array(conf.GetAtomPosition(j))
                distance = np.linalg.norm(pos_i - pos_j)

                # Create ghost bond features (bond type 4 for ghost bonds)
                ghost_bond_features = torch.tensor([4, 0, 0, distance], dtype=torch.float)

                # Add ghost bond (both directions for undirected graph)
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                edge_attrs.append(ghost_bond_features)
                edge_attrs.append(ghost_bond_features)

    # Convert lists to tensors
    x = torch.stack(node_features)  # Node feature matrix

    # Handle empty edges
    if len(edge_attrs) == 0:
        # If there are no edges, create empty tensors for edge_index and edge_attr
        edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge connectivity matrix
        edge_attr = torch.empty((0, 4), dtype=torch.float)  # Empty edge feature matrix
    else:
        # If there are edges, stack them into tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # Edge connectivity matrix
        edge_attr = torch.stack(edge_attrs)  # Edge feature matrix

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data