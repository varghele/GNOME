import torch
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem import rdchem
import numpy as np
from tqdm import tqdm
import mendeleev as me

# Temporary dictionaries to cache atomic properties
electronegativity_cache = {}
atomic_radius_cache = {}
electron_affinity_cache = {}
polarizability_cache = {}
ionization_energy_cache = {}
covalent_radius_cache = {}


def classify_carbon_environment(atom, mol):
    """Classify a carbon atom into structural categories based on its environment."""
    if atom.GetSymbol() != 'C':
        return "non-carbon"

    # Is it aromatic?
    if atom.GetIsAromatic():
        return "aromatic"

    # Check for carbonyl/carboxyl
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                   neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
            return "carbonyl/carboxyl"

    # Check for olefinic (C=C)
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'C' and mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                   neighbor.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
            return "olefinic"

    # Check for alkoxy/amino
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() in ['O', 'N']:
            return "alkoxy/amino"

    # If none of the above, it's aliphatic
    return "aliphatic"


# Function to create atom features
def get_atom_features(atom, mol=None):
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

    # Hybridization (one-hot encoded)
    hybridization = atom.GetHybridization()
    hybridization_types = [
        Chem.HybridizationType.SP,  # 0
        Chem.HybridizationType.SP2,  # 1
        Chem.HybridizationType.SP3,  # 2
        Chem.HybridizationType.SP3D,  # 3
        Chem.HybridizationType.SP3D2,  # 4
        Chem.HybridizationType.UNSPECIFIED  # 5
    ]
    hybridization_one_hot = torch.zeros(len(hybridization_types), dtype=torch.float)
    if hybridization in hybridization_types:
        hybridization_one_hot[hybridization_types.index(hybridization)] = 1.0

    # Additional features
    valence_electrons = atom.GetTotalValence()
    is_in_ring = int(atom.IsInRing())
    chirality = int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    radical_electrons = atom.GetNumRadicalElectrons()
    isotope = atom.GetIsotope() or 0  # Handle None case
    vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())

    # Check if we need to get properties from mendeleev
    if atomic_num not in electronegativity_cache:
        try:
            # Get all properties at once to avoid multiple element lookups
            element = me.element(atomic_num)

            # Cache electronegativity (Pauling scale)
            # Handle the case where electronegativity is a method that needs to be called
            try:
                en_value = element.electronegativity_pauling
                if callable(en_value):
                    en_value = 0.0  # Default if it's a method
                electronegativity_cache[atomic_num] = float(en_value) if en_value is not None else 0.0
            except (AttributeError, TypeError):
                electronegativity_cache[atomic_num] = 0.0

            # Cache additional atomic properties - ensuring all are float values
            try:
                atomic_radius_cache[atomic_num] = float(
                    element.atomic_radius) if element.atomic_radius is not None else 0.0
            except (AttributeError, TypeError):
                atomic_radius_cache[atomic_num] = 0.0

            try:
                electron_affinity_cache[atomic_num] = float(
                    element.electron_affinity) if element.electron_affinity is not None else 0.0
            except (AttributeError, TypeError):
                electron_affinity_cache[atomic_num] = 0.0

            try:
                polarizability_cache[atomic_num] = float(
                    element.dipole_polarizability) if element.dipole_polarizability is not None else 0.0
            except (AttributeError, TypeError):
                polarizability_cache[atomic_num] = 0.0

            try:
                # First ionization energy
                if hasattr(element, 'ionenergies') and element.ionenergies and 1 in element.ionenergies:
                    ionization_energy_cache[atomic_num] = float(element.ionenergies[1])
                else:
                    ionization_energy_cache[atomic_num] = 0.0
            except (AttributeError, TypeError):
                ionization_energy_cache[atomic_num] = 0.0

            try:
                covalent_radius_cache[atomic_num] = float(
                    element.covalent_radius) if element.covalent_radius is not None else 0.0
            except (AttributeError, TypeError):
                covalent_radius_cache[atomic_num] = 0.0
        except Exception:
            # If any error occurs, set all properties to 0.0
            electronegativity_cache[atomic_num] = 0.0
            atomic_radius_cache[atomic_num] = 0.0
            electron_affinity_cache[atomic_num] = 0.0
            polarizability_cache[atomic_num] = 0.0
            ionization_energy_cache[atomic_num] = 0.0
            covalent_radius_cache[atomic_num] = 0.0

    # Retrieve properties from cache
    electronegativity = electronegativity_cache.get(atomic_num, 0.0)
    atomic_radius = atomic_radius_cache.get(atomic_num, 0.0)
    electron_affinity = electron_affinity_cache.get(atomic_num, 0.0)
    polarizability = polarizability_cache.get(atomic_num, 0.0)
    ionization_energy = ionization_energy_cache.get(atomic_num, 0.0)
    covalent_radius = covalent_radius_cache.get(atomic_num, 0.0)

    # Add carbon environment classification as one-hot encoding
    if mol is not None:  # Make sure mol is provided
        environment = classify_carbon_environment(atom, mol)
        environment_types = ["non-carbon", "aliphatic", "alkoxy/amino", "aromatic", "olefinic", "carbonyl/carboxyl"]
        environment_one_hot = torch.zeros(len(environment_types), dtype=torch.float)
        if environment in environment_types:
            environment_one_hot[environment_types.index(environment)] = 1.0
    else:
        # Default to zeros if mol not provided
        environment_one_hot = torch.zeros(6, dtype=torch.float)  # 6 environment types

    # Combine features into a single vector
    atom_features = torch.cat([
        torch.tensor([atomic_num, degree, formal_charge, num_hydrogens, is_aromatic], dtype=torch.float),
        hybridization_one_hot,
        torch.tensor([
            valence_electrons,
            is_in_ring,
            chirality,
            radical_electrons,
            isotope,
            vdw_radius,
            electronegativity,
            atomic_radius,
            electron_affinity,
            polarizability,
            ionization_energy,
            covalent_radius
        ], dtype=torch.float),
        environment_one_hot
    ])

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

# Radial Basis Function (RBF) to adjust bond length
def radial_basis_function(bond_length, centers=np.linspace(0, 2, 20), gamma=10.0):
    """
    Apply a radial basis function to the bond length.
    :param bond_length: The bond length to transform.
    :param centers: Centers of the RBF (default: 20 centers between 0 and 2 Å).
    :param gamma: Width of the RBF (default: 10.0).
    :return: RBF-transformed bond length as a tensor.
    """
    return torch.tensor(np.exp(-gamma * (bond_length - centers) ** 2), dtype=torch.float)


# Function to create bond features
def get_bond_features(bond, mol):
    # Bond type (single, double, triple, aromatic)
    bond_type = bond.GetBondType()

    # Original bond type encoding (commented out)
    # bond_type_map = {
    #     rdchem.BondType.SINGLE: 0,
    #     rdchem.BondType.DOUBLE: 1,
    #     rdchem.BondType.TRIPLE: 2,
    #     rdchem.BondType.AROMATIC: 3,
    # }
    # bond_type_idx = bond_type_map.get(bond_type, 0)

    # One-hot encoding for bond type
    bond_type_one_hot = torch.zeros(5)  # 4 bond types: single, double, triple, aromatic, ghost
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_type_one_hot[0] = 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_type_one_hot[1] = 1
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_type_one_hot[2] = 1
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_type_one_hot[3] = 1
    else:
        bond_type_one_hot[4] = 1

    # Conjugation (whether the bond is conjugated)
    is_conjugated = int(bond.GetIsConjugated())

    # Ring membership (whether the bond is part of a ring)
    is_in_ring = int(bond.IsInRing())

    # Bond length
    bond_length = get_bond_length(bond, mol)

    # Adjusted bond length using RBF
    rbf_adjusted_bond_length = radial_basis_function(bond_length)

    # Combine features into a single vector
    bond_features = torch.cat([
        bond_type_one_hot,  # One-hot encoded bond type
        torch.tensor([is_conjugated, is_in_ring, bond_length], dtype=torch.float),  # Conjugation, ring membership, bond length
        rbf_adjusted_bond_length  # RBF-adjusted bond length
    ])

    return bond_features



def create_ghost_bond_features(distance):
    # One-hot encoding for bond types: [single, double, triple, aromatic, ghost]
    ghost_bond_one_hot = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)  # Ghost bond is the 5th type

    # Adjusted bond length using RBF
    rbf_adjusted_distance = radial_basis_function(distance)

    # Combine one-hot encoded bond type with other features (conjugation, ring membership, distance, RBF-adjusted distance)
    ghost_bond_features = torch.cat([
        ghost_bond_one_hot,  # One-hot encoded bond type (ghost bond)
        torch.tensor([0, 0, distance], dtype=torch.float),  # Other features (conjugation, ring membership, distance)
        rbf_adjusted_distance  # RBF-adjusted distance
    ])

    return ghost_bond_features



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

# Function to check if molecule is organometallic
def is_organometallic(mol):
    """
    Check if a molecule is organometallic by looking for metal atoms.
    """
    # List of atomic numbers for common metals
    metals = set(range(3, 5)) | set(range(11, 14)) | set(range(19, 33)) | set(range(37, 52)) | set(range(55, 85)) | set(
        range(87, 104))

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in metals:
            return True
    return False


def create_molecule_data(mol):
    if mol is None:
        return None

    # Extract 13C NMR shifts
    nmr_shifts = extract_nmr_shifts(mol)

    # Node features: atom features + NMR shift
    node_features = []
    for atom in mol.GetAtoms():
        atom_features = get_atom_features(atom, mol)

        # Get the NMR shift for the atom (if available)
        #nmr_shift = nmr_shifts.get(atom.GetIdx(), float('nan'))  # Assign NaN if no shift is available

        # Combine atom features and NMR shift into a single feature vector
        #node_feature = torch.cat([atom_features, torch.tensor([nmr_shift])])
        node_feature = atom_features
        node_features.append(node_feature)

    # Edge features: bond features
    edge_indices = []
    edge_attrs = []
    num_atoms = mol.GetNumAtoms()

    # Add real bonds
    for bond in mol.GetBonds():  # Add tqdm for bonds
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        bond_features = get_bond_features(bond, mol)

        # Add edge (both directions for undirected graph)
        edge_indices.append([start_atom, end_atom])
        edge_indices.append([end_atom, start_atom])
        edge_attrs.append(bond_features)
        edge_attrs.append(bond_features)

    """# Add ghost bonds between all pairs of atoms that are not connected by real bonds
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if not mol.GetBondBetweenAtoms(i, j):
                # Calculate distance between atoms i and j
                conf = mol.GetConformer()
                pos_i = np.array(conf.GetAtomPosition(i))
                pos_j = np.array(conf.GetAtomPosition(j))
                distance = np.linalg.norm(pos_i - pos_j)

                # Create ghost bond features (bond type 4 for ghost bonds)
                #ghost_bond_features = torch.tensor([4, 0, 0, distance], dtype=torch.float) # TODO: make dynamic
                ghost_bond_features = create_ghost_bond_features(distance)

                # Add ghost bond (both directions for undirected graph)
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                edge_attrs.append(ghost_bond_features)
                edge_attrs.append(ghost_bond_features)"""

    # Convert lists to tensors
    x = torch.stack(node_features)  # Node feature matrix

    # Handle empty edges
    if len(edge_attrs) == 0:
        # If there are no edges, create empty tensors for edge_index and edge_attr
        edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge connectivity matrix
        edge_attr = torch.empty((0, 7), dtype=torch.float)  # Empty edge feature matrix #TODO: Make dynamic
    else:
        # If there are edges, stack them into tensors
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # Edge connectivity matrix
        edge_attr = torch.stack(edge_attrs)  # Edge feature matrix

    # Create the target tensor (y) from the NMR shifts
    # Here, we assume that the NMR shift for each atom is the target
    y = torch.tensor([nmr_shifts.get(atom.GetIdx(), float('nan')) for atom in mol.GetAtoms()], dtype=torch.float)


    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data