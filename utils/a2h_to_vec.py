import torch
import numpy as np
import pickle
import logging
from pathlib import Path
from rdkit import RDLogger
import torch.nn.functional as F
from torch_geometric.data import Data

# Configure logging
logger = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')

def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two 3D coordinates."""
    return np.sqrt(np.sum((coord1 - coord2) ** 2))

def get_amino_acid_one_hot(res_name: str) -> np.ndarray:
    """Convert amino acid name to one-hot encoding."""
    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 
                   'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
                   'TYR', 'VAL']
    one_hot = np.zeros(len(amino_acids))
    if res_name in amino_acids:
        one_hot[amino_acids.index(res_name)] = 1
    return one_hot

def normalize_vector(vector, mean=None, std=None):
    """Normalize a vector using mean and standard deviation."""
    if mean is None:
        mean = np.mean(vector, axis=0)
    if std is None:
        std = np.std(vector, axis=0)
    return (vector - mean) / (std + 1e-8)

def normalize_distance(distances, max_dist=None):
    """Normalize distances to [0, 1] range."""
    if max_dist is None:
        max_dist = np.max(distances)
    return distances / (max_dist + 1e-8)

def read_a2h(file_path: str, all_atom: bool = False, pad: bool = True) -> Data:
    """
    Read and process A2H (Apo to Holo) protein structure data.
    
    Args:
        file_path (str): Path to the pickle file containing A2H data
        all_atom (bool): If True, process all atoms; if False, process only CA atoms
        pad (bool): If True, pad the output tensor to max_len
        
    Returns:
        Data: PyTorch Geometric Data object containing the processed tensor
    """
    # Set maximum length based on atom selection
    max_len = 1370 if all_atom else 136
    
    # Load data from pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    apo_structures = data['APO']
    holo_structures = data['HOLO']
    
    # Process each atom pair and calculate features
    node_features = []
    atom_info = []
    
    for apo, holo in zip(apo_structures, holo_structures):
        # Extract atom information
        a_atom_name, a_res_name, a_res_seq, a_coord = (
            apo['atom_name'], apo['res_name'], apo['res_seq'], apo['coord']
        )
        h_atom_name, h_res_name, h_res_seq, h_coord = (
            holo['atom_name'], holo['res_name'], holo['res_seq'], holo['coord']
        )
        
        # Skip non-CA atoms if all_atom is False
        if not all_atom and a_atom_name != 'CA':
            continue
            
        # Validate atom correspondence
        if (a_atom_name, a_res_name, a_res_seq) != (h_atom_name, h_res_name, h_res_seq):
            logger.warning(
                f"Atom mismatch detected:\n"
                f"Apo: {a_atom_name} {a_res_name} {a_res_seq}\n"
                f"Holo: {h_atom_name} {h_res_name} {h_res_seq}"
            )
            continue
        
        # Calculate features for each node
        displacement = h_coord - a_coord  # Displacement vector
        distance = np.linalg.norm(displacement)  # Magnitude of displacement
        
        # Get amino acid one-hot encoding
        aa_one_hot = get_amino_acid_one_hot(a_res_name)
        
        # Create node feature vector
        node_feature = np.concatenate([
            displacement,  # 3D displacement vector
            [distance],    # Scalar distance
            aa_one_hot    # Amino acid one-hot encoding (20 dimensions)
        ])
                
        node_features.append(node_feature)
        atom_info.append({
            'apo_coord': a_coord,
            'holo_coord': h_coord,
            'res_seq': a_res_seq,
            'res_name': a_res_name,
            'atom_name': a_atom_name
        })
    
    # Convert to tensor
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Create graph edges based on multiple criteria
    edge_index = []
    edge_attr = []
    
    for i in range(len(atom_info)):
        for j in range(i + 1, len(atom_info)):
            # 1. Sequential connection
            is_sequential = abs(atom_info[i]['res_seq'] - atom_info[j]['res_seq']) == 1
            
            # 2. Spatial proximity in APO structure
            apo_dist = calculate_distance(atom_info[i]['apo_coord'], atom_info[j]['apo_coord'])
            is_apo_close = apo_dist <= 8.0
            
            # 3. Spatial proximity in HOLO structure
            holo_dist = calculate_distance(atom_info[i]['holo_coord'], atom_info[j]['holo_coord'])
            is_holo_close = holo_dist <= 8.0
            
            # Add edge if any condition is met
            if is_sequential or is_apo_close or is_holo_close:
                edge_index.append([i, j])
                edge_index.append([j, i])
                
                # Create edge attributes
                edge_feature = np.array([
                    apo_dist,    # Distance in APO structure
                    holo_dist,   # Distance in HOLO structure
                    int(is_sequential),  # Is sequential connection
                    int(is_apo_close),   # Is close in APO
                    int(is_holo_close)   # Is close in HOLO
                ])
                edge_attr.extend([edge_feature, edge_feature])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,        # Node features
        edge_index=edge_index,  # Graph connectivity
        edge_attr=edge_attr,    # Edge features
        num_nodes=len(node_features)
    )
    
    return data

# Example usage
if __name__ == "__main__":
    file_path = Path(__file__).parent.parent / "data" / "a2h" / '1eb2_vec.pkl'
    # for file in file_path.glob('*.pkl'):
    result = read_a2h(str(file_path))
    if result.x.sum() > len(result.x):
    
        print(result.x)
        # print(result.edge_index)
        # print(result.edge_attr)
        print(f"Number of nodes: {result.num_nodes}")
        print(f"Number of edges: {result.edge_index.shape[1]}")
        print(f"Node features shape: {result.x.shape}")
    print('Done')