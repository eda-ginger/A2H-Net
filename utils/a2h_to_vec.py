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

def read_a2h(file_path: str, timesteps: int = 10, all_atom: bool = False, pad: bool = True) -> Data:
    """
    Read and process A2H (Apo to Holo) protein structure data.
    
    Args:
        file_path (str): Path to the pickle file containing A2H data
        timesteps (int): Number of interpolation steps between apo and holo structures
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
    
    # Process each atom pair
    interpolated_coords = []
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
            
        # Calculate interpolated coordinates
        interpolation = [
            a_coord + (h_coord - a_coord) * (i / (timesteps - 1))
            for i in range(timesteps)
        ]
        interpolated_coords.append(interpolation)
    
    # Convert to numpy array first, then to tensor for better performance
    interpolated_coords = np.array(interpolated_coords)
    interpolated_coords = np.transpose(interpolated_coords, (1, 0, 2))
    result = torch.from_numpy(interpolated_coords).float()
    
    # Pad if requested
    if pad:
        result = F.pad(result, (0, 0, 0, max_len - result.shape[1]))
    
    return Data(x=result)

# Example usage
if __name__ == "__main__":
    file_path = Path(__file__).parent.parent / "data" / "a2h" / "1a1e_vec.pkl"
    result = read_a2h(str(file_path))
    print(f"Data shape: {result.x.shape}")  # Access shape through x attribute
    print(f"First timestep: {result.x[0]}")
    
    # RNN input shape: (batch_size, timesteps, max_atom_num * 3)
    # 샘플마다 atom 종류가 다르므로 옳지 않은 선택 (만약 종류를 하나의 값으로 준다면? 순서의 영향을 완전히 해결하지는 못함)
    print(result.x[0].view(1, 1, -1))
    print(result.x[0].view(1, 1, -1).shape)

    # 모델의 작동 방식 (하나씩)
    # 데이터가 변환되었을 때 그게 어떤 정보가 동일한지? (어떤 정보를 가져갔는지)
    # 틀리더라도 논리적으로 설명할 수 있어야한다 (과정의 연결성을 이해해야함)
