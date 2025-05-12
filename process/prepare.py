import torch
import pandas as pd
from pathlib import Path
import sys
import os
import pickle
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader

from config import set_config
from utils.seq_to_graph import drug_to_graph
from utils.seq_to_vec import protein_seq_to_vec
from utils.a2h_to_vec import read_a2h

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='"%(asctime)s [%(levelname)s] %(message)s"',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PrepareData(Dataset):
    def __init__(self, args, folder='Refined', force_reload=False):
        self.args = args
        self.folder = folder
        self.ligand_path = Path(args.ligand) / self.folder
        self.protein_seq_path = Path(args.protein_seq) / self.folder
        self.protein_a2h_path = Path(args.protein_a2h)
        self.info = pd.read_csv(args.data_info)
        
        # Define cache path
        self.cache_path = Path(args.cache_dir) / f"{folder}_data.pkl"
        
        self.data_list = []
        if not force_reload and self.cache_path.exists():
            self.load_cache()
        else:
            self.load_data()
            self.save_cache()
        
    def load_cache(self):
        """Load data from cache file."""
        try:
            with open(self.cache_path, 'rb') as f:
                self.data_list = pickle.load(f)
            logger.info(f"Loaded {len(self.data_list)} items from cache for {self.folder}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.load_data()
            self.save_cache()
    
    def save_cache(self):
        """Save data to cache file."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.data_list, f)
            logger.info(f"Saved {len(self.data_list)} items to cache for {self.folder}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
        
    def load_data(self):
        """Load and process data from source files."""
        logger.info(f"Loading data for {self.folder}...")
        ligand_files = {f.stem.split('_')[0]: f for f in self.ligand_path.glob('*')}
        seq_files = {f.stem.split('_')[0]: f for f in self.protein_seq_path.glob('*')}
        a2h_files = {f.stem.split('_')[0]: f for f in self.protein_a2h_path.glob('*')}
        logger.info(f"Found {len(ligand_files)} ligand files, {len(seq_files)} sequence files, {len(a2h_files)} a2h files")
        
        common_pdb_codes = set(ligand_files) & set(seq_files) & set(a2h_files)
        logger.info(f"Found {len(common_pdb_codes)} common PDB codes")
        
        for pdb_code in tqdm(common_pdb_codes, desc=f"Processing {self.folder}"):
            try:
                ligand_data = drug_to_graph(ligand_files[pdb_code], file=True)
                seq_data = protein_seq_to_vec(seq_files[pdb_code])
                a2h_data = read_a2h(a2h_files[pdb_code])
                
                if ligand_data and seq_data and a2h_data:
                    self.data_list.append({
                        'pdb_code': pdb_code,
                        'ligand': ligand_data,
                        'sequence': seq_data,
                        'a2h': a2h_data,
                        'affinity': self.info.loc[self.info['PDB'] == pdb_code, 'LOGK'].values[0]
                    })
            except Exception as e:
                logger.warning(f"Failed to process {pdb_code}: {e}")
                continue
        
        logger.info(f"ligand data shape: {self.data_list[0]['ligand'].x.shape} (atom_num, features)")
        logger.info(f"sequence data shape: {self.data_list[0]['sequence'].x.shape} (1, protein length)")
        logger.info(f"a2h data shape: {self.data_list[0]['a2h'].x.shape} (timesteps, max_atom_num, coord)")
        
        diff = len(ligand_files) - len(self.data_list)
        logger.info(f"({self.folder}) failed to load {diff}: {len(ligand_files)} -> {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def collate_fn(self, batch):
        ligands = [item['ligand'] for item in batch]
        sequences = torch.stack([item['sequence'] for item in batch])
        a2h_data = torch.stack([item['a2h'].x for item in batch])  # Access x attribute of Data object

        return Batch.from_data_list(ligands), sequences, a2h_data


class CustomDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

if __name__ == "__main__":
    args = set_config()
    logger.info("Starting data preparation...")
    train = PrepareData(args, folder='Refined', force_reload=True)
    test = PrepareData(args, folder='CORE', force_reload=True)
    test = PrepareData(args, folder='CSAR', force_reload=True)
    logger.info("Data preparation completed.")
