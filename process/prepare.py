import torch
import pandas as pd
from pathlib import Path
import sys
import os
import pickle
import logging
import numpy as np
from sklearn.model_selection import KFold

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
    def __init__(self, args):
        self.args = args
        self.folder = args.folder
        self.force_reload = args.force_reload
        self.ligand_path = Path(args.ligand) / self.folder
        self.protein_seq_path = Path(args.protein_seq) / self.folder
        self.protein_a2h_path = Path(args.protein_a2h)
        self.info = pd.read_csv(args.data_info)
        
        # a2h
        self.timesteps = args.timesteps
        self.all_atom = args.all_atom
        
        # Define cache path
        self.cache_path = Path(args.cache_dir) / f"{self.folder}_data.pkl"
        
        self.data_list = []
        if not self.force_reload and self.cache_path.exists():
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
                a2h_data = read_a2h(a2h_files[pdb_code], timesteps=self.timesteps, all_atom=self.all_atom)
                
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


class CrossValidationSplit:
    def __init__(self, args):
        self.args = args
        self.folder = args.folder
        self.n_splits = args.n_splits
        self.cache_path = Path(args.cache_dir) / f"{self.folder}_cv_splits.pkl"
        self.data_cache_path = Path(args.cache_dir) / f"{self.folder}_data.pkl"
        self.splits = []
        
        if not args.force_reload and self.cache_path.exists():
            self.load_splits()
        else:
            self.generate_splits()
            self.save_splits()
    
    def load_splits(self):
        """Load cross-validation splits from cache."""
        try:
            with open(self.cache_path, 'rb') as f:
                self.splits = pickle.load(f)
            logger.info(f"Loaded {len(self.splits)} cross-validation splits from cache")
        except Exception as e:
            logger.warning(f"Failed to load CV splits cache: {e}")
            self.generate_splits()
            self.save_splits()
    
    def save_splits(self):
        """Save cross-validation splits to cache."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.splits, f)
            logger.info(f"Saved {len(self.splits)} cross-validation splits to cache")
        except Exception as e:
            logger.error(f"Failed to save CV splits cache: {e}")
    
    def generate_splits(self):
        """Generate cross-validation splits from existing data cache."""
        try:
            # Load data directly from the existing cache file
            with open(self.data_cache_path, 'rb') as f:
                data_list = pickle.load(f)
            logger.info(f"Loaded {len(data_list)} items from existing data cache")
            
            # Get all PDB codes
            pdb_codes = [item['pdb_code'] for item in data_list]
            
            # Generate splits using KFold
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.args.seed)
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(pdb_codes)):
                train_codes = [pdb_codes[i] for i in train_idx]
                val_codes = [pdb_codes[i] for i in val_idx]
                
                self.splits.append({
                    'fold': fold_idx,
                    'train_codes': train_codes,
                    'val_codes': val_codes
                })
            
            logger.info(f"Generated {len(self.splits)} cross-validation splits")
            
        except Exception as e:
            logger.error(f"Failed to generate splits from cache: {e}")
            raise
    
    def get_split(self, fold_idx):
        """Get train and validation PDB codes for a specific fold."""
        if fold_idx >= len(self.splits):
            raise ValueError(f"Fold index {fold_idx} is out of range. Total folds: {len(self.splits)}")
        
        split = self.splits[fold_idx]
        return split['train_codes'], split['val_codes']


if __name__ == "__main__":
    args = set_config()
    logger.info("Starting data preparation...")
    
    for reload in [True, False]:
        args.force_reload = reload

        args.folder = 'Refined'
        train = PrepareData(args)

        args.folder = 'CORE'
        test = PrepareData(args)

        args.folder = 'CSAR'
        test = PrepareData(args)
        logger.info("Data preparation completed.")
        
        args.folder = 'Refined'
        cv = CrossValidationSplit(args)
        logger.info("Cross-validation splits completed.")
        
        logger.info(f"Reload: {reload} operation completed")
        logger.info("--------------------------------")
        
        
