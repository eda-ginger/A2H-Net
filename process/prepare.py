import torch
import pandas as pd
from pathlib import Path
import sys
import os
import json
import pickle
import logging
import numpy as np
from sklearn.model_selection import KFold

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader

from config import set_config
from utils.seq_to_graph import drug_to_graph
from utils.seq_to_vec import integer_label_encoding
from utils.a2h_to_vec import read_a2h
# from utils.a2h_to_vec_new import read_a2h

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='"%(asctime)s [%(levelname)s] %(message)s"',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def sse_enconding(sse_path, max_sse_len):
    df = pd.read_csv(sse_path, index_col=0).drop(['idx'], axis=1).values
    sse_tensor = np.zeros((max_sse_len, 40))
    sse_tensor[:len(df)] = df[:max_sse_len]
    sse_tensor = torch.from_numpy(sse_tensor.astype(np.float32)).unsqueeze(0)
    return Data(x=sse_tensor)


class PrepareData:
    def __init__(self, args):
        # Store arguments and load CSV info
        self.args = args
        self.info = pd.read_csv(args.data_info)
        self.sse = Path(args.sse_path)
        
        # Set up a2h path and other config
        self.a2h_path = Path(args.protein_a2h)
        self.timesteps = args.timesteps
        self.all_atom = args.all_atom
        
        # Set up cache directory
        self.fold_num = args.n_splits
        # self.cache_path = Path(args.cache_dir)
        if args.model in ['DeepDTAF', 'CAPLA']:
            self.cache_path = Path(args.cache_dir) / 'pocket'
        else:
            self.cache_path = Path(args.cache_dir) / args.ligand
        self.cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache path: {self.cache_path}")
        
        # Check if cache exists or needs to be created
        self.force_reload = args.force_reload
        if not self.force_reload:
            self._check_cache()
        else:
            self._make_cache()

    def _check_cache(self):
        # Check if all required cache files exist
        for i in range(1, self.fold_num + 1):
            for split in ['trn', 'val']:
                path = self.cache_path / f"fold_{i}_{split}.pkl"
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found. Please run with force_reload=True.")
                
        for name in ["tst_core.pkl", "tst_csar.pkl"]:
            path = self.cache_path / name
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Please run with force_reload=True.")

    def _make_cache(self):
        # Create cache files for each fold and test set
        folds = sorted(self.cache_path.parent.glob('*fold_*.json'))
        
        if len(folds) == 0:
            raise ValueError(f"No folds found in {self.cache_path}")
        
        for fold in folds:
            fold_idx = fold.stem.split('_')[1]
            fold_info = json.load(open(fold))
            
            # Prepare train and val data for this fold
            fold_trn = self.info[self.info['PDB'].isin(fold_info['TRN'])]
            fold_val = self.info[self.info['PDB'].isin(fold_info['VAL'])]
            trn = [self._process_data(row) for _, row in tqdm(fold_trn.iterrows(), total=fold_trn.shape[0], desc=f"Fold {fold_idx} TRN")]
            val = [self._process_data(row) for _, row in tqdm(fold_val.iterrows(), total=fold_val.shape[0], desc=f"Fold {fold_idx} VAL")]
            logger.info(f"Fold-{fold_idx} TRN: {len(trn)}, VAL: {len(val)}")
            
            # Save train and val data to cache
            with open(self.cache_path / f"fold_{fold_idx}_trn.pkl", 'wb') as f:
                pickle.dump(trn, f)
            with open(self.cache_path / f"fold_{fold_idx}_val.pkl", 'wb') as f:
                pickle.dump(val, f)
                
        # Prepare and save test sets (CORE, CSAR)
        for set_name in ['CORE', 'CSAR']:
            df = self.info[self.info['SET'] == set_name]
            data = [self._process_data(row) for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=set_name)]
            with open(self.cache_path / f"tst_{set_name.lower()}.pkl", 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"TST({set_name}): {len(data)}")

    def _process_data(self, sample):
        pdb = sample['PDB']
        affinity = sample['LOGK']

        # Ligand encoding (graph or sequence)
        smi = sample['Ligand']
        smi = max(smi.split('.'), key=len) # remove metal atoms
        if self.args.ligand == 'graph':
            ligand = drug_to_graph(smi, file=False, graphdta=self.args.graphdta)
        else:
            if self.args.model in ['DeepDTAF', 'CAPLA']:
                ligand = integer_label_encoding(smi, tp='drug', max_length=150)
            else:
                ligand = integer_label_encoding(smi, tp='drug', max_length=100)
        
        # Protein
        if self.args.model in ['DeepDTAF', 'CAPLA']:
            global_path = self.sse / 'global' / f"{pdb}.csv"
            pocket_path = self.sse / 'pocket' / f"{pdb}.csv"
            protein = sse_enconding(global_path, max_sse_len=1000)
            pocket = sse_enconding(pocket_path, max_sse_len=63)

            # Check for missing data
            if ligand is None or protein is None or pocket is None or affinity is None:
                raise ValueError(f"None values found in data for {pdb}")

            return {
                'pdb_code': pdb,
                'ligand': ligand,
                'protein': protein,
                'pocket': pocket,
                'affinity': affinity
            }
            
        else:
            protein = integer_label_encoding(sample['Global'], tp='protein', max_length=1000)
            
            # a2h = read_a2h(self.a2h_path / f"{pdb}_a2h.pkl")
            # dummy a2h
            a2h = Data(x=torch.randn(1, 10))
        
            # Check for missing data
            if ligand is None or protein is None or a2h is None or affinity is None:
                raise ValueError(f"None values found in data for {pdb}")
        
            return {
                'pdb_code': pdb,
                'ligand': ligand,
                'protein': protein,
                'a2h': a2h,
                'affinity': affinity
            }


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    @staticmethod
    def collate_fn(batch):
        ligands = Batch.from_data_list([item['ligand'] for item in batch])
        affinities = torch.tensor([item['affinity'] for item in batch], dtype=torch.float).view(-1, 1)

        if 'pocket' in batch[0].keys():
            proteins = Batch.from_data_list([item['protein'] for item in batch])
            pocket = Batch.from_data_list([item['pocket'] for item in batch])
            return (ligands, proteins, pocket), affinities

        else:
            proteins = Batch.from_data_list([item['protein'] for item in batch])
            try:
                a2hs = Batch.from_data_list([item['a2hs'] for item in batch])
            except:
                a2hs = Batch.from_data_list([Data(x=torch.randn(1, 10)) for item in batch])
            return (ligands, proteins, a2hs), affinities


if __name__ == "__main__":
    args = set_config()
    logger.info("Starting data preparation...")

    for reload in [True, False]:
        args.force_reload = reload
        
        # Prepare data
        PrepareData(args)
        logger.info("PrepareData completed.")
        
        # Load data from cache (example)
        if args.model in ['DeepDTAF', 'CAPLA']:
            with open('cache/pocket/fold_1_val.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            with open('cache/seq/fold_1_val.pkl', 'rb') as f:
                data = pickle.load(f)
        
        # Create a small dataset for testing
        example = CustomDataset(data)
        loader = DataLoader(example, batch_size=5, shuffle=False, collate_fn=CustomDataset.collate_fn)
        for batch_idx, (inputs, affinities) in enumerate(loader):
            logger.info("Batch shapes:")
            logger.info(f"  Ligand batch shape: {inputs[0]}")
            logger.info(f"  Protein batch shape: {inputs[1]}")
            logger.info(f"  Pocket or A2H batch shape: {inputs[2]}")
            logger.info(f"  Affinity batch shape: {affinities.shape}")
            break
        logger.info("CustomDataset completed.")
        
        logger.info(f"Reload: {reload} operation completed")
        logger.info("--------------------------------")