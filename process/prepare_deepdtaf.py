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
        # self.info = pd.read_csv('data/DeepDTAF_data.csv')
        self.info = pd.read_csv('data/capla_dataset.csv')
        self.sse = Path('data/preprocessed/sse')
        
        # self.cache_path = Path(args.cache_dir)
        self.cache_path = Path(args.cache_dir) / 'CAPLA'
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Check if cache exists or needs to be created
        self.force_reload = args.force_reload
        if not self.force_reload:
            self._check_cache()
        else:
            self._make_cache()

    def _check_cache(self):
        # Check if all required cache files exist
        if not (self.cache_path / f"CAPLA_trn.pkl").exists():
            raise FileNotFoundError(f"{self.cache_path / f'CAPLA_trn.pkl'} not found. Please run with force_reload=True.")
        if not (self.cache_path / f"CAPLA_val.pkl").exists():
            raise FileNotFoundError(f"{self.cache_path / f'CAPLA_val.pkl'} not found. Please run with force_reload=True.")
        if not (self.cache_path / f"CAPLA_core.pkl").exists():
            raise FileNotFoundError(f"{self.cache_path / f'CAPLA_core.pkl'} not found. Please run with force_reload=True.")
        # if not (self.cache_path / f"DeepDTAF_tst105.pkl").exists():
        #     raise FileNotFoundError(f"{self.cache_path / f'DeepDTAF_tst105.pkl'} not found. Please run with force_reload=True.")
        # if not (self.cache_path / f"DeepDTAF_tst71.pkl").exists():
        #     raise FileNotFoundError(f"{self.cache_path / f'DeepDTAF_tst71.pkl'} not found. Please run with force_reload=True.")
        
        logger.info("All cache files exist.")

    def _make_cache(self):
        # Prepare train and val data for this fold
        trn_df = self.info[self.info['Split'] == 'TRN']
        val_df = self.info[self.info['Split'] == 'VAL']
        trn = [self._process_data(row) for _, row in tqdm(trn_df.iterrows(), total=trn_df.shape[0], desc=f"TRN")]
        val = [self._process_data(row) for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0], desc=f"VAL")]
        logger.info(f"TRN: {len(trn)}, VAL: {len(val)}")
            
        # Save train and val data to cache
        with open(self.cache_path / f"CAPLA_trn.pkl", 'wb') as f:
            pickle.dump(trn, f)
        with open(self.cache_path / f"CAPLA_val.pkl", 'wb') as f:
            pickle.dump(val, f)
                
        # Prepare and save test sets (CORE, tst105, tst71)
        core_df = self.info[self.info['Split'] == 'TST']
        # csar51_df = self.info[self.info['Set2'] == 'CSAR51']
        # csar36_df = self.info[self.info['Set2'] == 'CSAR36']
        # tst105_df = self.info[self.info['Set'] == 'TST105']
        # tst71_df = self.info[self.info['Set'] == 'TST71']
        
        core = [self._process_data(row) for _, row in tqdm(core_df.iterrows(), total=core_df.shape[0], desc=f"CORE")]
        # csar51 = [self._process_data(row) for _, row in tqdm(csar51_df.iterrows(), total=csar51_df.shape[0], desc=f"CSAR51")]
        # csar36 = [self._process_data(row) for _, row in tqdm(csar36_df.iterrows(), total=csar36_df.shape[0], desc=f"CSAR36")]

        # tst105 = [self._process_data(row) for _, row in tqdm(tst105_df.iterrows(), total=tst105_df.shape[0], desc=f"TST105")]
        # tst71 = [self._process_data(row) for _, row in tqdm(tst71_df.iterrows(), total=tst71_df.shape[0], desc=f"TST71")]
        
        with open(self.cache_path / f"CAPLA_core.pkl", 'wb') as f:
            pickle.dump(core, f)
        logger.info(f"CORE: {len(core)}")
        
        # with open(self.cache_path / f"CAPLA_csar51.pkl", 'wb') as f:
        #     pickle.dump(csar51, f)
        # logger.info(f"CSAR51: {len(csar51)}")
        
        # with open(self.cache_path / f"CAPLA_csar36.pkl", 'wb') as f:
        #     pickle.dump(csar36, f)
        # logger.info(f"CSAR36: {len(csar36)}")
        
        # with open(self.cache_path / f"DeepDTAF_tst105.pkl", 'wb') as f:
        #     pickle.dump(tst105, f)
        # logger.info(f"TST105: {len(tst105)}")
        # with open(self.cache_path / f"DeepDTAF_tst71.pkl", 'wb') as f:
        #     pickle.dump(tst71, f)
        # logger.info(f"TST71: {len(tst71)}")

    def _process_data(self, sample):
        pdb = sample['PDB']
        
        # Ligand encoding (graph or sequence)
        smi = sample['Ligand']
        # smi = max(smi.split('.'), key=len) # remove metal atoms

        if args.model in ['DeepDTAF', 'CAPLA']:
            ligand = integer_label_encoding(smi, tp='drug', max_length=150)

            global_path = self.sse / 'global' / f"{pdb}.csv"
            pkt_path = self.sse / 'pocket' / f"{pdb}.csv"
            
            protein = sse_enconding(global_path, max_sse_len=1000)
            pocket = sse_enconding(pkt_path, max_sse_len=63)
            affinity = sample['LOGK']
            
            # logger.info(protein.x.shape)
            # logger.info(pocket.x.shape)
            # raise
            
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
            ligand = integer_label_encoding(smi, tp='drug', max_length=100)
            protein = integer_label_encoding(sample['Global'], tp='protein', max_length=1000)
            a2hs = [None for _ in range(len(protein))]
            
            # Affinity value
            affinity = sample['LOGK']
            
            # Check for missing data
            if ligand is None or protein is None or affinity is None:
                raise ValueError(f"None values found in data for {pdb}")
            
            return {
                'pdb_code': pdb,
                'ligand': ligand,
                'protein': protein,
                'a2hs': a2hs,
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
        # args.force_reload = reload
        args.force_reload = False
        
        # Prepare data
        PrepareData(args)
        logger.info("PrepareData completed.")
        
        # logger.info(args.model)
        
        # Load data from cache (example)
        with open('cache/CAPLA/CAPLA_trn.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Create a small dataset for testing
        example = CustomDataset(data)
        loader = DataLoader(example, batch_size=5, shuffle=False, collate_fn=CustomDataset.collate_fn)
        for batch_idx, (inputs, affinities) in enumerate(loader):
            logger.info("Batch shapes:")
            logger.info(f"  Ligand batch shape: {inputs[0]}")
            logger.info(f"  Protein batch shape: {inputs[1]}")
            logger.info(f"  A2H batch shape: {inputs[2]}")
            logger.info(f"  Affinity batch shape: {affinities.shape}")
            break
        logger.info("CustomDataset completed.")
        
        logger.info(f"Reload: {reload} operation completed")
        logger.info("--------------------------------")
        
        
        
        
        