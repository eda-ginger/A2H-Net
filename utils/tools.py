########################################################################################################################
########## Import
########################################################################################################################

import torch
import random
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import logging
logger = logging.getLogger(__name__)

import os
import shutil
import wandb

########################################################################################################################
########## Functions
########################################################################################################################


def set_log(path_output, log_message):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(path_output / log_message),
            logging.StreamHandler()
        ]
    )


def set_random_seeds(seed: int):
    pyg.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"set seed: {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(args):
    """Get the appropriate torch device based on arguments."""
    if args.device == 'cuda' and torch.cuda.is_available():
        return torch.device(f'cuda:{args.gpu}')
    return torch.device('cpu')


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_metrics(metrics_dict, epoch, phase, args, fold_idx=None):
    """
    Log metrics to console (using logger) and to Weights & Biases.
    phase: 'train', 'val', or 'test_CORE', 'test_CSAR'
    """
    log_str = f"Epoch: {epoch}, Fold: {fold_idx if fold_idx is not None else '-'}, Phase: {phase}"
    for key, value in metrics_dict.items():
        log_str += f", {key}: {value:.4f}"
    logger.info(log_str)

    if args.use_wandb:
        wandb_log_dict = {}
        prefix = f"Fold_{fold_idx}/" if fold_idx is not None else ""
        prefix += f"{phase}/"
        
        for key, value in metrics_dict.items():
            wandb_log_dict[prefix + key] = value
        wandb_log_dict[prefix + 'epoch'] = epoch
        wandb.log(wandb_log_dict)


def save_checkpoint(state, is_best, filename="checkpoint.pt", best_filename="model_best.pt", output_dir="."):
    """Save model checkpoint and log metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save only model state_dict
    model_state = {
        'state_dict': state['state_dict']
    }
    
    # Save model state
    filepath = output_dir / filename
    torch.save(model_state, filepath)
    
    # Log metadata
    log_file = output_dir / "checkpoint_metadata.log"
    with open(log_file, 'a') as f:
        f.write(f"\n=== Checkpoint saved at {filepath} ===\n")
        f.write(f"Epoch: {state['epoch']}\n")
        f.write(f"Best Metric: {state['best_metric']:.4f}\n")
        f.write(f"Args: {state['args']}\n")
        f.write("="*50 + "\n")
    
    if is_best:
        best_filepath = output_dir / best_filename
        shutil.copyfile(filepath, best_filepath)
        logger.info(f"Saved new best model to {best_filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=torch.device('cpu')):
    """Load model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load only model state_dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Try to load metadata from log file
    log_file = checkpoint_path.parent / "checkpoint_metadata.log"
    if log_file.exists():
        with open(log_file, 'r') as f:
            logger.info(f"Loading metadata from {log_file}")
            # Read the last checkpoint entry
            log_content = f.read().split("="*50)[-2]  # Get the last entry before the current one
            logger.info(f"Checkpoint metadata:\n{log_content}")
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def init_wandb(args, fold_idx=None):
    """Initialize Weights & Biases."""
    config_dict = vars(args)
    config_dict['fold_idx'] = fold_idx
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"Fold_{fold_idx}",
        config=config_dict,
    )
    return run

from process.models import *

def get_model(args):
    if args.model == 'DeepDTA':
        model = DeepDTA()
    elif args.model == 'GraphDTA_GCN':
        model = GCNNet()
    elif args.model == 'GraphDTA_GAT':
        model = GATNet()
    elif args.model == 'GraphDTA_GIN':
        model = GINConvNet()
    elif args.model == 'GraphDTA_GAT_GCN':
        model = GAT_GCN()
    elif args.model == 'A2HNet_GAT':
        model = A2HNet_GAT()
    elif args.model == 'A2HNet_SEQ':
        model = A2HNet_SEQ()
    elif args.model == 'DeepDTAF':
        model = DeepDTAF()
    elif args.model == 'CAPLA':
        model = CAPLA()
    return model


if __name__ == "__main__":
    print("hello")
    # file_path = "data/sequence/CORE/1bcu_sequence.fasta"
    # selected_sequence = read_fasta(file_path)
    # print(f"Selected chain: {selected_sequence}")