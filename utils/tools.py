########################################################################################################################
########## Import
########################################################################################################################

import torch
import random
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader

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
        return torch.device('cuda')
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


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_filename="model_best.pth.tar", output_dir="."):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(output_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)
        print(f"Saved new best model to {best_filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=torch.device('cpu')):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None, 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', float('inf')) # Assuming lower is better for best_metric (e.g., val_loss)
    
    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Could not load optimizer state: {e}. Optimizer will be re-initialized.")

    print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {start_epoch+1}. Best metric: {best_metric:.4f}")
    return model, optimizer, start_epoch, best_metric


def init_wandb(args, fold_idx=None):
    """Initialize Weights & Biases."""
    if args.use_wandb:
        run_name = args.wandb_run_name
        if fold_idx is not None:
            run_name = f"{args.wandb_run_name}_fold_{fold_idx}" if args.wandb_run_name else f"fold_{fold_idx}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity, # Add your wandb entity here if needed
            name=run_name,
            config=vars(args),
            reinit=True # Allow reinitialization for multiple folds
        )
        print(f"Weights & Biases initialized for project '{args.wandb_project}', run '{run_name}'")


if __name__ == "__main__":
    print("hello")
    # file_path = "data/sequence/CORE/1bcu_sequence.fasta"
    # selected_sequence = read_fasta(file_path)
    # print(f"Selected chain: {selected_sequence}")