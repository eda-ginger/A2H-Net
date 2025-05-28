import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
import sys
import numpy as np
from pathlib import Path
import argparse # For argparse.Namespace
import logging # Added logging
from tqdm import tqdm
import wandb
import json # Added for saving metrics

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools import get_model
from config import set_config # Keep for potential use, though args come from main.py
from utils.tools import set_seed, get_device, count_parameters, save_checkpoint, load_checkpoint
from utils.metrics import calculate_regression_metrics
from process.prepare import CustomDataset

# amp
from torch.cuda.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='"%(asctime)s [%(levelname)s] %(message)s"',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, fold_idx, scaler, scheduler, args):
    model.train()
    total_loss = 0
    
    # Create progress bar
    pbar = tqdm(train_loader, desc=f'Fold {fold_idx} - Epoch {epoch}', 
                leave=True, dynamic_ncols=True)
    
    for batch_idx, (inputs, affinities) in enumerate(pbar):
        ligands, sequences, pockets = inputs
        
        ligands = ligands.to(device)
        sequences = sequences.to(device)
        pockets = pockets.to(device)
        affinities = affinities.to(device)

        optimizer.zero_grad()
        
        if args.apex:
            with autocast():
                predictions = model((ligands, sequences, pockets))
                loss = loss_fn(predictions, affinities)
            
            # Scaler 사용
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if args.scheduler:
                scheduler.step()
        else:
            predictions = model((ligands, sequences, pockets))
            loss = loss_fn(predictions, affinities)
            loss.backward()
            if args.scheduler:
                scheduler.step()
            else:
                optimizer.step()

        total_loss += loss.item()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_reals = []

    with torch.no_grad():
        for batch_idx, (inputs, affinities) in enumerate(data_loader):
            ligands, sequences, a2hs = inputs

            ligands = ligands.to(device)
            sequences = sequences.to(device)
            a2hs = a2hs.to(device)
            affinities = affinities.to(device)

            predictions = model((ligands, sequences, a2hs))
            loss = loss_fn(predictions, affinities)
            
            total_loss += loss.item()
            all_preds.append(predictions.detach())
            all_reals.append(affinities.detach())
            
    avg_loss = total_loss / len(data_loader)
    all_preds_cat = torch.cat(all_preds, dim=0)
    all_reals_cat = torch.cat(all_reals, dim=0)
    
    eval_metrics = calculate_regression_metrics(all_preds_cat, all_reals_cat)
    return avg_loss, eval_metrics

def Train_CV(args):
    device = get_device(args)
    
    # path
    if args.model in ['DeepDTAF', 'CAPLA']:
        cache_path = Path(args.cache_dir) / 'Base' / 'Pocket'
    else:
        cache_path = Path(args.cache_dir) / 'Base' / 'Sequence'
    logger.info(f"Cache path: {cache_path}")
    
    # load dataset            
    trn_pkl = cache_path / f'Base_trn.pkl'
    with open(trn_pkl, 'rb') as f:
        trn_samples = pickle.load(f)
    train_dataset = CustomDataset(trn_samples)

    val_pkl = cache_path / f'Base_val.pkl'
    with open(val_pkl, 'rb') as f:
        val_samples = pickle.load(f)
    val_dataset = CustomDataset(val_samples)
    logger.info(f"TRN: {len(train_dataset)}, VAL: {len(val_dataset)}")
    
    core_pkl = cache_path / f'Base_core.pkl'
    with open(core_pkl, 'rb') as f:
        core_samples = pickle.load(f)
    core_dataset = CustomDataset(core_samples)
    logger.info(f"CORE: {len(core_dataset)}")
    
    tst105_pkl = cache_path / f'Base_tst105.pkl'
    with open(tst105_pkl, 'rb') as f:
        tst105_samples = pickle.load(f)
    tst105_dataset = CustomDataset(tst105_samples)
    logger.info(f"TST105: {len(tst105_dataset)}")
    
    # main loop
    seeds = [16875, 16876, 33927, 33928, 40000]
    for s_idx, seed in enumerate(seeds):
        fold_idx = s_idx + 1

        # args are passed from main.py
        logger.info(f"Seed: {seed}")
        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
    
        # Test datasets (CORE, TST105)
        overall_test_metrics = {'CORE': [], 'TST105': []}

        logger.info(f"\n--- Starting Fold {s_idx+1}/{len(seeds)} ---")

        # Initialize wandb for each fold
        if args.use_wandb:
            wandb.init(
                project=args.project,
                name=f"Fold_{fold_idx}",
                config=vars(args)
            )
            
        # create dataloaders
        trn_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CustomDataset.collate_fn, num_workers=args.n_workers, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                collate_fn=CustomDataset.collate_fn, num_workers=args.n_workers)
        
        # create model
        model = get_model(args).to(device)
        logger.info(f"Model: {args.model} | Parameters: {count_parameters(model)}")
        logger.info(f'Model Architecture:\n{model}')

        # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer = optim.AdamW(model.parameters())
        
        if args.loss == 'mse_mean':
            loss_fn = nn.MSELoss(reduction='mean')
        elif args.loss == 'mse_sum':
            loss_fn = nn.MSELoss(reduction='sum')
        else:
            raise ValueError(f"Invalid loss function: {args.loss}")
        
        if args.apex:
            scaler = GradScaler()
        else:
            scaler = None
        
        if args.scheduler:
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, 
                                                      epochs=args.n_epochs, 
                                                      steps_per_epoch=len(trn_loader))
        else:
            scheduler = None

        best_val_metric = float('inf')
        epochs_no_improve = 0
        start_epoch = 0
        
        fold_output_dir = Path('logs') / Path(args.project) / f"fold_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        logger.info(f"Starting training for fold {fold_idx} from epoch {start_epoch+1}")
        for epoch in range(start_epoch + 1, args.n_epochs + 1):
            train_loss = train_one_epoch(model, trn_loader, optimizer, loss_fn, device, epoch, fold_idx, scaler, scheduler, args)
            val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)

            if args.aim == 'rmse':
                current_val_metric = val_metrics['rmse']
            elif args.aim == 'mse':
                current_val_metric = val_loss
            else:
                raise ValueError(f"Invalid aim: {args.aim}")

            if current_val_metric < best_val_metric:
                logger.info(f"New best validation {args.aim.upper()}: {current_val_metric:.4f}")
                best_val_metric = current_val_metric
                epochs_no_improve = 0
                
                filename = 'checkpoint_last.pt'
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_metric': best_val_metric,
                    'args': vars(args) # Save args as dict
                }, is_best=True, filename=filename, output_dir=fold_output_dir)
            else:
                epochs_no_improve += 1
            
            # Log losses to wandb
            if args.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss})

            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} for fold {fold_idx}")
                break
        
        # fold finished
        logger.info(f"Finished training for fold {fold_idx}. Best Val Metric ({args.aim.upper()}): {best_val_metric:.4f}")

        # evaluate best model on test sets
        logger.info(f"Evaluating best model from fold {fold_idx} on test sets...")
        best_model_path = fold_output_dir / "model_best.pt"
        
        model_test = get_model(args).to(device)
        model_test = load_checkpoint(best_model_path, model_test) 

        # testing
        for test_set_name in ['CORE', 'TST105']:
            if test_set_name == 'CORE':
                test_dataset = core_dataset
            else:
                test_dataset = tst105_dataset

            test_loader = DataLoader(test_dataset, batch_size=300, shuffle=False, 
                                    collate_fn=CustomDataset.collate_fn, num_workers=args.n_workers)
            
            _, test_metrics = evaluate(model_test, test_loader, loss_fn, device)
            overall_test_metrics[test_set_name].append(test_metrics)
            
            # save fold's test metrics
            fold_metrics_file = fold_output_dir / f"{test_set_name}_metrics.json"
            test_metrics_serializable = {k: float(v) for k, v in test_metrics.items()}
            with open(fold_metrics_file, 'w') as f:
                json.dump(test_metrics_serializable, f, indent=4)

            if args.use_wandb:
                wandb.log({
                    f"{test_set_name}/rmse": test_metrics['rmse'],
                    f"{test_set_name}/mae": test_metrics['mae'],
                    f"{test_set_name}/sd": test_metrics['sd'],
                    f"{test_set_name}/pcc": test_metrics['pcc'],
                    f"{test_set_name}/r2": test_metrics['r2'],
                    f"{test_set_name}/ci": test_metrics['ci']
                })

        # Finish wandb run for this fold
        if args.use_wandb:
            wandb.finish()

    logger.info("Training and evaluation finished.")
    
    # Calculate average metrics across all folds
    avg_test_metrics = {}
    for test_set in ['CORE', 'TST105']:
        metrics_sum = {metric: 0.0 for metric in overall_test_metrics[test_set][0].keys()}
        for fold_metrics in overall_test_metrics[test_set]:
            for metric, value in fold_metrics.items():
                metrics_sum[metric] += value
        
        avg_test_metrics[test_set] = {
            metric: float(value / len(overall_test_metrics[test_set]))  # Convert to Python float
            for metric, value in metrics_sum.items()
        }
    
    # Save average metrics to file
    metrics_file = Path('logs') / Path(args.project) / "average_test_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(avg_test_metrics, f, indent=4)
    
    logger.info(f"Average test metrics saved to {metrics_file}")
    
    # Log average metrics
    logger.info("\nAverage Test Metrics:")
    for test_set, metrics in avg_test_metrics.items():
        logger.info(f"\n{test_set} Test Set:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    args = set_config()
    Train_CV(args)
