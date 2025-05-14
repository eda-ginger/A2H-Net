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

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from config import set_config # Keep for potential use, though args come from main.py
from utils.tools import set_seed, get_device, count_parameters, log_metrics, save_checkpoint, load_checkpoint, init_wandb
from utils.metrics import calculate_regression_metrics
from process.prepare import PrepareData, CrossValidationSplit, FoldDataset, collate_fn # Added FoldDataset, collate_fn
# Import your model (adjust path and name if different)
from process.a2h_model import A2HNet 
import wandb


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='"%(asctime)s [%(levelname)s] %(message)s"',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args, fold_idx):
    model.train()
    total_loss = 0
    all_preds = []
    all_reals = []

    for batch_idx, (inputs, affinities) in enumerate(train_loader):
        ligands, sequences, a2hs = inputs
        
        ligands = ligands.to(device)
        sequences = sequences.to(device)
        a2hs = a2hs.to(device)
        affinities = affinities.to(device)

        optimizer.zero_grad()
        predictions = model((ligands, sequences, a2hs))
        loss = loss_fn(predictions, affinities)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(predictions.detach())
        all_reals.append(affinities.detach())

    avg_loss = total_loss / len(train_loader)
    all_preds_cat = torch.cat(all_preds, dim=0)
    all_reals_cat = torch.cat(all_reals, dim=0)
    
    train_metrics = calculate_regression_metrics(all_preds_cat, all_reals_cat)
    train_metrics['loss'] = avg_loss
    log_metrics(train_metrics, epoch, "train", args, fold_idx) # log_metrics now uses logger
    return avg_loss, train_metrics

def evaluate(model, data_loader, loss_fn, device, epoch, phase, args, fold_idx):
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
    eval_metrics['loss'] = avg_loss
    log_metrics(eval_metrics, epoch, phase, args, fold_idx) # log_metrics now uses logger
    return avg_loss, eval_metrics

def Train_CV(args):
    # args are passed from main.py
    set_seed(args.seed)
    device = get_device(args)
    
    # load full dataset (Refined)
    full_data = PrepareData(args).data_list
    if not full_data:
        logger.error("CRITICAL: Refined data list is empty. Please ensure Refined_data.pkl exists and is valid. Exiting.")
        return
    logger.info(f"Loaded {len(full_data)} items from Refined dataset for FoldDataset.")
    
    # load cv_splits
    cv_splitter = CrossValidationSplit(args)

    # Test datasets (CORE, CSAR)
    overall_test_metrics = {'CORE': [], 'CSAR': []}

    for fold_idx in range(args.n_splits):
        logger.info(f"\n--- Starting Fold {fold_idx + 1}/{args.n_splits} ---")
        if args.use_wandb:
            init_wandb(args, fold_idx=fold_idx + 1)

        train_pdb_codes, val_pdb_codes = cv_splitter.get_split(fold_idx)
        logger.info(f"Fold {fold_idx + 1}: {len(train_pdb_codes)} train, {len(val_pdb_codes)} val samples.")

        train_dataset = FoldDataset(full_data, train_pdb_codes)
        val_dataset = FoldDataset(full_data, val_pdb_codes)
        
        # check if datasets are empty
        if len(train_dataset) == 0:
            logger.warning(f"Train dataset for fold {fold_idx+1} is empty. Skipping fold.")
            if args.use_wandb and wandb.run:
                 wandb.finish()
            continue
        if len(val_dataset) == 0:
            logger.warning(f"Validation dataset for fold {fold_idx+1} is empty. Skipping fold.")
            if args.use_wandb and wandb.run:
                 wandb.finish()
            continue
            
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=collate_fn, num_workers=args.n_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_fn, num_workers=args.n_workers)
        

        raise Exception("Stop here")
        

        
        # Ensure sequence_len_example and a2h_max_atoms are in args (added to config.py)
        # These are needed for the A2HNet placeholder model.
        if not hasattr(args, 'sequence_len_example') or not hasattr(args, 'a2h_max_atoms'):
            logger.error("'sequence_len_example' or 'a2h_max_atoms' not found in args. Please add them to config.py")
            # Provide default values or raise an error if critical
            # Forcing defaults here for the code to run, but this should be configured properly.
            args.sequence_len_example = getattr(args, 'sequence_len_example', 1000) # Example default
            args.a2h_max_atoms = getattr(args, 'a2h_max_atoms', 290)       # Example default
            logger.warning(f"Using defaults: sequence_len_example={args.sequence_len_example}, a2h_max_atoms={args.a2h_max_atoms}")

        # create model
        model = A2HNet(
            ligand_input_dim=args.ligand_input_dim,
            seq_embed_dim=args.sequence_embed_dim, 
            seq_len_example=args.sequence_len_example, 
            a2h_timesteps=args.timesteps,
            a2h_max_atoms=args.a2h_max_atoms, 
            a2h_coord_dim=args.a2h_input_channels,
            hidden_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers,
            cnn_out_channels=args.cnn_out_channels,
            dropout_rate=args.dropout_rate
        ).to(device)
        
        logger.info(f"Model: {args.model_name} | Parameters: {count_parameters(model)}")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss_fn = nn.MSELoss()

        best_val_metric = float('inf')
        epochs_no_improve = 0
        start_epoch = 0
        
        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx+1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # Checkpoint loading for continuing training a specific fold (simplified)
        fold_checkpoint_path = os.path.join(fold_output_dir, "checkpoint.pth.tar")
        if args.checkpoint_to_load == "resume_fold" and os.path.exists(fold_checkpoint_path):
            logger.info(f"Resuming training for fold {fold_idx+1} from {fold_checkpoint_path}")
            model, optimizer, start_epoch, best_val_metric = load_checkpoint(fold_checkpoint_path, model, optimizer, device)
        elif args.checkpoint_to_load and args.checkpoint_to_load != "resume_fold":
             logger.info(f"Loading global checkpoint {args.checkpoint_to_load} for fold {fold_idx+1}. This might overwrite fold progress if not intended.")
             # This would typically be used if test_only=True, or starting all folds from one pretrained model
             model, _, _, _ = load_checkpoint(args.checkpoint_to_load, model, device=device) 

        if not args.test_only:
            logger.info(f"Starting training for fold {fold_idx+1} from epoch {start_epoch+1}")
            for epoch in range(start_epoch + 1, args.n_epochs + 1):
                train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args, fold_idx + 1)
                val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device, epoch, "val", args, fold_idx + 1)
                
                current_val_metric = val_metrics['rmse'] 
                if current_val_metric < best_val_metric:
                    best_val_metric = current_val_metric
                    epochs_no_improve = 0
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_metric': best_val_metric,
                        'args': vars(args) # Save args as dict
                    }, is_best=True, output_dir=fold_output_dir)
                else:
                    epochs_no_improve += 1
                
                if args.use_wandb:
                    wandb.log({f"Fold_{fold_idx+1}/val/best_val_metric_so_far": best_val_metric, f"Fold_{fold_idx+1}/epoch": epoch})

                if epochs_no_improve >= args.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} for fold {fold_idx+1}")
                    break
            logger.info(f"Finished training for fold {fold_idx + 1}. Best Val Metric (RMSE): {best_val_metric:.4f}")

        logger.info(f"Evaluating best model from fold {fold_idx + 1} on test sets...")
        best_model_path = os.path.join(fold_output_dir, "model_best.pth.tar")
        
        if args.test_only and args.checkpoint_to_load and args.checkpoint_to_load != "resume_fold":
            # If test_only and a global checkpoint is given, use that for all fold evaluations.
            # This assumes the global checkpoint path is what we want to test.
            # The model would have been loaded before the training loop if test_only was true AND checkpoint_to_load was set.
            # For clarity, ensure model is loaded from the global checkpoint if test_only
            logger.info(f"TEST_ONLY mode: Using checkpoint {args.checkpoint_to_load} for testing.")
            model_test, _, _, _ = load_checkpoint(args.checkpoint_to_load, model, device=device) # re-assign to model_test
            if model_test is None: # Check if loading failed
                 logger.error(f"Failed to load checkpoint {args.checkpoint_to_load} for test_only mode. Skipping test evaluation.")
                 if args.use_wandb and wandb.run:
                    wandb.finish()
                 continue # Skip to next fold or end
        elif os.path.exists(best_model_path):
            model_test, _, _, _ = load_checkpoint(best_model_path, model, device=device)
            if model_test is None:
                 logger.error(f"Failed to load best model from {best_model_path}. Skipping test evaluation.")
                 if args.use_wandb and wandb.run:
                    wandb.finish()
                 continue
        else:
            logger.error(f"No model found for testing fold {fold_idx+1}. Best model path: {best_model_path} does not exist, and not in test_only mode with a global checkpoint.")
            if args.use_wandb and wandb.run:
                 wandb.finish()
            continue
            
        for test_set_name in ['CORE', 'CSAR']:
            logger.info(f"Loading {test_set_name} test set for fold {fold_idx+1}...")
            test_load_args = argparse.Namespace(**vars(args))
            test_load_args.folder = test_set_name
            test_load_args.force_reload = False
            try:
                test_data_provider = PrepareData(test_load_args)
                if not test_data_provider.data_list:
                    logger.warning(f"Warning: {test_set_name} data list is empty. Skipping {test_set_name} test.")
                    continue
                
                # Use all data from the test set provider
                test_pdb_codes = [item['pdb_code'] for item in test_data_provider.data_list]
                test_dataset = FoldDataset(test_data_provider.data_list, test_pdb_codes)
                
                if len(test_dataset) == 0:
                    logger.warning(f"Warning: {test_set_name} test set is empty after FoldDataset. Skipping.")
                    continue
                
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                       collate_fn=collate_fn, num_workers=args.n_workers)
                
                _, test_metrics = evaluate(model_test, test_loader, loss_fn, device, args.n_epochs, f"test_{test_set_name}", args, fold_idx + 1)
                overall_test_metrics[test_set_name].append(test_metrics)
                if args.use_wandb:
                     wandb.log({f"Fold_{fold_idx+1}/Test_Summary/{test_set_name}_RMSE": test_metrics.get('rmse', float('nan'))})
            except FileNotFoundError:
                logger.warning(f"Warning: {test_set_name}_data.pkl not found for {test_set_name}. Skipping test.")
            except Exception as e:
                logger.error(f"Error loading or evaluating {test_set_name} for fold {fold_idx+1}: {e}", exc_info=True)

        if args.use_wandb and wandb.run:
            wandb.finish()

    logger.info("\n--- Overall Cross-Validation Test Results ---")
    for test_set_name, metrics_list in overall_test_metrics.items():
        if not metrics_list:
            logger.info(f"No test results for {test_set_name}.")
            continue
        
        avg_metrics = {}
        # Ensure all metrics dictionaries in metrics_list are not None and have keys
        valid_metrics_list = [m for m in metrics_list if m and isinstance(m, dict)]
        if not valid_metrics_list:
            logger.info(f"No valid metrics recorded for {test_set_name}.")
            continue
            
        # Get keys from the first valid metrics dictionary
        sample_keys = valid_metrics_list[0].keys()
        for key in sample_keys:
            if isinstance(valid_metrics_list[0][key], (int, float)):
                valid_values = [m[key] for m in valid_metrics_list if key in m and not np.isnan(m[key])]
                if valid_values:
                    avg_metrics[key] = np.mean(valid_values)
                    avg_metrics[key + '_std'] = np.std(valid_values)
                else:
                    avg_metrics[key] = np.nan
                    avg_metrics[key + '_std'] = np.nan                   
            # else: # Not strictly needed if all metrics are numeric
            #     avg_metrics[key] = 'N/A' 

        logger.info(f"Average Test Metrics for {test_set_name} ({len(valid_metrics_list)} valid folds):")
        for key, value in avg_metrics.items():
             logger.info(f"  {key}: {value:.4f}") # Unified print for float and std
        
        if args.use_wandb and args.n_splits > 0 : # Log summary if CV was run
            # Create a new summary run or use a specific step in the project.
            # This example won't create a new W&B run here, but you could.
            # For instance, log to a main project summary, perhaps not tied to a fold.
            summary_log_dict = {}
            for key, value in avg_metrics.items():
                summary_log_dict[f"Summary_Test_{test_set_name}/{key}"] = value
            
            # Need to re-init for a summary run OR ensure one is active from main.py
            # For simplicity, this final log might be better handled in main.py after run_training_cv completes.
            # if wandb.run is None and args.use_wandb: # If no run active (e.g. after fold runs finished)
            #    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=f"{args.wandb_run_name}_Summary" if args.wandb_run_name else "Overall_Summary", config=vars(args), job_type="summary")
            #    wandb.log(summary_log_dict)
            #    wandb.finish()
            # else: # If a run is somehow still active, just log (might mix with last fold's run)
            #    wandb.log(summary_log_dict) 
            logger.info(f"Consider logging summary metrics to W&B manually or with a dedicated summary run: {summary_log_dict}")

    logger.info("Training and evaluation finished.")



if __name__ == "__main__":
    args = set_config()
    Train_CV(args)
