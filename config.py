import argparse

def set_config():
    parser = argparse.ArgumentParser()
    
    # Data arguments from prepare.py (assuming these are still needed for context)
    # parser.add_argument('--ligand', type=str, default='./data/ligand/', help='Ligand data path')
    parser.add_argument('--ligand', type=str, default='graph', choices=['graph', 'seq'], help='Ligand data path')
    parser.add_argument('--protein_seq', type=str, default='./data/sequence/', help='Protein sequence data path')
    parser.add_argument('--protein_a2h', type=str, default='./data/preprocessed/a2h/', help='Protein A2H data path')
    parser.add_argument('--data_info', type=str, default='./data/data_info.csv', help='Path to data summary CSV (e.g., PDB codes and affinities)')
    parser.add_argument('--cache_dir', type=str, default='./cache/', help='Directory for cached data and splits')
    parser.add_argument('--folder', type=str, default='Refined', choices=['Refined', 'CORE', 'CSAR', 'Test'], help='Dataset folder name for prepare.py or direct loading')
    parser.add_argument('--all_folder', action=argparse.BooleanOptionalAction, default=False, help='If True, process all folders in prepare.py (usually False for training script)')
    parser.add_argument('--force_reload', action=argparse.BooleanOptionalAction, default=False, help='Force reload data in prepare.py, bypassing cache')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation') # Default changed from 10 for faster runs if needed
    parser.add_argument('--timesteps', type=int, default=10, help='Timesteps for a2h data processing')
    parser.add_argument('--all_atom', action=argparse.BooleanOptionalAction, default=False, help='Use all atoms for a2h data processing')

    # baseline
    parser.add_argument('--graphdta', action=argparse.BooleanOptionalAction, default=True, help='Use graphdta for a2h data processing')
    parser.add_argument('--protein_length', type=int, default=1000, help='Protein length for sequence data processing')
    
    # Training process arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=300, help='Patience for early stopping based on validation loss')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for DataLoader (0 for main process)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training (cuda or cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use for training')
 
    model_list = ['GraphDTA_GAT', 'GraphDTA_GCN', 'GraphDTA_GIN', 'GraphDTA_GAT_GCN', 'A2HNet_GAT', 'DeepDTA', 'A2HNet_SEQ']
    model_name = model_list[-1]
    # # Model specific arguments (placeholder - adjust based on your A2HNetModel)
    parser.add_argument('--model', type=str, default=model_name, help='Name of the model to use')
    # parser.add_argument('--ligand_input_dim', type=int, default=55, help='Dimension of ligand node features from drug_to_graph') # From GnS num_features_xd
    # parser.add_argument('--sequence_input_dim', type=int, default=1, help='Input channels for sequence data (e.g., 1 if (1,L))') 
    # parser.add_argument('--sequence_embed_dim', type=int, default=128, help='Embedding dimension for sequence tokens (if applicable)')
    # parser.add_argument('--a2h_input_channels', type=int, default=3, help='Number of coordinate channels for A2H data (e.g., 3 for x,y,z)')
    # parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for various layers in the model')
    # parser.add_argument('--gnn_layers', type=int, default=3, help='Number of GNN layers for ligand processing')
    # parser.add_argument('--cnn_out_channels', type=int, default=32, help='Output channels for sequence CNN') # From GnS n_filters
    # parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularization')
    
    # Weights & Biases arguments
    parser.add_argument('--project', type=str, default=model_name, help='Directory to save checkpoints and logs')
    # parser.add_argument('--project', type=str, default='test1', help='Directory to save checkpoints and logs')
    parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=True, help='Use Weights & Biases for logging')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_config()
    print(args)

