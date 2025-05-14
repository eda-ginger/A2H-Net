import argparse

def set_config():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--ligand', type=str, default='./data/ligand/', help='ligand path')
    parser.add_argument('--protein_seq', type=str, default='./data/sequence/', help='protein sequence path')
    parser.add_argument('--protein_a2h', type=str, default='./data/a2h/', help='protein apo_to_holopath')
    parser.add_argument('--data_info', type=str, default='./data_info.csv', help='summary')
    parser.add_argument('--cache_dir', type=str, default='./cache/', help='cache directory')
    parser.add_argument('--folder', type=str, default='Refined', help='folder name')
    parser.add_argument('--all_folder', action=argparse.BooleanOptionalAction, default=True) # if True, all folder will be processed
    parser.add_argument('--force_reload', action=argparse.BooleanOptionalAction, default=False)

    # project    
    parser.add_argument('--project_name', type=str, default='Try', help='project name')
    
    # training process
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=300, help='num of epochs')
    parser.add_argument('--n_workers', type=int, default=10, help='num of workers for dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_cuda', type=int, default=0)
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=False)

    # model parameters
    parser.add_argument('--one_shot', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_config()
    print(args)

