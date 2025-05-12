
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



if __name__ == "__main__":
    print("hello")
    # file_path = "data/sequence/CORE/1bcu_sequence.fasta"
    # selected_sequence = read_fasta(file_path)
    # print(f"Selected chain: {selected_sequence}")