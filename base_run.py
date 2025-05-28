from config import *
from process.prepare_base import *
from process.train_base import *

if __name__ == '__main__':
    ## set config
    args = set_config()
    PrepareData(args)
    Train_CV(args)