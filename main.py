from config import *
from process.prepare import *
from process.train import *

if __name__ == '__main__':
    ## set config
    args = set_config()
    PrepareData(args)
    Train_CV(args)