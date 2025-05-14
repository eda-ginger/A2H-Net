from config import *
from process.prepare import *
from process.train import *

if __name__ == '__main__':
    ## set config
    args = set_config()
    if args.all_folder:
        for folder in ['Refined', 'CORE', 'CSAR']:
            args.folder = folder
            PrepareData(args)
    else:
        PrepareData(args)
    
    ## split
    args.folder = 'Refined'
    CrossValidationSplit(args)
    
    ## training & evaluation
    Train_CV(args)