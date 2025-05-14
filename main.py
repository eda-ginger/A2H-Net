from config import *
from process.prepare import *

if __name__ == '__main__':
    ## set config
    args = set_config()
    if args.all_folder:
        for folder in ['Refined', 'CORE', 'CSAR']:
            args.folder = folder
            PrepareData(args)
    else:
        PrepareData(args)
    
    ## read & split (train & valid)
    args.folder = 'Refined'
    CrossValidationSplit(args)
    
    ## training
    
    
    ## evaluation
    
    
    # pd.run(sub_to_run, split=True, expand=True)
    # cv = CrossValidation(args)
    # seed_all(args.random_seed)
    # cv.n_fold_CV(subject=sub_to_run, fold=args.fold, reproduce=args.reproduce)