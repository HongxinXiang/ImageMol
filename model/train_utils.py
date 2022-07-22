import os
import random
import numpy as np
import pandas as pd
import torch


def fix_train_random_seed(seed=2021):
    # fix random seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def load_smiles(txt_file):
    '''
    :param txt_file: should be {dataset}_processed_ac.csv
    :return:
    '''
    df = pd.read_csv(txt_file)
    smiles = df["smiles"].values.flatten().tolist()
    return smiles

