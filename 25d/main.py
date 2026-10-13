
import os 
import gc
import numpy as np
import random
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
from torch.cuda.amp import autocast, GradScaler
from torch import nn

from Train.Trainer25 import Trainer
from XAI.xai import MRIExplainer
from utils.utils import seed_everything
from configs import get_config

cfg = get_config()



if __name__ == '__main__':

    seed_everything()   
    device =  torch.device(cfg.device)
    train_df = pd.read_csv(cfg.paths.train_root)
    train_df['BraTS21ID'] = train_df['BraTS21ID'].map('{:05}'.format)
    values_to_drop = ['00109', '00123', '00709']
    df = train_df[~train_df['BraTS21ID'].isin(values_to_drop)]
    n_splits = 10          # each fold = 90% train / 10% val
    folds_to_run = 3       # only run the first 3

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(train_df, train_df['MGMT_value']))

    for fold in range(folds_to_run):
        train_idx, val_idx = splits[fold]
        train = train_df.iloc[train_idx].reset_index(drop=True)
        val   = train_df.iloc[val_idx].reset_index(drop=True)
        print(f'len train: {len(train)}, val:{len (val)}')
        trainer = Trainer(
                train=train, val=val,
            epochs=8, device=device,
            save_path=f'./assets/25weights/best_model_fold{fold}.pth',
            scaler=GradScaler()
        )
        trainer.training()

        if fold < folds_to_run - 1:
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

    # ----------------- XAI ---------------------- 

    explainer = MRIExplainer(weights_dir='./assets/25weights', folds=range(3), device='cuda')

    # one case (scan_id must look like the value in df['BraTS21ID'], e.g. '00005')
    res = explainer.explain(df, scan_id='00005', split='train')
    explainer.plot(res, scan_id='00005', save_path='xai_00005.png')

    # average view over many cases
    explainer.global_modality_importance(df, df['BraTS21ID'].tolist()[:40])

    



    