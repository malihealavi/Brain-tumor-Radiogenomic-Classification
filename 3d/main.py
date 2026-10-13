
import os 
import gc
import numpy as np
import random
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
from torch.cuda.amp import GradScaler
from torch import nn

from utils.utils import seed_everything
from Train.Trainer import Trainer
from data.augment import get_train_transform, get_val_transform
from configs import get_config
cfg = get_config()

val_transform_func = get_val_transform(num_slices=cfg.glob.NUM_SLICES, image_size=cfg.glob.IMAGE_SIZE)
train_transform_func = get_train_transform(num_slices=cfg.glob.NUM_SLICES, image_size=cfg.glob.IMAGE_SIZE)


# ==========================================
# Main Execution / 5-Fold Cross Validation
# ==========================================
if __name__ == '__main__':

    seed_everything()
    train_df = pd.read_csv(cfg.paths.train_root)
    train_df['BraTS21ID'] = train_df['BraTS21ID'].map('{:05}'.format)
    values_to_drop = ['00109', '00123', '00709']
    df = train_df[~train_df['BraTS21ID'].isin(values_to_drop)].reset_index(drop=True)

    device = torch.device(cfg.device)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_paths = []

    # Iterate through the 5 folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['MGMT_value'])):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        
        trainer = Trainer(
            train_df=train_fold,
            val_df=val_fold,
            fold_num=fold,
            epochs=20, 
            device=device, 
            train_transform_func= train_transform_func, # assumed defined globally
            val_transform_func= val_transform_func,     # assumed defined globally
            scaler=GradScaler()
        )

        trainer.training()
        model_paths.append(trainer.model_save_path)
