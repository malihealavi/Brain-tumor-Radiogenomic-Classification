import gc 
import math
from typing import Any, Callable, Dict, Optional

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from model.Model3d import Model_3D                                   # 3D model
from data.BrianMRIDataset3D import  BrainMRIDataset3D        # 3D dataset
from model.Model25d import MRI25DFusionModel                         # 2.5D model
from data.BrainMRIDataset25D import (                                        # 2.5D dataset + transforms
    BrainMRIDataset ,
  
)

from data.augment import(
      get_train_transform,
      get_val_transform
)
from utils.utils import plot_loss
from configs import get_config
cfg = get_config()


TOTAL_BACKBONE_BLOCKS   = 12   
UNFREEZE_START_CYCLE    = 4     
CYCLES_PER_BLOCK        = 2     
RESNET_STAGES        = ["layer1", "layer2", "layer3", "layer4"]
MAX_UNFROZEN_BLOCKS  = 1
# ── collate for train/val (returns modalities + labels) ───────────────
def mri_collate_fn(batch):
    num_modalities = len(batch[0][0])
    modalities_out = [
        torch.stack([item[0][m] for item in batch], dim=0)
        for m in range(num_modalities)
    ]
    labels_out = torch.stack([item[1] for item in batch])
    return modalities_out, labels_out

# ── collate for test (no labels) ──────────────────────────────────────--
def mri_collate_fn_test(batch):
    # FIX BUG 1: test items are plain lists, no label
    num_modalities = len(batch[0])
    modalities_out = [
        torch.stack([item[m] for item in batch], dim=0)
        for m in range(num_modalities)
    ]
    return modalities_out


class Trainer():
    def __init__(self,
                 train: pd.DataFrame,
                 val: pd.DataFrame,
                 epochs: int,
                 device: str,
                 scaler: Optional[torch.amp.GradScaler] = None,
                 save_path : str=''): 
        self.save_path = save_path
        self.n_splits = 3
        self.train_transform = get_train_transform(num_slices=cfg.glob.NUM_SLICES, image_size=cfg.glob.IMAGE_SIZE)
        self.val_transform   = get_val_transform(num_slices=cfg.glob.NUM_SLICES, image_size=cfg.glob.IMAGE_SIZE)

        self.model = MRI25DFusionModel(
            d=cfg.model.d_25d, nhead=cfg.model.nhead_25d, num_layers=cfg.model.num_layers_25d,
            num_modalities=cfg.model.num_modalities_25d, max_slices=cfg.glob.NUM_SLICES
        )
        self.model.to(device)
        self.epochs    = epochs
        backbone_params = [p for n, p in self.model.named_parameters() if "slice_encoder.backbone" in n]
        other_params    = [p for n, p in self.model.named_parameters() if "slice_encoder.backbone" not in n]
        self.optimizer = optim.AdamW(
            [{"params": backbone_params, "lr": cfg.train.lr_bb_25D},
             {"params": other_params,    "lr": cfg.train.lr_head_25D}],
            weight_decay=cfg.train.weight_decay_25D)
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
      
        self.device    = device
        self.scaler    = scaler
        self.accumulation_steps = 4

        pos_weights       = (train['MGMT_value'] == 0).sum() / (train['MGMT_value'] == 1).sum()
        pos_weight_tensor = torch.tensor([pos_weights], dtype=torch.float32).to(device)
        self.criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.best_auc     = 0.0
        self.batch_size= cfg.train.batch_size_25D
        self.train = train.reset_index(drop=True)
        self.val   = val.reset_index(drop=True)

        train_ds = BrainMRIDataset(df=self.train, scan_ids=self.train['BraTS21ID'],
                                   split='train', augmentation=self.train_transform)
        val_ds   = BrainMRIDataset(df=self.val,   scan_ids=self.val['BraTS21ID'],
                                   split='train', augmentation=self.val_transform)

        self.train_loader = DataLoader(
            dataset=train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, collate_fn=mri_collate_fn
        )
        self.valid_loader = DataLoader(
            dataset=val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=mri_collate_fn
        )
       
    def _set_frozen_bn_eval(self):
        for n, m in self.model.named_modules():
            if "slice_encoder.backbone" in n and isinstance(m, nn.BatchNorm2d):
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()
                    
    def _unfreeze_slice_encoder(self, cycle):
        if cycle < UNFREEZE_START_CYCLE:
            n_unfrozen = 0
        else:
            n_unfrozen = 1 + (cycle - UNFREEZE_START_CYCLE) // CYCLES_PER_BLOCK
        n_unfrozen = min(n_unfrozen, MAX_UNFROZEN_BLOCKS)
        trainable_stages = RESNET_STAGES[len(RESNET_STAGES) - n_unfrozen:]
    
        unfrozen_params = 0
        for n, p in self.model.named_parameters():
            if "slice_encoder.backbone" not in n:
                continue
            # conv1, bn1 (stem) and fc always stay frozen
            p.requires_grad = any(f".{s}." in n for s in trainable_stages)
            if p.requires_grad:
                unfrozen_params += 1
        if n_unfrozen >= 3:
            new_bs, self.accumulation_steps = 4, 8
        else:
            new_bs = self.batch_size
    
        if new_bs != self.batch_size:
            self.batch_size = new_bs
            self.train_loader = DataLoader(self.train_loader.dataset, batch_size=new_bs,
                                           shuffle=True, num_workers=4, pin_memory=True,
                                           collate_fn=mri_collate_fn)
            self.valid_loader = DataLoader(self.valid_loader.dataset, batch_size=new_bs,
                                       shuffle=False, num_workers=4, pin_memory=True,
                                       collate_fn=mri_collate_fn)
        print(f"[cycle {cycle}] backbone: {trainable_stages or 'all frozen'} trainable "
          f"({unfrozen_params} params) | bs: {self.batch_size} | acc_steps: {self.accumulation_steps}")
        
    def _unpack(self, batch):
        modalities, targets = batch
        modalities = [m.to(self.device, dtype=torch.float32) for m in modalities]
        targets    = targets.to(self.device, dtype=torch.float32)
        return modalities, targets

    def train_one_epoch_accumulation(self, epoch):
        self.model.train()
        self._set_frozen_bn_eval()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader,
                            desc=f"Epoch {epoch}/{self.epochs} [Train]")
        self.optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            modalities, targets = self._unpack(batch)

            if i == 0:
                print(f'len modality: {len(modalities)}')
                print(f'modality[0] shape: {modalities[0].shape}')

            if self.scaler is not None:
                with torch.amp.autocast(device_type='cuda'):  
                    outputs = self.model(modalities)     
                    # outputs = outputs*0.9 + 0.05
                    loss    = self.criterion(outputs, targets)
                    loss    = loss / self.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(modalities)  
                # outputs = outputs*0.9 + 0.05
                loss    = self.criterion(outputs, targets)
                loss    = loss / self.accumulation_steps
                loss.backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.accumulation_steps * targets.size(0)
            progress_bar.set_postfix(
                {'loss': f"{loss.item() * self.accumulation_steps:.4f}"})

        return running_loss / len(self.train_loader.dataset)

    def valid_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_targets = [], []

        progress_bar = tqdm(self.valid_loader,
                            desc=f"Epoch {epoch}/{self.epochs} [Valid]")

        with torch.no_grad():
            for batch in progress_bar:
                modalities, targets = self._unpack(batch)

                if self.scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):  
                        outputs = self.model(modalities)           
                        loss    = self.criterion(outputs, targets)
                else:
                    outputs = self.model(modalities) 
                    loss    = self.criterion(outputs, targets)

                running_loss += loss.item() * targets.size(0)   
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.valid_loader.dataset)
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            print('batch only contains a single class')
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def training(self):
        print(f"Starting Training on device: {self.device}")
        patience = 0
        train_loss, valid_loss, valid_auc = [], [], []
        
        for epoch in range(1, self.epochs + 1):
            self._unfreeze_slice_encoder(epoch)      
            tr_loss           = self.train_one_epoch_accumulation(epoch)
            val_loss, val_auc = self.valid_one_epoch(epoch)
            train_loss.append(tr_loss)
            valid_loss.append(val_loss)
            valid_auc.append(val_auc)

            print(f"\nEpoch {epoch} Summary: Train Loss = {tr_loss:.4f} | "
                  f"Valid Loss = {val_loss:.4f} | Valid AUC = {val_auc:.4f} | "
                  f"LR:{self.optimizer.param_groups[0]['lr']:.6f}\n")

            self.scheduler.step()
            if val_auc > self.best_auc:
                print(f"Validation AUC improved from {self.best_auc:.4f} "
                      f"to {val_auc:.4f}. Saving best_model.pth...\n")
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.save_path)
                patience = 0
            else:
                patience += 1
            if patience == cfg.train.early_stopping_patience:
                break

        plot_loss(valid_loss, train_loss, valid_auc, f'loss_trend_{Path(self.save_path).stem}.png')
        print(f"Training Complete. Best Validation AUC: {self.best_auc:.4f}")

    def submit(self):
        print("\n--- Starting Test Inference ---")
        test_df = pd.read_csv(cfg.paths.test_root)
        test_df['BraTS21ID'] = test_df['BraTS21ID'].map('{:05}'.format)
        self.val_transform   = get_val_transform(num_slices=cfg.glob.NUM_SLICES, image_size=cfg.glob.IMAGE_SIZE)

        test_ds = BrainMRIDataset(test_df, scan_ids=test_df['BraTS21ID'],
                                  split='test', augmentation=self.val_transform)

        test_loader = DataLoader(
            dataset=test_ds, batch_size=cfg.train.batch_size_25D, shuffle=False,
            num_workers=4, pin_memory=True,
            collate_fn=mri_collate_fn_test
        )

        self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
        # self.model.load_state_dict(torch.load('/kaggle/input/2-5model/best_model_fold0.pth', map_location=self.device))
        self.model.eval()

        all_preds = []
        progress_bar = tqdm(test_loader, desc="[Test Inference]")

        with torch.no_grad():
            for modalities in progress_bar:
                modalities = [m.to(self.device, dtype=torch.float32) for m in modalities]

                if self.scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(modalities)
                else:
                    outputs = self.model(modalities)

                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)

        test_df['MGMT_value'] = all_preds
        test_df[['BraTS21ID', 'MGMT_value']].to_csv('submission.csv', index=False)
        print("Inference Complete! Saved predictions to submission.csv")

    def submitKFold(self, weights_dir='.', folds=None, image_size=cfg.glob.IMAGE_SIZE):
        folds = list(range(self.n_splits)) if folds is None else list(folds)
        print("\n--- Starting Test Inference ---")

        test_df = pd.read_csv(cfg.paths.test_root)
        test_df['BraTS21ID'] = test_df['BraTS21ID'].map('{:05}'.format)

        test_ds = BrainMRIDataset(test_df, scan_ids=test_df['BraTS21ID'], split='test',
                                  augmentation=get_val_transform(num_slices=cfg.glob.NUM_SLICES,
                                                                 image_size=image_size))
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4,
                                 pin_memory=True, collate_fn=mri_collate_fn_test)

        # load all fold models once (small models, so they fit in memory together)
        models = []
        for fold in folds:
            m = MRI25DFusionModel(d=256, nhead=4, num_layers=2,
                                  num_modalities=4, max_slices=32)
            m.to(self.device)
            m.load_state_dict(torch.load(f'{weights_dir}/best_model_fold{fold}.pth',
                                         map_location=self.device))
            m.eval()
            models.append(m)

        fold_preds = [[] for _ in models]
        with torch.no_grad():
            for modalities in tqdm(test_loader, desc="[Test Inference]"):
                modalities = [m.to(self.device, dtype=torch.float32) for m in modalities]
                for k, model in enumerate(models):
                    if self.scaler is not None:
                        with torch.amp.autocast(device_type='cuda'):
                            outputs = model(modalities)
                    else:
                        outputs = model(modalities)
                    fold_preds[k].extend(torch.sigmoid(outputs).float().cpu().numpy().reshape(-1))

        fold_preds = np.stack([np.array(p) for p in fold_preds])   # (n_folds, n_test)
        assert fold_preds.shape[1] == len(test_df), "prediction count != test rows"

        test_df['MGMT_value'] = fold_preds.mean(axis=0)
        test_df[['BraTS21ID', 'MGMT_value']].to_csv('submission.csv', index=False)
        print(f"Inference Complete! Averaged {len(models)} folds -> submission.csv")