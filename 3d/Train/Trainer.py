
import math
from typing import Any, Callable, Dict, Optional

import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from model.Model3d import Model_3D                                   # 3D model
from data.BrainMRIDataset3D import  BrainMRIDataset3D                # 3D dataset
from configs import get_config
cfg = get_config()

from data.augment import(
      get_train_transform,
      get_val_transform
)
from utils.utils import plot_loss
            
class Trainer():
    def __init__(self, 
                 train_df: pd.DataFrame, 
                 val_df: pd.DataFrame,
                 fold_num: int,
                 epochs: int, 
                 device: str, 
                 train_transform_func,  
                 val_transform_func,   
                 scaler: Optional[torch.cuda.amp.GradScaler]=None):
        """
        Args:
            train_df: Training fold dataframe
            val_df: Validation fold dataframe
            fold_num: Current fold index (used for saving distinct model weights)
            epochs: Number of epochs
            device: cpu / gpu 
            scaler: Optional GradScaler for mixed precision.
        """
        self.train_transform = train_transform_func
        self.val_transform = val_transform_func
        self.model = Model_3D(bb=cfg.model.classifier_backbone_3D)
        self.model.to(device)
        self.epochs = epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.train.lr_3D, weight_decay=cfg.train.weight_decay_3D)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6)
        self.device = device
        self.scaler = scaler
        self.accumulation_steps = 4 
        self.fold_num = fold_num
        self.model_save_path = f'./assets/3d_weights/best_3d_model_fold_{fold_num}.pth'
        
        # ------------------- Loss & Eval ------------------
        pos_weights = (train_df['MGMT_value'] == 0).sum() / (train_df['MGMT_value'] == 1).sum()  
        pos_weight_tensor = torch.tensor([pos_weights], dtype=torch.float32).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.best_auc = 0.0
        
        # ------------------- Data Loader------------------
        train = train_df.reset_index(drop=True)
        val = val_df.reset_index(drop=True)
        
        train_ds = BrainMRIDataset3D(train, scan_ids=train['BraTS21ID'], split='train', augmentation=self.train_transform)
        val_ds = BrainMRIDataset3D(val, scan_ids=val['BraTS21ID'], split='train', augmentation=self.val_transform)

        self.train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True,
                                       num_workers=4, pin_memory=True) 
        
        self.valid_loader = DataLoader(dataset=val_ds, batch_size=4, shuffle=False,
                                       num_workers=4, pin_memory=True) 

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")

        for inputs, targets in progress_bar:
            inputs = inputs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs).squeeze(-1) 
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs).squeeze(-1)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        return running_loss / len(self.train_loader.dataset)

    def train_one_epoch_accumulation(self, epoch):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        self.optimizer.zero_grad()

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs).squeeze(-1) 
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs).squeeze(-1)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps
                loss.backward()

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.accumulation_steps * inputs.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item() * self.accumulation_steps:.4f}"})

        return running_loss / len(self.train_loader.dataset)
    
    def valid_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_targets = [], []
        progress_bar = tqdm(self.valid_loader, desc=f"Epoch {epoch}/{self.epochs} [Valid]")

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs).squeeze(-1)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs).squeeze(-1)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
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
        print(f"\n--- Starting Training for Fold {self.fold_num} on device: {self.device} ---")
        patience = 0
        train_loss, valid_loss, valid_auc = [], [], []
        
        for epoch in range(1, self.epochs + 1):
            tr_loss = self.train_one_epoch_accumulation(epoch)
            val_loss, val_auc = self.valid_one_epoch(epoch)
            
            train_loss.append(tr_loss)
            valid_loss.append(val_loss)
            valid_auc.append(val_auc)
            
            print(f"\nEpoch {epoch} Summary: Train Loss = {tr_loss:.4f} | Valid Loss = {val_loss:.4f} | Valid AUC = {val_auc:.4f} | LR:{self.optimizer.param_groups[0]['lr']:.8f}\n")
            self.scheduler.step()
            
            if val_auc > self.best_auc:
                print(f"Validation AUC improved from {self.best_auc:.4f} to {val_auc:.4f}. Saving {self.model_save_path}...\n")
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), self.model_save_path)
                patience = 0
            else:
                patience += 1 
                
            if patience == 4:
                print("Early stopping triggered.")
                break
                
        plot_loss(valid_loss, train_loss, valid_auc, f'loss_trend_fold_{self.fold_num}.png')     
        print(f"Training Complete for Fold {self.fold_num}. Best Validation AUC: {self.best_auc:.4f}")