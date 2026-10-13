import matplotlib.pyplot as plt
import seaborn as sns

import os 
import gc
import numpy as np
import random
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
from configs import get_config
cfg = get_config()

def seed_everything(seed=cfg.train.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Warning: the following lines can slow down training performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_loss(val_losses, train_losses,val_auc  ,path):
    epochs = range(1, len(train_losses) + 1)
    
    # Plot both training and validation losses
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o', linewidth=2)
    plt.plot(epochs, val_auc, label='Validation AUC', color='green', marker='o', linewidth=2)
    
    # Styling the plot
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Ensure x-axis only shows integer values for epochs
    plt.xticks(epochs)
    
    # Add a legend to differentiate the lines
    plt.legend(fontsize=11)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = path
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    # 4. Clean up the memory (highly recommended for background runs)
    plt.close()
    
    print(f"Plot successfully saved to {os.path.abspath(plot_path)}")
