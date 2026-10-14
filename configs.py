
from dataclasses import dataclass, field
import os
import torch

def _get_default_device() -> str:
    """Returns 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class DataPaths:
    test_root = './assets/sample_submission.csv'
    train_root = './assets/train_labels.csv'
   


@dataclass
class GlobalVar:
    NUM_SLICES: int =  32
    IMAGE_SIZE :int = 256
    mri_keys : list = ["FLAIR", "T1w", "T1wCE", "T2w"]

@dataclass
class ModelConfig:
    classifier_backbone_3D: str = "resnet18"
    classifier_backbone_25D: str = "resnet18"
    d_25d: int= 256
    nhead_25d: int= 4
    num_layers_25d: int= 3
    num_modalities_25d: int= 4



@dataclass
class TrainConfig:
    seed: int = 42
    batch_size_3D: int = 4
    batch_size_25D: int = 8
    lr_3D: float = 5e-4
    lr_head_25D: float = 1e-4
    lr_bb_25D: float = 1e-5
    weight_decay_3D = 1e-2
    weight_decay_25D = 1e-1
    early_stopping_patience: int = 3
    epochs_3D = 8
    epochs_25D = 8


@dataclass
class InferenceConfig:
    image_size: int = 1024
    classification_threshold: float = 0.45
    segmentation_threshold: float = 0.5



@dataclass
class Config:
    paths: DataPaths = field(default_factory=DataPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    glob : GlobalVar = field(default_factory=InferenceConfig)
    device: str = field(default_factory=_get_default_device)

def get_config() -> Config:
    return Config()