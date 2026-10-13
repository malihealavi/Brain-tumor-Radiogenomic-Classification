import torch
from torch.utils.data import Dataset

from data.dicom3dloader import load_dicom_images_3d
from configs import get_config
cfg = get_config()

def stack_neighbors(volume: torch.Tensor, window: int = 1) -> torch.Tensor:
    """
    volume: (N, H, W)  — single modality, N slices
    returns: (N, 2*window+1, H, W)  — each slice paired with neighbors
    """
    N, H, W = volume.shape
    slices = []
    for i in range(N):
        neighbors = []
        for offset in range(-window, window + 1):
            idx = max(0, min(N - 1, i + offset))
            neighbors.append(volume[idx])
        slices.append(torch.stack(neighbors, dim=0))  # (3, H, W)
    return torch.stack(slices, dim=0)  # (N, 3, H, W)
    
class BrainMRIDataset(Dataset):
    def __init__(self, df, scan_ids, split , img_size = cfg.glob.IMAGE_SIZE, num_images= cfg.glob.NUM_SLICES ,augmentation=None , stack_neighbor=True):
        self.scan_ids = scan_ids
        self.augmentation = augmentation
        self.num_images = num_images
        self.img_size = img_size  
        self.split = split  
        self.train_labels = df
        self.stack_neighbor =stack_neighbor
        self.mri_type = ['FLAIR' , 'T1w' , 'T1wCE', 'T2w']
        
    def __len__(self):
        return len(self.scan_ids)
 
    def __getitem__(self, idx):
        
        flair_3d = load_dicom_images_3d(scan_id =  self.train_labels['BraTS21ID'].iloc[idx], num_imgs=self.num_images, 
                                        img_size=self.img_size, mri_type="FLAIR", split= self.split)
        t1w_3d = load_dicom_images_3d(scan_id = self.train_labels['BraTS21ID'].iloc[idx], num_imgs=self.num_images, 
                                        img_size=self.img_size, mri_type="T1w", split= self.split)
        t1wce_3d = load_dicom_images_3d(scan_id = self.train_labels['BraTS21ID'].iloc[idx], num_imgs=self.num_images, 
                                        img_size=self.img_size, mri_type="T1wCE", split = self.split)
        t2w_3d = load_dicom_images_3d(scan_id = self.train_labels['BraTS21ID'].iloc[idx], num_imgs=self.num_images, 
                                        img_size=self.img_size, mri_type="T2w", split = self.split)
       
        if self.augmentation:
            data_dict = {
                "FLAIR": flair_3d,
                "T1w": t1w_3d,
                "T1wCE": t1wce_3d,
                "T2w": t2w_3d
            }
            # MONAI applies the identical random matrix transform to all dictionary keys
            augmented = self.augmentation(data_dict)
            
            # Squeeze out the channel dimension added by EnsureChannelFirstd
            flair_3d = augmented["FLAIR"].squeeze(0)
            t1wce_3d = augmented["T1wCE"].squeeze(0)
            t2w_3d = augmented["T2w"].squeeze(0)
            t1w_3d = augmented["T1w"].squeeze(0)
            
        else:
            # Convert to torch tensors if no augmentation is run
            flair_3d = torch.as_tensor(flair_3d, dtype=torch.float32)
            t1w_3d = torch.as_tensor(t1w_3d, dtype=torch.float32)
            t1wce_3d = torch.as_tensor(t1wce_3d, dtype=torch.float32)
            t2w_3d = torch.as_tensor(t2w_3d, dtype=torch.float32)
            
        if self.stack_neighbor:
            flair_3d = stack_neighbors(flair_3d, window = 1)
            t1w_3d = stack_neighbors(t1w_3d, window = 1)
            t1wce_3d = stack_neighbors(t1wce_3d, window = 1)
            t2w_3d = stack_neighbors(t2w_3d, window = 1)
        
        if self.split == 'train':
            target = self.train_labels['MGMT_value'].iloc[idx]
            return [flair_3d ,t1w_3d, t1wce_3d, t2w_3d ], torch.tensor(target)
        else:
            return [flair_3d ,t1w_3d, t1wce_3d, t2w_3d ] 
