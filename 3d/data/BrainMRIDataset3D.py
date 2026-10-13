
import torch

from configs import get_config
cfg = get_config()
from data.dicom3dloader import load_dicom_images_3d
class BrianMRIDataset3D(Dataset):
    def __init__(self, df, scan_ids, split , img_size = cfg.glob.IMAGE_SIZE, num_images= cfg.glob.NUM_SLICES ,augmentation=None):
        self.scan_ids = scan_ids
        self.augmentation = augmentation
        self.num_images = num_images
        self.img_size = img_size  
        self.split = split  
        self.train_labels = df
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
        # print(f'shape is :{t2w_3d.shape}')
            
        if self.split == 'train':
            target = self.train_labels['MGMT_value'].iloc[idx]
            return torch.stack([flair_3d ,t1w_3d, t1wce_3d, t2w_3d ], dim=0) , target
        else:
            return torch.stack([flair_3d ,t1w_3d, t1wce_3d, t2w_3d ], dim=0) 
            
        