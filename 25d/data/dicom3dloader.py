
from glob import glob
import pydicom
import numpy as np 
from pydicom.pixel_data_handlers.util import apply_voi_lut
import re

def load_dicom_image(path, img_size, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(path)
    if voi_lut:
        try:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        except Exception:
            data = dicom.pixel_array
    else:
        data = dicom.pixel_array
        
    data = data.astype(np.float32)
          
    # # # Resize to the standard bounding dimensions
    # data = cv2.resize(data, (img_size, img_size))
    return data


def load_dicom_images_3d(scan_id, num_imgs, img_size, mri_type="FLAIR", split="train", rotate=0, data_directory="."):
    path_pattern = f"/kaggle/input/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/{split}/{scan_id}/{mri_type}/*.dcm"
    files = sorted(glob(path_pattern), 
                   key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    if len(files) == 0:
        print(scan_id,':',mri_type)
        return np.zeros(( num_imgs, img_size, img_size), dtype=np.float32)
    
    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    # plt.imshow(load_dicom_image(files[middle], img_size=256, rotate=0) , cmap='gray')   
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    
    img3d = np.stack([load_dicom_image(f, img_size=img_size, rotate=rotate, voi_lut=True) for f in files[p1:p2]], axis=0) 
    # ----------- img3d = num_slices * H*W
    # return img3d  
    return img3d

