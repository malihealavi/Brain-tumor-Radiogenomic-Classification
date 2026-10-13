from monai.networks.nets import resnet10 ,resnet18, EfficientNetBN
import torch
from torch import nn


class Model_3D(nn.Module):
    def __init__(self, pretrained=True , bb='resnet10'):
        super(Model_3D, self).__init__()
        
        if bb == 'resnet10':
            self.backbone = resnet10(
                spatial_dims=3,          # Explicitly set to 3D convolutions
                n_input_channels=4,      # FLAIR, T1w, T1wCE, T2w 
                num_classes=1            # Single logit output for Binary Classification
            )
            
        elif bb == 'resnet18':
            self.backbone = resnet18(
                spatial_dims=3,          # Explicitly set to 3D convolutions
                n_input_channels=4,      # FLAIR, T1w, T1wCE, T2w 
                num_classes=1            # Single logit output for Binary Classification
            )
            
        elif bb == 'efficientnet':
            self.backbone = EfficientNetBN(
                model_name="efficientnet-b0",
                spatial_dims=3,          # Explicitly set to 3D convolutions
                in_channels=4,           # 4 distinct channels: FLAIR, T1w, T1wCE, T2w
                num_classes=1,           # Single logit output for Binary Classification
                pretrained=False         # Note: 3D pretrained weights are rare, usually set to False unless using a specific medical checkpoint
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return logits
