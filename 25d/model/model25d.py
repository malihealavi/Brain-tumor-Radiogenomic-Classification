
import torch
from torch import nn
import torchvision.models as models
from transformers import AutoImageProcessor, AutoModel
# ─────────────────────────────────────────────
# Shared 2D CNN slice encoder
# ─────────────────────────────────────────────
class SliceEncoder(nn.Module):
    """
    Input:  (B*N_m, 3, H, W)   — flat batch of triple-stacked slices
    Output: (B*N_m, d)          — one feature vector per slice
    """
    def __init__(self, d: int = 256, bb:str = 'resnet18'):
        super().__init__()
        self.proj = None
        
        if bb == 'resnet18':
            backbone = models.resnet18(weights=None)
            state_dict = torch.load("/kaggle/input/datasets/malihealavi/cnn-weights/resnet18_imagenet.pth", map_location="cpu")
            backbone.load_state_dict(state_dict)
            print(f'Selected Encoder: {bb}')
            feature = backbone.fc.in_features
            backbone.fc = nn.Linear(feature, d)
            for p in backbone.parameters():
                p.requires_grad = False
            
        elif bb== 'convnext':
            backbone = models.convnext_tiny(weights=None)
            state_dict = torch.load("/kaggle/input/datasets/malihealavi/cnn-weights/convnext_tiny.pth", map_location="cpu")
            backbone.load_state_dict(state_dict)
            feature = backbone.classifier[2].in_features
            backbone.classifier[2] = nn.Linear(feature, d)
        elif bb == 'dinov2_small':
            
            backbone = AutoModel.from_pretrained('/kaggle/input/models/metaresearch/dinov2/pytorch/base/1', local_files_only=True)
            feature = backbone.config.hidden_size          # 384 for dinov2-base
            self.proj = nn.Linear(feature, d)              # dino has no head to swap, so project manually
            print(f'Selected Encoder: {bb}')
            for p in backbone.parameters():
                p.requires_grad = False

        self.backbone = backbone
    def forward(self, x):
        if self.proj is not None:          # any DINO-style backbone (v2, v3, ...)
            outputs = self.backbone(x)
            x = outputs.last_hidden_state[:, 0, :]   # CLS token (summary vector)
            x = self.proj(x)
        else:
            x = self.backbone(x)
            # x = torch.flatten(x, 1)
        return x
# ─────────────────────────────────────────────
#      Within-modality slice attention
# ─────────────────────────────────────────────
class SliceAttentionPooler(nn.Module):
    """
    Input:  (B, N_m, d)  + optional mask (B, N_m)
    Output: (B, d)        — one vector per modality per case
    """
    def __init__(self, d: int = 256, nhead: int = 4,
                 num_layers: int = 2, max_slices: int = 32):
        super().__init__()
        self.pos_emb   = nn.Embedding(max_slices, d)
        # cls_token : a vector with size d that concat to slices of each modality ==>  shape changes from (B, N, d) to (B, N + 1, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d)) 
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=0.5, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, key_padding_mask=None):
        B, N, d = x.shape
        # add positional encoding
        pos = torch.arange(N, device=x.device) 
        x = x + self.pos_emb(pos)
        # Copies the CLS token for every item in the batch (1 cls per modality)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
            
        x = self.transformer(x, src_key_padding_mask=key_padding_mask) 
        return x[:, 0] 

# ─────────────────────────────────────────────
#  Cross-modality fusion attention
# ─────────────────────────────────────────────
class CrossModalFusion(nn.Module):
    """
    Input:  list of M tensors each (B, d) | for each Modality we have B vectors with size d
    Output: (B, d)  — fused across all modalities | all Modalities merge together so One vector per case
    """
    def __init__(self, d: int = 256, nhead: int = 4,
                 num_layers: int = 2, num_modalities: int = 4):
        super().__init__()
        self.mod_emb = nn.Embedding(num_modalities, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=0.5, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

    def forward(self, modality_vecs):
        # modality_vecs: list of 4 tensors each (B, d)
        x   = torch.stack(modality_vecs, dim=1)         # (B, 4, d)
        ids = torch.arange(x.size(1), device=x.device)
        x   = x + self.mod_emb(ids)                     # (B, 4, d)
        x   = self.transformer(x)                        # (B, 4, d)
        return x.mean(dim=1)                             # (B, d)


# ─────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────
class MRI25DFusionModel(nn.Module):
    """
    Input per forward call:
        modalities  — list of 4 tensors, each (B, N_m, 3, H, W)
                      N_m is 64 or 32 depending on the modality,
                      already neighbor-stacked by the Dataset.
        masks       — list of 4 bool tensors, each (B, N_m)
                      True = padding slice (case had fewer real slices
                      than N_m). Pass None if no padding needed.

    Output: (B,)  scalar in [0, 1]
    """
    def __init__(self,
                 d:              int = 256,
                 nhead:          int = 4,
                 num_layers:     int = 2,
                 num_modalities: int = 4,
                 max_slices:     int = 32):
        super().__init__()

        # one shared CNN for all slices of all modalities
        self.slice_encoder = SliceEncoder(d=d)

        # one independent slice-attention pooler per modality
        self.slice_poolers = nn.ModuleList([
            SliceAttentionPooler(d=d, nhead=nhead,
                                 num_layers=num_layers,
                                 max_slices=max_slices)
            for _ in range(num_modalities)
        ])

        self.fusion = CrossModalFusion(d=d, nhead=nhead,
                                       num_layers=num_layers,
                                       num_modalities=num_modalities)

        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d // 2, 1),
        )

    def forward(self, modalities, masks=None):
        """
        modalities: list of 4 tensors  (B, N_m, 3, H, W)
        masks:      list of 4 tensors  (B, N_m) bool  or  None
        """
        if masks is None:
            masks = [None] * len(modalities)

        modality_vecs = []

        for slices, pooler, mask in zip(modalities, self.slice_poolers, masks):
            # slices: (B, N_m, 3, H, W)
            B, N_m, C, H, W = slices.shape

            # ── Stage 2: CNN ──────────────────────────────────────────
            # flatten batch and slice dims so CNN sees one big pile
            flat  = slices.view(B * N_m, C, H, W)   # (B*N_m, 3, H, W)
            feats = self.slice_encoder(flat)          # (B*N_m, d)
            feats = feats.view(B, N_m, -1)           # (B, N_m, d)

            # ── Stage 3: slice attention ──────────────────────────────
            pooled = pooler(feats, mask)              # (B, d)
            modality_vecs.append(pooled)

        # ── Stage 4: cross-modal fusion ───────────────────────────────
        fused = self.fusion(modality_vecs)            # (B, d)

        # ── Stage 5: MLP head ─────────────────────────────────────────
        return self.head(fused).squeeze(-1)           # (B,)