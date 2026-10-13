import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data.augment import get_val_transform
from model.model25d import MRI25DFusionModel
from data.BrainMRIDataset25D import BrainMRIDataset
from configs import get_config
cfg = get_config()


MRI_NAMES = cfg.glob.mri_keys


class MRIExplainer:
    """
    Explains your saved fold models (MRI25DFusionModel with the ResNet18 slice encoder).
    Uses your existing: MRI25DFusionModel, BrainMRIDataset, get_val_transform, cfg.glob.NUM_SLICES, cfg.glob.IMAGE_SIZE.

    3 simple explanations:
      1. Modality importance : how much the prediction changes if one scan type is blanked out
      2. Slice importance    : which slices the model relied on
      3. Grad-CAM heatmap    : where inside the slice the model looked
    """

    def __init__(self, weights_dir='.', folds=range(5), device='cuda'):
        self.device = device
        self.val_tf = get_val_transform(num_slices=cfg.glob.NUM_SLICES, image_size=cfg.glob.IMAGE_SIZE)
        self.models = []
        for f in folds:
            m = MRI25DFusionModel(d=256, nhead=4, num_layers=2, num_modalities=4, max_slices=32)
            m.load_state_dict(torch.load(f'{weights_dir}/best_model_fold{f}.pth', map_location='cpu'))
            self.models.append(m.to(device).eval())
        print(f'Loaded {len(self.models)} fold model(s)')

    # ── load one case with the SAME val transform used in training ─────────
    def _get_case(self, df, scan_id, split='train'):
        row = df[df['BraTS21ID'] == scan_id].reset_index(drop=True)
        assert len(row) == 1, f'scan_id {scan_id!r} not found in df["BraTS21ID"]'
        ds = BrainMRIDataset(row, scan_ids=row['BraTS21ID'], split=split, augmentation=self.val_tf)
        item = ds[0]
        mods, label = item if split == 'train' else (item, None)
        mods = [m.unsqueeze(0).to(self.device, dtype=torch.float32) for m in mods]  # (1, N, 3, H, W)
        return mods, label

    # ── 1) modality importance: blank one scan type, see how prediction moves ─
    @torch.no_grad()
    def _modality_effect(self, model, mods):
        p_full = torch.sigmoid(model(mods)).item()
        effect = []
        for k in range(len(mods)):
            blank = [torch.zeros_like(m) if j == k else m for j, m in enumerate(mods)]
            effect.append(p_full - torch.sigmoid(model(blank)).item())
        return p_full, np.array(effect)          # effect[k] = prob drop when modality k is removed

    # ── 2) slice importance + 3) Grad-CAM, for ONE fold model ──────────────
    def _explain_one(self, model, mods):
        p_full, effect = self._modality_effect(model, mods)

        acts, grads, feats, fgrads = {}, {}, {}, {}

        def make_hook(store_out, store_grad):
            def hook(_, __, out):
                i = len(store_out)               # hooks fire once per modality, in order
                store_out[i] = out.detach()
                out.register_hook(lambda g, i=i: store_grad.__setitem__(i, g))
            return hook

        layer4 = model.slice_encoder.backbone.layer4
        layer4.requires_grad_(True)              # only so gradients can reach layer4 (cheap)
        h1 = layer4.register_forward_hook(make_hook(acts, grads))                # for Grad-CAM
        h2 = model.slice_encoder.register_forward_hook(make_hook(feats, fgrads)) # for slice importance

        model.zero_grad()
        model(mods).sum().backward()             # gradient of the logit ("methylated" score)
        h1.remove(); h2.remove()
        model.zero_grad(set_to_none=True)

        cams, imps = [], []
        for i, m in enumerate(mods):
            A, G = acts[i], grads[i]                                            # (N, 512, h, w)
            cam = F.relu((G.mean(dim=(2, 3), keepdim=True) * A).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=m.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
            cams.append(cam.cpu().numpy())                                      # (N, H, W)
            imps.append((fgrads[i] * feats[i]).sum(-1).abs().cpu().numpy())     # (N,)

        cmax = max(c.max() for c in cams) + 1e-8
        imax = max(s.max() for s in imps) + 1e-8
        return dict(prob=p_full, effect=effect,
                    cams=[c / cmax for c in cams], imps=[s / imax for s in imps])

    # ── explain one case with all loaded folds (averaged) ──────────────────
    def explain(self, df, scan_id, split='train'):
        mods, label = self._get_case(df, scan_id, split)
        rs = [self._explain_one(m, mods) for m in self.models]
        return dict(
            probs=np.array([r['prob'] for r in rs]),
            effect=np.mean([r['effect'] for r in rs], axis=0),
            cams=[np.mean([r['cams'][k] for r in rs], axis=0) for k in range(4)],
            imps=[np.mean([r['imps'][k] for r in rs], axis=0) for k in range(4)],
            volumes=[m[0, :, 1].cpu().numpy() for m in mods],   # centre channel of each triplet -> (N, H, W)
            label=label,
        )

    # ── one figure for both technical and non-technical readers ────────────
    def plot(self, res, scan_id='', save_path=None):
        p, sd = res['probs'].mean(), res['probs'].std()
        influence = np.abs(res['effect']) * 100          # percentage points
        top = int(influence.argmax())

        lines = [f'Scan: {scan_id}', '',
                 f'Predicted chance MGMT is methylated: {p * 100:.0f}%',
                 f"Fold models {'agree' if sd < 0.1 else 'DISAGREE'} (spread: {sd * 100:.0f} points)",
                 f'Most important scan type: {MRI_NAMES[top]}']
        if res['label'] is not None:
            lines.append(f"True label: {'methylated' if int(res['label']) == 1 else 'not methylated'}")
        lines += ['', "Red/yellow = image areas that pushed the", "prediction toward 'methylated'.",
                  '', 'Research prototype. Not for medical use.']

        fig = plt.figure(figsize=(16, 11))
        gs = GridSpec(3, 4, height_ratios=[0.9, 2, 1.1], figure=fig)

        # top-left: plain-language summary
        ax = fig.add_subplot(gs[0, :2]); ax.axis('off')
        ax.text(0, 1, '\n'.join(lines), va='top', fontsize=12)

        # top-right: which scan type matters most
        ax = fig.add_subplot(gs[0, 2:])
        ax.barh(MRI_NAMES, influence, color=['tab:red' if i == top else 'tab:gray' for i in range(4)])
        ax.invert_yaxis()
        ax.set_xlabel('Change in prediction when this scan is removed (points)')
        ax.set_title('Which scan type matters most?')

        # middle: Grad-CAM on the most important slice of each scan type
        for k, name in enumerate(MRI_NAMES):
            ax = fig.add_subplot(gs[1, k])
            s = int(res['imps'][k].argmax())
            img = res['volumes'][k][s]
            lo, hi = np.percentile(img, [1, 99.5])
            img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
            cam = res['cams'][k][s]
            cam = cam / (cam.max() + 1e-8)
            ax.imshow(img, cmap='gray')
            ax.imshow(np.ma.masked_less(cam, 0.25), cmap='jet', alpha=0.5, vmin=0, vmax=1)
            ax.set_title(f'{name} - slice {s}\nchange if removed: {influence[k]:.1f} pts')
            ax.axis('off')

        # bottom: which slices matter
        ax = fig.add_subplot(gs[2, :])
        for k, name in enumerate(MRI_NAMES):
            ax.plot(res['imps'][k], marker='o', ms=3, label=name)
        ax.set_xlabel('Slice index')
        ax.set_ylabel('Relative importance')
        ax.set_title('Which slices did the model rely on?')
        ax.legend(ncol=4)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    # ── dataset-level view: which scan type matters on average ─────────────
    def global_modality_importance(self, df, scan_ids, split='train', save_path=None):
        per_case = []
        for sid in scan_ids:
            mods, _ = self._get_case(df, sid, split)
            per_case.append(np.mean([self._modality_effect(m, mods)[1] for m in self.models], axis=0))
        imp = np.abs(np.array(per_case)) * 100            # (cases, 4) in points
        mean, sem = imp.mean(0), imp.std(0) / np.sqrt(len(imp))

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(MRI_NAMES, mean, yerr=sem, color='tab:blue', capsize=4)
        ax.set_ylabel('Avg change in prediction (points)')
        ax.set_title(f'Scan type importance (avg of {len(imp)} cases)')
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return dict(zip(MRI_NAMES, mean.round(2)))

explainer = MRIExplainer(weights_dir='/kaggle/input/2-5model', folds=range(3), device='cuda')
train_df = pd.read_csv(cfg.paths.train_root)
train_df['BraTS21ID'] = train_df['BraTS21ID'].map('{:05}'.format)
values_to_drop = ['00109', '00123', '00709']
train_df = train_df[~train_df['BraTS21ID'].isin(values_to_drop)]
# ───────────────────────── usage ─────────────────────────

device = torch.device(cfg.device)
n_splits = 10          # each fold = 90% train / 10% val
folds_to_run = 3       # only run the first 3

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
splits = list(skf.split(train_df, train_df['MGMT_value']))

for fold in range(folds_to_run):
    if fold == 0:
        train_idx, val_idx = splits[fold]
        train = train_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold0   = train_df.iloc[val_idx].reset_index(drop=True)
    elif fold == 1:
        train_idx, val_idx = splits[fold]
        train = train_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold1   = train_df.iloc[val_idx].reset_index(drop=True)
    elif fold == 2:
        train_idx, val_idx = splits[fold]
        train = train_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold2   = train_df.iloc[val_idx].reset_index(drop=True)
    
        break

train_df = train
