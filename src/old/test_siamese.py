"""
Pipeline d'inférence pour détection de changement multi-modal MSI vs HSI.

Charge deux scènes (HSI Before + MSI After):
1. Génère HSI2_rec à partir de MSI After
2. Calcule l'incertitude
3. Compare les embeddings MSI After vs HSI Before
4. Visualise les résultats

Utilisation:
    python inference_change_detection_clean.py --config config.yaml --data_dir /path/to/scenes --output_dir results
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm


import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.constants import WVL_PRS, INTERP_MATRIX, SRF_MATRIX
from src.models.dual_branch import DualBranchNAFNet
from src.models.unet_residual import GradualExpansionUNet_residual
from src.models.contrastive import SiameseNetwork_heterogene


# ============================================================================
# DATASETS
# ============================================================================

class MSIDataset(Dataset):
    """Dataset pour MSI (Sentinel-2)."""
    def __init__(self, msi_path, patch_size=256, n_channels=12):
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        with xr.open_dataset(msi_path) as ds:
            var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
            cube = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
        
        h, w, c = cube.shape
        self.h_crop = h - (h % patch_size)
        self.w_crop = w - (w % patch_size)
        self.cube = cube[:self.h_crop, :self.w_crop, :self.n_channels]
        
        self.n_patches_h = self.h_crop // patch_size
        self.n_patches_w = self.w_crop // patch_size
        
        print(f"📊 MSI loaded: {self.cube.shape} | Patches: {self.n_patches_h}x{self.n_patches_w}")
    
    def __len__(self):
        return self.n_patches_h * self.n_patches_w
    
    def __getitem__(self, idx):
        i = idx // self.n_patches_w
        j = idx % self.n_patches_w
        y, x = i * self.patch_size, j * self.patch_size
        
        patch = self.cube[y:y+self.patch_size, x:x+self.patch_size, :]
        patch = np.transpose(patch, (2, 0, 1))
        return torch.from_numpy(patch).float(), idx


class HSIDataset(Dataset):
    """Dataset pour HSI (Prisma)."""
    def __init__(self, hsi_path, patch_size=256, n_channels=195):
        self.patch_size = patch_size
        self.n_channels = n_channels
        
        with xr.open_dataset(hsi_path) as ds:
            var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
            cube = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
        
        h, w, c = cube.shape
        self.h_crop = h - (h % patch_size)
        self.w_crop = w - (w % patch_size)
        self.cube = cube[:self.h_crop, :self.w_crop, :self.n_channels]
        
        self.n_patches_h = self.h_crop // patch_size
        self.n_patches_w = self.w_crop // patch_size
        
        print(f"📊 HSI loaded: {self.cube.shape} | Patches: {self.n_patches_h}x{self.n_patches_w}")
    
    def __len__(self):
        return self.n_patches_h * self.n_patches_w
    
    def __getitem__(self, idx):
        i = idx // self.n_patches_w
        j = idx % self.n_patches_w
        y, x = i * self.patch_size, j * self.patch_size
        
        patch = self.cube[y:y+self.patch_size, x:x+self.patch_size, :]
        patch = np.transpose(patch, (2, 0, 1))
        return torch.from_numpy(patch).float(), idx


class MSIInterpolatedDataset(Dataset):
    """Dataset pour MSI interpolé vers HSI via INTERP_MATRIX."""
    def __init__(self, msi_path, patch_size=256, n_channels_hsi=195):
        self.patch_size = patch_size
        self.n_channels_hsi = n_channels_hsi
        
        with xr.open_dataset(msi_path) as ds:
            var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
            cube_msi = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
        
        h, w, c_msi = cube_msi.shape
        self.h_crop = h - (h % patch_size)
        self.w_crop = w - (w % patch_size)
        cube_msi = cube_msi[:self.h_crop, :self.w_crop, :]
        
        # Appliquer INTERP_MATRIX sur la scène entière
        print("🔄 Interpolation globale via INTERP_MATRIX...")
        msi_chw = np.transpose(cube_msi, (2, 0, 1))
        interp_numpy = INTERP_MATRIX @ msi_chw.reshape(c_msi, -1)
        c_hsi = interp_numpy.shape[0]
        self.cube = interp_numpy.reshape(c_hsi, self.h_crop, self.w_crop)
        self.cube = np.transpose(self.cube, (1, 2, 0))[:, :, :self.n_channels_hsi]
        
        self.n_patches_h = self.h_crop // patch_size
        self.n_patches_w = self.w_crop // patch_size
        
        print(f"📊 MSI Interpolated: {self.cube.shape} | Patches: {self.n_patches_h}x{self.n_patches_w}")
    
    def __len__(self):
        return self.n_patches_h * self.n_patches_w
    
    def __getitem__(self, idx):
        i = idx // self.n_patches_w
        j = idx % self.n_patches_w
        y, x = i * self.patch_size, j * self.patch_size
        
        patch = self.cube[y:y+self.patch_size, x:x+self.patch_size, :]
        patch = np.transpose(patch, (2, 0, 1))
        return torch.from_numpy(patch).float(), idx


# ============================================================================
# UTILS
# ============================================================================

def get_rgb_indices(wvl=WVL_PRS):
    """Trouve les indices RGB."""
    return [
        np.argmin(np.abs(wvl - 665.0)),  # Red
        np.argmin(np.abs(wvl - 560.0)),  # Green
        np.argmin(np.abs(wvl - 490.0)),  # Blue
    ]


def to_rgb(cube_chw, indices):
    """Convertit un cube (C, H, W) en RGB."""
    valid_idx = [i for i in indices if i < cube_chw.shape[0]]
    if len(valid_idx) == 3:
        rgb = np.transpose(cube_chw[valid_idx], (1, 2, 0))
    else:
        rgb = np.transpose(cube_chw[:3], (1, 2, 0))
    return np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)


def infer_scene(msi_dataset, hsi_dataset, msi_interp_dataset,
                model_rec, model_unc, model_siamese, device, use_amp=True):
    """Inférence complète sur une scène."""
    model_rec.eval()
    model_unc.eval()
    model_siamese.eval()
    
    hsi2_rec_list = []
    unc_list = []
    score_list = []
    
    for idx in tqdm(range(len(msi_dataset)), desc="Inférence"):
        msi_patch, _ = msi_dataset[idx]
        hsi_patch, _ = hsi_dataset[idx]
        msi_interp_patch, _ = msi_interp_dataset[idx]
        
        # Ajouter batch dimension
        msi = msi_patch.unsqueeze(0).to(device)
        hsi = hsi_patch.unsqueeze(0).to(device)
        msi_interp = msi_interp_patch.unsqueeze(0).to(device)
        
        with torch.inference_mode():
            with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                # Reconstruction: MSI → HSI
                hsi_rec = model_rec(msi, msi_interp)
                
                # Incertitude
                unc_input = torch.cat([msi, hsi_rec], dim=1)
                unc = model_unc(unc_input)
                
                # Embeddings Siamois
                z_msi, z_hsi = model_siamese(msi, hsi)
                
                # Calcul des scores
                if z_msi.dim() == 4 and z_hsi.dim() == 4:
                    z_msi_norm = F.normalize(z_msi, p=2, dim=1)
                    z_hsi_norm = F.normalize(z_hsi, p=2, dim=1)
                    scores = torch.norm(z_msi_norm - z_hsi_norm, p=2, dim=1, keepdim=True)
                else:
                    # Si embeddings globaux, broadcast sur la grille
                    dist = torch.norm(z_msi - z_hsi, p=2, dim=1)
                    scores = dist.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, msi.shape[2], msi.shape[3])
        
        hsi2_rec_list.append(hsi_rec.squeeze(0).cpu().numpy())
        unc_list.append(unc.squeeze(0).cpu().numpy())
        score_list.append(scores.squeeze(0).cpu().numpy())
    
    return hsi2_rec_list, unc_list, score_list


def visualize_results(hsi1_data, msi_data, hsi2_rec_data, unc_data, score_data, output_path, max_samples=6):
    """Visualise 6 colonnes: MSI | HSI1 | HSI2_rec | UNC | Error | Score."""
    n_samples = min(len(hsi1_data), max_samples)
    fig, axes = plt.subplots(n_samples, 6, figsize=(24, 4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)
    
    rgb_hsi = get_rgb_indices(WVL_PRS)
    rgb_msi = [3, 2, 1]
    
    for i in range(n_samples):
        hsi1_chw = hsi1_data[i]
        msi_chw = msi_data[i]
        hsi2_rec_chw = hsi2_rec_data[i]
        unc_chw = unc_data[i]
        score_chw = score_data[i]
        
        # Col 0: MSI
        axes[i, 0].imshow(to_rgb(msi_chw, rgb_msi))
        if i == 0: axes[i, 0].set_title("MSI After", fontweight='bold')
        axes[i, 0].axis('off')
        
        # Col 1: HSI Before
        axes[i, 1].imshow(to_rgb(hsi1_chw, rgb_hsi))
        if i == 0: axes[i, 1].set_title("HSI Before", fontweight='bold')
        axes[i, 1].axis('off')
        
        # Col 2: HSI2 Rec
        axes[i, 2].imshow(to_rgb(hsi2_rec_chw, rgb_hsi))
        if i == 0: axes[i, 2].set_title("HSI2 Rec", fontweight='bold')
        axes[i, 2].axis('off')
        
        # Col 3: Uncertainty
        if unc_chw.ndim == 3:
            unc_map = np.mean(unc_chw, axis=0)
        else:
            unc_map = unc_chw
        axes[i, 3].imshow(unc_map, cmap='hot')
        if i == 0: axes[i, 3].set_title("Uncertainty", fontweight='bold')
        axes[i, 3].axis('off')
        
        # Col 4: Reconstruction Error |HSI1 - HSI2_rec|
        min_c = min(hsi1_chw.shape[0], hsi2_rec_chw.shape[0])
        error_map = np.mean(np.abs(hsi1_chw[:min_c] - hsi2_rec_chw[:min_c]), axis=0)
        axes[i, 4].imshow(error_map, cmap='YlOrRd')
        if i == 0: axes[i, 4].set_title("Rec Error", fontweight='bold')
        axes[i, 4].axis('off')
        
        # Col 5: Embedding Distance Score
        if score_chw.ndim == 3:
            score_map = score_chw[0]
        else:
            score_map = score_chw
        axes[i, 5].imshow(score_map, cmap='viridis')
        if i == 0: axes[i, 5].set_title("Embedding Score", fontweight='bold')
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change Detection Inference Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="results_change_detection")
    parser.add_argument("--scenes", type=str, nargs='+', default=None, help="Scene names (default: all)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}\n")
    
    n_msi = config["data"].get("n_msi", 12)
    n_hsi = config["data"].get("n_hsi", 195)
    patch_size = config["data"].get("patch_size", 256)
    
    # Load models
    print("🏗️ Loading models...")
    eval_cfg = config.get("evaluation", {})
    
    # Reconstruction
    model_rec = GradualExpansionUNet_residual(
        in_msi=n_msi, in_hsi=n_hsi,
        base_channel=config["model"].get("base_channel", 64)
    ).to(device).eval()
    ckpt = torch.load(eval_cfg["weights_path_reconstruction"], map_location=device, weights_only=False)
    model_rec.load_state_dict(ckpt.get("model_state_dict", ckpt))
    
    # Uncertainty
    model_unc = DualBranchNAFNet(
        n_msi=n_msi, n_hsi=n_hsi, out_channels=n_hsi,
        width=config["model_uncertainty"].get("base_channel", 64)
    ).to(device).eval()
    ckpt = torch.load(eval_cfg["weights_path_uncertainty"], map_location=device, weights_only=False)
    model_unc.load_state_dict(ckpt.get("model_state_dict", ckpt))
    
    # Siamese
    model_siamese = SiameseNetwork_heterogene(
        in_channel_msi=n_msi, in_channel_hsi=n_hsi,
        base_channels=config["model_siamese"].get("base_channels", 32),
        latent_dim=config["model_siamese"].get("latent_dim", 64)
    ).to(device).eval()
    ckpt = torch.load(eval_cfg["weights_path_siamese"], map_location=device, weights_only=False)
    model_siamese.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    
    print("✅ Models loaded\n")
    
    # Scene list
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = [
            "aranjuez", "arborea", "athens", "baltijsk", "bari", "beer_sheva",
            "beheira", "beirut", "belgrade", "binh_dai", "brasilia", "camerino",
            "cape_town", "codigoro", "copenhagen", "copperton", "istanbul"
        ]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process scenes
    for scene in tqdm(scenes, desc="Scenes"):
        scene_dir = Path(args.data_dir) / scene
        if not scene_dir.is_dir():
            continue
        
        hsi_before = scene_dir / f"{scene}-prs.nc"
        msi_after = scene_dir / f"{scene}-s2.nc"
        
        if not hsi_before.exists() or not msi_after.exists():
            print(f"⚠️ Missing files for {scene}")
            continue
        
        try:
            print(f"\n🎬 Processing {scene}...")
            
            hsi_dataset = HSIDataset(str(hsi_before), patch_size, n_hsi)
            msi_dataset = MSIDataset(str(msi_after), patch_size, n_msi)
            msi_interp_dataset = MSIInterpolatedDataset(str(msi_after), patch_size, n_hsi)
            
            hsi2_rec, unc, scores = infer_scene(
                msi_dataset, hsi_dataset, msi_interp_dataset,
                model_rec, model_unc, model_siamese, device,
                use_amp=eval_cfg.get("use_amp", True)
            )
            
            # Visualization
            scene_output = output_dir / scene
            scene_output.mkdir(exist_ok=True)
            
            # Convert to numpy arrays with right format
            hsi1_data = [hsi_dataset[i][0].numpy() for i in range(len(hsi_dataset))]
            msi_data = [msi_dataset[i][0].numpy() for i in range(len(msi_dataset))]
            
            visualize_results(
                hsi1_data, msi_data, hsi2_rec, unc, scores,
                scene_output / "change_detection_results.png"
            )
            
            print(f"✅ {scene} done")
            
        except Exception as e:
            print(f"❌ {scene}: {e}")
            continue
    
    print(f"\n🎉 Done! Results in {output_dir}")


