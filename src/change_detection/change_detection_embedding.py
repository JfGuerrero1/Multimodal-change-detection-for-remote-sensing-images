#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║         DETECTION DE CHANGEMENT MSI/HSI - DOUBLE MODELE SIAMOIS            ║ 
║  Comparaison de plusieurs modeles avec embeddings                          ║
║   Inference globale -> Calibration T1 (NFA) -> Application T2 -> Visu      ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import yaml
from tqdm import tqdm

matplotlib = __import__('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.constants import WVL_PRS, INTERP_MATRIX
from src.models.dual_branch import DualBranchNAFNet
from src.models.unet_residual import GradualExpansionUNet_residual
from src.models.contrastive import SiameseNetwork_heterogene


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT GLOBAL ET INTERPOLATION DES SCENES
# ════════════════════════════════════════════════════════════════════════════

def load_scene_cube(path, n_channels):
    """Charge un cube de donnees a partir d'un fichier NetCDF et limite le nombre de canaux."""
    with xr.open_dataset(path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        cube = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    return cube[:, :, :n_channels]


def get_interpolated_msi(msi_cube, n_channels_hsi=195):
    """Interpole un cube multispectral (MSI) vers l'espace hyperspectral (HSI) via la matrice de passage."""
    h, w, c_msi = msi_cube.shape
    msi_chw = np.transpose(msi_cube, (2, 0, 1))
    interp_numpy = INTERP_MATRIX @ msi_chw.reshape(c_msi, -1)
    c_hsi = interp_numpy.shape[0]
    interp_cube = np.transpose(interp_numpy.reshape(c_hsi, h, w), (1, 2, 0))
    return interp_cube[:, :, :n_channels_hsi]


def load_scene_data(scene_dir, scene, n_msi=12, n_hsi=195):
    """Charge l'ensemble des donnees pour une scene donnee (T1 before, T2 after, interpolations et verite terrain)."""
    hsi_before_path = scene_dir / f"{scene}-before-prs.nc"
    if not hsi_before_path.exists():
        candidates = list(scene_dir.glob("*-before-prs.nc")) + list(scene_dir.glob("*-prs.nc"))
        if candidates:
            hsi_before_path = candidates[0]
            
    msi_after_path = scene_dir / f"{scene}-after-s2.nc"
    if not msi_after_path.exists():
        candidates = list(scene_dir.glob("*-after-s2.nc")) + list(scene_dir.glob("*-s2.nc"))
        if candidates:
            msi_after_path = candidates[0]

    msi_before_path = scene_dir / f"{scene}-before-s2.nc"
    if not msi_before_path.exists():
        candidates = list(scene_dir.glob("*-before-s2.nc"))
        msi_before_path = candidates[0] if candidates else msi_after_path
            
    gt_path = scene_dir / f"{scene}-cd-binary.nc"
    if not gt_path.exists():
        candidates = list(scene_dir.glob("*-cd-binary.nc"))
        if candidates:
            gt_path = candidates[0]
            
    if not hsi_before_path.exists() or not msi_after_path.exists():
        raise FileNotFoundError(f"Fichiers HSI ou MSI introuvables dans {scene_dir}")

    # Chargement du cube HSI T1 (Before)
    with xr.open_dataset(hsi_before_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        hsi_before = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if hsi_before.ndim == 3 and hsi_before.shape[2] > n_hsi:
        hsi_before = hsi_before[:, :, :n_hsi]
    
    # Chargement du cube MSI T2 (After)
    with xr.open_dataset(msi_after_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        msi_after = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if msi_after.ndim == 3 and msi_after.shape[2] > n_msi:
        msi_after = msi_after[:, :, :n_msi]

    # Chargement du cube MSI T1 (Before)
    with xr.open_dataset(msi_before_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        msi_before = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if msi_before.ndim == 3 and msi_before.shape[2] > n_msi:
        msi_before = msi_before[:, :, :n_msi]
    
    # Harmonisation des dimensions spatiales minimales entre les differents capteurs
    h_min = min(hsi_before.shape[0], msi_after.shape[0], msi_before.shape[0])
    w_min = min(hsi_before.shape[1], msi_after.shape[1], msi_before.shape[1])
    hsi_before = hsi_before[:h_min, :w_min, :]
    msi_after = msi_after[:h_min, :w_min, :]
    msi_before = msi_before[:h_min, :w_min, :]
    
    # Generation des interpolations MSI vers HSI
    def interpolate_msi(msi_cube):
        msi_chw = np.transpose(msi_cube, (2, 0, 1))
        c_msi, h, w = msi_chw.shape
        interp_numpy = INTERP_MATRIX @ msi_chw.reshape(c_msi, -1)
        c_hsi = interp_numpy.shape[0]
        return np.transpose(interp_numpy.reshape(c_hsi, h, w), (1, 2, 0))[:, :, :n_hsi]

    msi_interp_after = interpolate_msi(msi_after)
    msi_interp_before = interpolate_msi(msi_before)
    
    # Chargement optionnel du masque de verite terrain (Ground Truth)
    gt_mask = None
    if gt_path.exists():
        with xr.open_dataset(gt_path) as ds:
            gt = np.nan_to_num(ds["change"].values, nan=0.0).astype(np.float32)
        if gt.ndim > 2:
            gt = gt.squeeze()
        gt_mask = gt[:h_min, :w_min]
        
    return msi_before, msi_after, hsi_before, msi_interp_before, msi_interp_after, gt_mask


# ════════════════════════════════════════════════════════════════════════════
# METHODES DE DETECTION ET COMPARAISON D'EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════

def compute_embedding_score(z_1, z_2):
    """Calcule la distance L2 brute entre les embeddings apres projection sur la sphere unitaire."""
    if z_1 is None or z_2 is None:
        return None

    if isinstance(z_1, torch.Tensor):
        z_1_norm = F.normalize(z_1, p=2, dim=1)
        z_2_norm = F.normalize(z_2, p=2, dim=1)

        if z_1_norm.dim() == 4:
            score = torch.norm(z_1_norm - z_2_norm, p=2, dim=1, keepdim=True)
            score = score.squeeze(0).squeeze(0).cpu().numpy()
        else:
            score = torch.norm(z_1_norm - z_2_norm, p=2, dim=0).cpu().numpy()
    else:
        if z_1.ndim == 4:
            z_1 = z_1.squeeze(0)
            z_2 = z_2.squeeze(0)
        
        eps = 1e-8
        z_1_norm = z_1 / (np.linalg.norm(z_1, axis=0, keepdims=True) + eps)
        z_2_norm = z_2 / (np.linalg.norm(z_2, axis=0, keepdims=True) + eps)
        
        score = np.linalg.norm(z_1_norm - z_2_norm, axis=0)

    return np.squeeze(score)


# ════════════════════════════════════════════════════════════════════════════
# SEUIL DYNAMIQUE ET NFA (Number of False Alarms)
# ════════════════════════════════════════════════════════════════════════════

def compute_dynamic_nfa_threshold(map_data, nfa_target=0.01):
    """Calcule le seuil dynamique base sur le percentile cible de la carte de score."""
    valid_values = map_data[np.isfinite(map_data)]
    if len(valid_values) == 0:
        return 0.0
    percentile = (1.0 - nfa_target) * 100.0
    return float(np.percentile(valid_values, percentile))


def find_optimal_threshold_nfa(map_data, target_nfa=0.01):
    """determine le seuil optimal pour la carte de score en respectant le taux de fausses alarmes."""
    threshold = compute_dynamic_nfa_threshold(map_data, nfa_target=target_nfa)
    return threshold, 0.0


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE GLOBALE AVEC DEUX MODELES SIAMOIS ET RECONSTRUCTION HSI (z_hsi_rec)
# ════════════════════════════════════════════════════════════════════════════

def infer_scene_global_dual_siamese(msi_cube, hsi_cube, msi_interp_cube,
                                   model_rec, model_unc, model_siamese_1, model_siamese_2, 
                                   device, use_rec_unc_model1=False, use_amp=True):
    """Execute l'inference globale pour la reconstruction, l'incertitude et l'extraction des embeddings z_msi, z_hsi et z_hsi_rec."""
    model_rec.eval()
    model_unc.eval()
    if model_siamese_1: model_siamese_1.eval()
    if model_siamese_2: model_siamese_2.eval()
    
    msi_t = torch.from_numpy(np.transpose(msi_cube, (2, 0, 1))).unsqueeze(0).float().to(device)
    hsi_t = torch.from_numpy(np.transpose(hsi_cube, (2, 0, 1))).unsqueeze(0).float().to(device)
    msi_interp_t = torch.from_numpy(np.transpose(msi_interp_cube, (2, 0, 1))).unsqueeze(0).float().to(device)
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
            hsi_rec = model_rec(msi_t, msi_interp_t)
            unc_input = torch.cat([msi_t, hsi_rec], dim=1)
            
            unc_output = model_unc(unc_input)
            if isinstance(unc_output, tuple):
                unc = unc_output[0]
            else:
                unc = unc_output
            
            if use_rec_unc_model1:
                msi_input_1 = torch.cat([msi_t, hsi_rec, unc], dim=1)
            else:
                msi_input_1 = msi_t
                
            # Modele siamois 1
            z_msi_1, z_hsi_1 = (model_siamese_1(msi_input_1, hsi_t) if model_siamese_1 else (None, None))
            _, z_hsi_1_rec = (model_siamese_1(msi_input_1, hsi_rec) if model_siamese_1 else (None, None))

            # Modele siamois 2
            z_msi_2, z_hsi_2 = (model_siamese_2(msi_t, hsi_t) if model_siamese_2 else (None, None))
            _, z_hsi_2_rec = (model_siamese_2(msi_t, hsi_rec) if model_siamese_2 else (None, None))
            
    return {
        'hsi_rec': hsi_rec.squeeze(0).cpu().numpy(),
        'uncertainty': unc.squeeze(0).cpu().numpy(),
        'z_msi_1': z_msi_1.squeeze(0).cpu().numpy() if z_msi_1 is not None else None,
        'z_hsi_1': z_hsi_1.squeeze(0).cpu().numpy() if z_hsi_1 is not None else None,
        'z_hsi_1_rec': z_hsi_1_rec.squeeze(0).cpu().numpy() if z_hsi_1_rec is not None else None,
        'z_msi_2': z_msi_2.squeeze(0).cpu().numpy() if z_msi_2 is not None else None,
        'z_hsi_2': z_hsi_2.squeeze(0).cpu().numpy() if z_hsi_2 is not None else None,
        'z_hsi_2_rec': z_hsi_2_rec.squeeze(0).cpu().numpy() if z_hsi_2_rec is not None else None,
    }


# ════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS (GLOBALE ET PAR PATCH)
# ════════════════════════════════════════════════════════════════════════════

def visualize_all_methods_global(results_dict, output_path, calibrated_thresholds=None):
    """Genere une vue comparative globale des cartes de changement."""
    n_methods = len(results_dict)
    ncols = 2
    nrows = (n_methods + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (name, detection_map) in enumerate(results_dict.items()):
        im = axes[idx].imshow(detection_map, cmap='viridis', vmin=0, vmax=1)
        
        if calibrated_thresholds and name in calibrated_thresholds:
            threshold = calibrated_thresholds[name]['threshold']
            title_text = f"{name}\nThresh(T1): {threshold:.3f}"
        else:
            threshold, _ = find_optimal_threshold_nfa(detection_map)
            title_text = f"{name}\nThresh: {threshold:.3f}"

        axes[idx].set_title(title_text, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Vue comparative globale sauvegardee : {output_path}")


def extract_patch(img, y, x, size):
    """Extrait un patch spatial d'une image ou d'un cube multidimensionnel."""
    if img is None:
        return None
    if img.ndim == 3:
        if img.shape[0] < img.shape[2]:
            return img[:, y:y+size, x:x+size]
        else:
            return img[y:y+size, x:x+size, :]
    else:
        return img[y:y+size, x:x+size]


def save_patch_visualization(
    p_msi, p_hsi_bef, p_hsi_rec, p_unc, p_methods_dict, gt_patch, patch_out_path, calibrated_thresholds, target_nfa=1.0
):
    """Sauvegarde la visualisation detaillee pour un patch donne."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    idx_hsi = [30, 20, 10]
    idx_msi = [3, 2, 1]
    
    def safe_to_rgb(img, indices):
        if img is None:
            return np.zeros((100, 100, 3))
        if img.ndim == 3 and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] >= max(indices):
            rgb = img[:, :, indices]
        else:
            rgb = img[:, :, :3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return rgb

    rgb_msi = safe_to_rgb(p_msi, idx_msi)
    rgb_rec = safe_to_rgb(p_hsi_rec, idx_hsi)
    rgb_hsi = safe_to_rgb(p_hsi_bef, idx_hsi)
    
    if p_unc is not None:
        if p_unc.ndim == 3 and p_unc.shape[0] < p_unc.shape[2]:
            p_unc = np.transpose(p_unc, (1, 2, 0))
        min_c = min(p_hsi_bef.shape[-1] if p_hsi_bef.ndim==3 else 1, p_unc.shape[-1])
        unc_map = np.mean(p_unc[:, :, :min_c], axis=-1)
    else:
        unc_map = np.zeros((256, 256))

    panels = list(p_methods_dict.items())
    m1_name, m1_map = panels[0] if len(panels) > 0 else ('Score 1', np.zeros((256, 256)))
    m2_name, m2_map = panels[1] if len(panels) > 1 else ('Score 2', np.zeros((256, 256)))

    t_m1 = calibrated_thresholds.get(m1_name, {}).get('threshold', 0.5)
    t_m2 = calibrated_thresholds.get(m2_name, {}).get('threshold', 0.5)

    # Ligne 1 : Images RGB
    axes[0, 0].imshow(rgb_msi)
    axes[0, 0].set_title("MSI After (T2)", fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_rec)
    axes[0, 1].set_title("HSI Reconstructed (T2)", fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgb_hsi)
    axes[0, 2].set_title("HSI Before (T1)", fontweight='bold')
    axes[0, 2].axis('off')
    
    # Ligne 2 : Cartes continues (Incertitude et Scores)
    im1 = axes[1, 0].imshow(unc_map, cmap='inferno')
    axes[1, 0].set_title("Uncertainty Map", fontweight='bold')
    axes[1, 0].axis('off')
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(m1_map, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title("Embedding Score (Model 1)", fontweight='bold', fontsize=9)
    axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 2].imshow(m2_map, cmap='jet', vmin=0, vmax=1)
    axes[1, 2].set_title("Embedding Score (Model 2)", fontweight='bold', fontsize=9)
    axes[1, 2].axis('off')
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Ligne 3 : Cartes binarisees et verite terrain
    axes[2, 0].imshow(m1_map > t_m1, cmap='gray')
    axes[2, 0].set_title(f"Change Map (Model 1)\nThresh={t_m1:.3f}", fontweight='bold', fontsize=9)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(m2_map > t_m2, cmap='gray')
    axes[2, 1].set_title(f"Change Map (Model 2)\nThresh={t_m2:.3f}", fontweight='bold', fontsize=9)
    axes[2, 1].axis('off')
    
    if gt_patch is not None:
        axes[2, 2].imshow(gt_patch, cmap='gray')
        axes[2, 2].set_title("Ground Truth (Patch)\n(Blanc = Changement)", fontweight='bold', fontsize=9)
    else:
        axes[2, 2].text(0.5, 0.5, "No GT Available", ha='center', va='center')
        axes[2, 2].set_title("Ground Truth", fontweight='bold', fontsize=9)
    axes[2, 2].axis('off')
    
    plt.suptitle(f"Patch Change Detection - Dual Siamese (Target NFA = {target_nfa*100}%)", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(patch_out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE (MAIN)
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Detection de Changement - Double Modele Siamois & Visu")
    parser.add_argument("--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="results_dual_siamese_comparison")
    parser.add_argument("--scenes", type=str, nargs='+', default=None)
    parser.add_argument("--target_nfa", type=float, default=1.0)
    parser.add_argument("--patch_size", type=int, default=256)
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    n_msi = config["data"].get("n_msi", 12)
    n_hsi = config["data"].get("n_hsi", 195)
    patch_size = args.patch_size
    
    siamese_cfg = config.get("model_siamese", {})
    use_rec_unc_model1 = siamese_cfg.get("use_rec_unc_model1", False)
    
    print("Loading models (Reconstruction, Uncertainty, 2 Siamese Networks)...")
    print(f"Option [MSI + HSI_rec + UNC] pour le Modele 1 : {use_rec_unc_model1}")
    
    eval_cfg = config.get("evaluation", {})
    
    model_rec = GradualExpansionUNet_residual(
        in_msi=n_msi, in_hsi=n_hsi, base_channel=config["model"].get("base_channel", 64)
    ).to(device).eval()
    ckpt = torch.load(eval_cfg["weights_path_reconstruction"], map_location=device, weights_only=False)
    model_rec.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    
    model_unc = DualBranchNAFNet(
        n_msi=n_msi, n_hsi=n_hsi, out_channels=n_hsi, width=config["model_uncertainty"].get("base_channel", 64)
    ).to(device).eval()
    ckpt = torch.load(eval_cfg["weights_path_uncertainty"], map_location=device, weights_only=False)
    model_unc.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    
    in_channels_msi_s1 = (n_msi + 2 * n_hsi) if use_rec_unc_model1 else n_msi
    
    model_siamese_1 = SiameseNetwork_heterogene(
        in_channel_msi=in_channels_msi_s1, in_channel_hsi=n_hsi,
        base_channels=siamese_cfg.get("base_channels", 32),
        latent_dim=siamese_cfg.get("latent_dim", 64)
    ).to(device).eval()
    path_s1 = eval_cfg.get("weights_path_siamese_1", eval_cfg.get("weights_path_siamese"))
    ckpt1 = torch.load(path_s1, map_location=device, weights_only=False)
    model_siamese_1.load_state_dict(ckpt1.get("model_state_dict", ckpt1), strict=False)
    
    path_s2 = eval_cfg.get("weights_path_siamese_2", eval_cfg.get("weights_path_siamese"))
    model_siamese_2 = None
    if path_s2:
        model_siamese_2 = SiameseNetwork_heterogene(
            in_channel_msi=n_msi, in_channel_hsi=n_hsi,
            base_channels=siamese_cfg.get("base_channels", 32),
            latent_dim=siamese_cfg.get("latent_dim", 64)
        ).to(device).eval()
        ckpt2 = torch.load(path_s2, map_location=device, weights_only=False)
        model_siamese_2.load_state_dict(ckpt2.get("model_state_dict", ckpt2), strict=False)
        
    print("All models loaded successfully\n")
    
    scenes = args.scenes if args.scenes else ["aranjuez", "athens", "beirut", "belgrade", "paris", "shanghai"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scene in tqdm(scenes, desc="Scenes"):
        scene_dir = Path(args.data_dir) / scene
        if not scene_dir.is_dir():
            print(f"Dossier introuvable pour la scene : {scene_dir}")
            continue
        
        try:
            print(f"\nTraitement de la scene {scene}...")
            
            # Chargement des donnees T1, T2 et verite terrain
            msi_before_cube, msi_after_cube, hsi_before_cube, msi_interp_before, msi_interp_after, gt_mask = load_scene_data(
                scene_dir, scene, n_msi=n_msi, n_hsi=n_hsi
            )
            
            # Calibration sur T1 pour trouver les seuils optimaux
            print(f"  -> [{scene}] Calibration sur T1...")
            inf_t1 = infer_scene_global_dual_siamese(
                msi_before_cube, hsi_before_cube, msi_interp_before,
                model_rec, model_unc, model_siamese_1, model_siamese_2,
                device, use_rec_unc_model1=use_rec_unc_model1
            )
            
            # Utilisation de z_hsi_1_rec / z_hsi_2_rec pour comparer l'embedding reconstruit sur T1
            score_1_t1 = compute_embedding_score(inf_t1['z_msi_1'], inf_t1['z_hsi_1_rec'])
            score_2_t1 = compute_embedding_score(inf_t1['z_msi_2'], inf_t1['z_hsi_2_rec']) if model_siamese_2 else None
            
            t_m1, _ = find_optimal_threshold_nfa(score_1_t1, target_nfa=args.target_nfa)
            t_m2, _ = find_optimal_threshold_nfa(score_2_t1, target_nfa=args.target_nfa) if score_2_t1 is not None else (0.5, 0.0)
            
            calibrated_thresholds = {
                'Embedding Score (Model 1)': {'threshold': t_m1},
            }
            if score_2_t1 is not None:
                calibrated_thresholds['Embedding Score (Model 2)'] = {'threshold': t_m2}
                
            # Application sur T2
            print(f"  -> [{scene}] Application sur T2...")
            inf_t2 = infer_scene_global_dual_siamese(
                msi_after_cube, hsi_before_cube, msi_interp_after,
                model_rec, model_unc, model_siamese_1, model_siamese_2,
                device, use_rec_unc_model1=use_rec_unc_model1
            )
            
            # Comparaison entre z_msi_1 de T2 et z_hsi_1 de T1 (reference initiale)
            score_1_t2 = compute_embedding_score(inf_t2['z_msi_1'], inf_t1['z_hsi_1'])
            score_2_t2 = compute_embedding_score(inf_t2['z_msi_2'], inf_t1['z_hsi_2']) if model_siamese_2 else None
            
            all_methods_results = {
                'Embedding Score (Model 1)': score_1_t2
            }
            if score_2_t2 is not None:
                all_methods_results['Embedding Score (Model 2)'] = score_2_t2
                
            scene_output = output_dir / scene
            scene_output.mkdir(parents=True, exist_ok=True)
            
            # Visualisation globale
            visualize_all_methods_global(all_methods_results, scene_output / "all_methods_comparison_global.png", calibrated_thresholds)
            
            # Decoupage et visualisation par patchs
            print(f"  -> [Patch Slicing] Generation des visualisations par patch ({patch_size}x{patch_size})...")
            patches_vis_dir = scene_output / "patches_visualizations"
            patches_vis_dir.mkdir(parents=True, exist_ok=True)
            
            h, w = msi_after_cube.shape[:2]
            patch_count = 0
            
            for y in range(0, h - (h % patch_size), patch_size):
                for x in range(0, w - (w % patch_size), patch_size):
                    p_msi = extract_patch(msi_after_cube, y, x, patch_size)
                    p_hsi_bef = extract_patch(hsi_before_cube, y, x, patch_size)
                    p_hsi_rec = extract_patch(inf_t2['hsi_rec'], y, x, patch_size)
                    p_unc = extract_patch(inf_t2['uncertainty'], y, x, patch_size)
                    p_gt = extract_patch(gt_mask, y, x, patch_size) if gt_mask is not None else None
                    
                    p_methods = {
                        'Embedding Score (Model 1)': extract_patch(score_1_t2, y, x, patch_size)
                    }
                    if score_2_t2 is not None:
                        p_methods['Embedding Score (Model 2)'] = extract_patch(score_2_t2, y, x, patch_size)
                        
                    patch_out_path = patches_vis_dir / f"patch_y{y}_x{x}.png"
                    save_patch_visualization(
                        p_msi, p_hsi_bef, p_hsi_rec, p_unc,
                        p_methods, p_gt, patch_out_path,
                        calibrated_thresholds, target_nfa=args.target_nfa
                    )
                    patch_count += 1
            
            print(f"    -> {patch_count} patchs sauvegardes dans : {patches_vis_dir}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de la scene {scene}: {e}")
            import traceback
            traceback.print_exc()

    print("\nTraitement de toutes les scenes termine avec succes !")

if __name__ == "__main__":
    main()