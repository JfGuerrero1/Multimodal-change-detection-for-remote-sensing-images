"""
╔════════════════════════════════════════════════════════════════════════════╗
║         DÉTECTION DE CHANGEMENT MSI/HSI (Calibration T1 & Percentile)      ║
║                                                                            ║
║  1. Lecture automatique des scènes et poids depuis le YAML                 ║
║  2. Calibration du seuil (via Percentile) sur T1 (Before)                  ║
║  3. Application sur T2 (After) et visualisation globale de la scène        ║
║  4. Découpage en patchs et sauvegarde de la visualisation par patch        ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
from tqdm import tqdm

matplotlib = __import__('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.constants import INTERP_MATRIX, WVL_PRS
from src.models.dual_branch import DualBranchNAFNet
from src.models.unet_residual import GradualExpansionUNet_residual


# ════════════════════════════════════════════════════════════════════════════
# 📊 UTILITAIRES & MÉTRIQUES & PATCHS
# ════════════════════════════════════════════════════════════════════════════

def compute_dynamic_nfa_threshold(map_data, nfa_target=0.01):
    """Calcule le seuil dynamique basé sur le percentile cible (ex: 1% -> 99e percentile)"""
    valid_values = map_data[np.isfinite(map_data)]
    if len(valid_values) == 0:
        return 0.0
    percentile = (1.0 - nfa_target) * 100.0
    return float(np.percentile(valid_values, percentile))


def compute_metrics(score_map, gt_mask, threshold):
    """Calcule Precision, Recall et F1-score par rapport au Ground Truth"""
    if gt_mask is None:
        return 0.0, 0.0, 0.0
    
    pred_binary = (score_map > threshold).astype(bool)
    gt_binary = (gt_mask > 0.5) if gt_mask.max() > 1.0 else gt_mask.astype(bool)
    
    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & (~gt_binary))
    fn = np.sum((~pred_binary) & gt_binary)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1


def extract_patch(array_3d, i, j, patch_size=256):
    """Extrait un patch d'un tableau 2D ou 3D en gérant les bords"""
    if array_3d is None:
        return None
    h, w = array_3d.shape[:2]
    h_end = min(i + patch_size, h)
    w_end = min(j + patch_size, w)
    if array_3d.ndim == 3:
        return array_3d[i:h_end, j:w_end, :]
    else:
        return array_3d[i:h_end, j:w_end]


# ════════════════════════════════════════════════════════════════════════════
# 🏗️ CONSTRUCTION MODÈLES
# ════════════════════════════════════════════════════════════════════════════

def build_models(config, device):
    """Charge les modèles de reconstruction et incertitude depuis les chemins du YAML"""
    eval_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})
    n_msi = data_cfg.get("n_msi", 12)
    n_hsi = data_cfg.get("n_hsi", 195)
    
    model_rec = GradualExpansionUNet_residual(
        in_msi=n_msi,
        in_hsi=n_hsi,
        base_channel=config["model"].get("base_channel", 64)
    ).to(device).eval()
    
    ckpt_rec = torch.load(eval_cfg["weights_path_reconstruction"], map_location=device, weights_only=False)
    model_rec.load_state_dict(ckpt_rec.get("model_state_dict", ckpt_rec), strict=False)
    
    model_unc = DualBranchNAFNet(
        n_msi=n_msi,
        n_hsi=n_hsi,
        out_channels=n_hsi,
        width=config["model_uncertainty"].get("base_channel", 64)
    ).to(device).eval()
    
    ckpt_unc = torch.load(eval_cfg["weights_path_uncertainty"], map_location=device, weights_only=False)
    model_unc.load_state_dict(ckpt_unc.get("model_state_dict", ckpt_unc), strict=False)
    
    return model_rec, model_unc


# ════════════════════════════════════════════════════════════════════════════
# 📥 CHARGEMENT DONNÉES SCÈNE (T1 & T2)
# ════════════════════════════════════════════════════════════════════════════

def load_scene_data(scene_dir, scene, n_msi=12, n_hsi=195):
    """Charge T1 (Before) et T2 (After) ainsi que le masque GT."""
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

    # 1. Chargement HSI Before (T1)
    with xr.open_dataset(hsi_before_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        hsi_before = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if hsi_before.ndim == 3 and hsi_before.shape[2] > n_hsi:
        hsi_before = hsi_before[:, :, :n_hsi]
    
    # 2. Chargement MSI After (T2)
    with xr.open_dataset(msi_after_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        msi_after = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if msi_after.ndim == 3 and msi_after.shape[2] > n_msi:
        msi_after = msi_after[:, :, :n_msi]

    # 3. Chargement MSI Before (T1)
    with xr.open_dataset(msi_before_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        msi_before = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if msi_before.ndim == 3 and msi_before.shape[2] > n_msi:
        msi_before = msi_before[:, :, :n_msi]
    
    # Harmonisation des dimensions spatiales
    h_min = min(hsi_before.shape[0], msi_after.shape[0], msi_before.shape[0])
    w_min = min(hsi_before.shape[1], msi_after.shape[1], msi_before.shape[1])
    hsi_before = hsi_before[:h_min, :w_min, :]
    msi_after = msi_after[:h_min, :w_min, :]
    msi_before = msi_before[:h_min, :w_min, :]
    
    # Interpolations MSI -> HSI
    def interpolate_msi(msi_cube):
        msi_chw = np.transpose(msi_cube, (2, 0, 1))
        c_msi, h, w = msi_chw.shape
        interp_numpy = INTERP_MATRIX @ msi_chw.reshape(c_msi, -1)
        c_hsi = interp_numpy.shape[0]
        return np.transpose(interp_numpy.reshape(c_hsi, h, w), (1, 2, 0))[:, :, :n_hsi]

    msi_interp_after = interpolate_msi(msi_after)
    msi_interp_before = interpolate_msi(msi_before)
    
    # Chargement Ground Truth
    gt_mask = None
    if gt_path.exists():
        with xr.open_dataset(gt_path) as ds:
            gt = np.nan_to_num(ds["change"].values, nan=0.0).astype(np.float32)
        if gt.ndim > 2:
            gt = gt.squeeze()
        gt_mask = gt[:h_min, :w_min]
        
    return msi_before, msi_after, hsi_before, msi_interp_before, msi_interp_after, gt_mask


# ════════════════════════════════════════════════════════════════════════════
# 🧠 INFÉRENCE GLOBALE
# ════════════════════════════════════════════════════════════════════════════

def infer_scene(msi_cube, msi_interp, model_rec, model_unc, device):
    msi_t = torch.from_numpy(np.transpose(msi_cube, (2, 0, 1))).unsqueeze(0).float().to(device)
    interp_t = torch.from_numpy(np.transpose(msi_interp, (2, 0, 1))).unsqueeze(0).float().to(device)
    
    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=True):
            hsi_rec = model_rec(msi_t, interp_t)
            unc = model_unc(torch.cat([msi_t, hsi_rec], dim=1))
            if isinstance(unc, tuple):
                unc = unc[0]
    return {
        'hsi_rec': hsi_rec.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        'uncertainty': unc.squeeze(0).permute(1, 2, 0).cpu().numpy()
    }


# ════════════════════════════════════════════════════════════════════════════
# 🎨 VISUALISATION GLOBALE & PAR PATCH
# ════════════════════════════════════════════════════════════════════════════

def get_rgb_indices(wvl=WVL_PRS):
    if wvl is None:
        return [0, 1, 2]
    return [
        np.argmin(np.abs(wvl - 665.0)),
        np.argmin(np.abs(wvl - 560.0)),
        np.argmin(np.abs(wvl - 490.0)),
    ]


def to_rgb(cube, indices):
    valid_idx = [i for i in indices if i < cube.shape[-1]]
    rgb = cube[:, :, valid_idx] if len(valid_idx) == 3 else cube[:, :, :3]
    for i in range(rgb.shape[-1]):
        v_min = np.percentile(rgb[:, :, i], 2)
        v_max = np.percentile(rgb[:, :, i], 98)
        rgb[:, :, i] = np.clip((rgb[:, :, i] - v_min) / (v_max - v_min + 1e-8), 0, 1)
    return rgb


def get_confusion_rgb(score_map, gt_mask, threshold):
    """Génère la matrice de confusion : Blanc (TP), Vert (FP), Magenta (FN)"""
    h, w = score_map.shape
    pred_binary = (score_map > threshold).astype(bool)
    
    if gt_mask is None:
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[pred_binary] = [1.0, 1.0, 1.0]
        return rgb
    
    gt_binary = (gt_mask > 0.5) if gt_mask.max() > 1.0 else gt_mask.astype(bool)
    
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    tp = pred_binary & gt_binary       # Vrai Positif -> Blanc
    fp = pred_binary & (~gt_binary)    # Faux Positif -> Vert
    fn = (~pred_binary) & gt_binary    # Faux Négatif -> Magenta
    
    rgb[tp] = [1.0, 1.0, 1.0]
    rgb[fp] = [0.0, 1.0, 0.0]
    rgb[fn] = [1.0, 0.0, 1.0]
    return rgb


def save_scene_visualization(
    msi_after, hsi_before, hsi_rec_t2, uncertainty_t2,
    method_rec_t2, method_weighted_t2, gt_mask, output_path,
    t_rec, t_weighted, target_nfa=0.01
):
    """Sauvegarde la visualisation globale de toute la scène"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    idx_hsi = get_rgb_indices()
    idx_msi = [3, 2, 1]
    
    rgb_msi = to_rgb(msi_after, idx_msi)
    rgb_rec = to_rgb(hsi_rec_t2, idx_hsi)
    rgb_hsi = to_rgb(hsi_before, idx_hsi)
    
    min_c = min(hsi_before.shape[-1], hsi_rec_t2.shape[-1])
    unc_map = np.mean(uncertainty_t2[:, :, :min_c], axis=-1)
    
    conf_rec = get_confusion_rgb(method_rec_t2, gt_mask, t_rec)
    conf_weighted = get_confusion_rgb(method_weighted_t2, gt_mask, t_weighted)
    
    prec_rec, rec_rec, f1_rec = compute_metrics(method_rec_t2, gt_mask, t_rec)
    prec_w, rec_w, f1_w = compute_metrics(method_weighted_t2, gt_mask, t_weighted)
    
    # LIGNE 1
    axes[0, 0].imshow(rgb_msi)
    axes[0, 0].set_title("MSI After (T2) - Global", fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_rec)
    axes[0, 1].set_title("HSI Reconstructed (T2) - Global", fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgb_hsi)
    axes[0, 2].set_title("HSI Before (T1) - Global", fontweight='bold')
    axes[0, 2].axis('off')
    
    # LIGNE 2
    im1 = axes[1, 0].imshow(method_rec_t2, cmap='inferno')
    axes[1, 0].set_title("T2 Rec. Error Map (E)", fontweight='bold')
    axes[1, 0].axis('off')
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(unc_map, cmap='inferno')
    axes[1, 1].set_title("T2 Uncertainty Map", fontweight='bold')
    axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 2].imshow(method_weighted_t2, cmap='inferno')
    axes[1, 2].set_title("T2 Weighted Map (E / Unc)", fontweight='bold')
    axes[1, 2].axis('off')
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # LIGNE 3
    title_rec = f"Rec Error | F1={f1_rec:.3f} (P={prec_rec:.3f}, R={rec_rec:.3f})"
    axes[2, 0].imshow(conf_rec)
    axes[2, 0].set_title(title_rec, fontweight='bold', fontsize=9)
    axes[2, 0].axis('off')
    
    title_weighted = f"Weighted | F1={f1_w:.3f} (P={prec_w:.3f}, R={rec_w:.3f})"
    axes[2, 1].imshow(conf_weighted)
    axes[2, 1].set_title(title_weighted, fontweight='bold', fontsize=9)
    axes[2, 1].axis('off')
    
    if gt_mask is not None:
        axes[2, 2].imshow(gt_mask, cmap='gray')
        axes[2, 2].set_title("Ground Truth (Global)\n(Blanc=Change)", fontweight='bold', fontsize=9)
    else:
        axes[2, 2].text(0.5, 0.5, "No GT Available", ha='center', va='center')
        axes[2, 2].set_title("Ground Truth", fontweight='bold', fontsize=9)
    axes[2, 2].axis('off')
    
    plt.suptitle(f"Global Scene Change Detection (Target Rate={target_nfa*100}%)", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_patch_visualization(
    msi_after_patch, hsi_before_patch, hsi_rec_patch, uncertainty_patch,
    method_rec_patch, method_weighted_patch, gt_patch, output_path,
    t_rec, t_weighted, target_nfa=0.01
):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    idx_hsi = get_rgb_indices()
    idx_msi = [3, 2, 1]
    
    rgb_msi = to_rgb(msi_after_patch, idx_msi)
    rgb_rec = to_rgb(hsi_rec_patch, idx_hsi)
    rgb_hsi = to_rgb(hsi_before_patch, idx_hsi)
    
    min_c = min(hsi_before_patch.shape[-1], hsi_rec_patch.shape[-1])
    unc_map = np.mean(uncertainty_patch[:, :, :min_c], axis=-1)
    
    conf_rec = get_confusion_rgb(method_rec_patch, gt_patch, t_rec)
    conf_weighted = get_confusion_rgb(method_weighted_patch, gt_patch, t_weighted)
    
    prec_rec, rec_rec, f1_rec = compute_metrics(method_rec_patch, gt_patch, t_rec)
    prec_w, rec_w, f1_w = compute_metrics(method_weighted_patch, gt_patch, t_weighted)
    
    # LIGNE 1
    axes[0, 0].imshow(rgb_msi)
    axes[0, 0].set_title("MSI After (T2)", fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_rec)
    axes[0, 1].set_title("HSI Reconstructed (T2)", fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgb_hsi)
    axes[0, 2].set_title("HSI Before (T1)", fontweight='bold')
    axes[0, 2].axis('off')
    
    # LIGNE 2
    im1 = axes[1, 0].imshow(method_rec_patch, cmap='inferno')
    axes[1, 0].set_title("T2 Rec. Error Map (E)", fontweight='bold')
    axes[1, 0].axis('off')
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(unc_map, cmap='inferno')
    axes[1, 1].set_title("T2 Uncertainty Map", fontweight='bold')
    axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 2].imshow(method_weighted_patch, cmap='inferno')
    axes[1, 2].set_title("T2 Weighted Map (E / Unc)", fontweight='bold')
    axes[1, 2].axis('off')
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # LIGNE 3
    title_rec = f"Rec Error | F1={f1_rec:.3f} (P={prec_rec:.3f}, R={rec_rec:.3f})"
    axes[2, 0].imshow(conf_rec)
    axes[2, 0].set_title(title_rec, fontweight='bold', fontsize=9)
    axes[2, 0].axis('off')
    
    title_weighted = f"Weighted | F1={f1_w:.3f} (P={prec_w:.3f}, R={rec_w:.3f})"
    axes[2, 1].imshow(conf_weighted)
    axes[2, 1].set_title(title_weighted, fontweight='bold', fontsize=9)
    axes[2, 1].axis('off')
    
    if gt_patch is not None:
        axes[2, 2].imshow(gt_patch, cmap='gray')
        axes[2, 2].set_title("Ground Truth (Patch)\n(Blanc=Change)", fontweight='bold', fontsize=9)
    else:
        axes[2, 2].text(0.5, 0.5, "No GT Available", ha='center', va='center')
        axes[2, 2].set_title("Ground Truth", fontweight='bold', fontsize=9)
    axes[2, 2].axis('off')
    
    plt.suptitle(f"Patch Change Detection (Target Rate={target_nfa*100}%)", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Détection de Changement MSI/HSI par Patchs")
    parser.add_argument("--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--output_dir", type=str, default="results_change_detection")
    parser.add_argument("--scenes", type=str, nargs='+', default=None, help="Scènes spécifiques (optionnel)")
    parser.add_argument("--target_nfa", type=float, default=0.01, help="Valeur cible du taux (ex: 0.01 ou 0.05)")
    parser.add_argument("--patch_size", type=int, default=256, help="Taille des patchs (défaut: 256)")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}\n")
    
    data_cfg = config.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "data/mumucd"))
    n_msi = data_cfg.get("n_msi", 12)
    n_hsi = data_cfg.get("n_hsi", 195)
    
    scenes = args.scenes if args.scenes else data_cfg.get("scenes", [])
    scenes = [s for s in scenes if s != "urls.txt"]
    
    print(f"⚙️ Dossier de données : {data_dir}")
    print(f"⚙️ Nombre de scènes à traiter : {len(scenes)}")
    print(f"⚙️ Taux cible (Percentile) : {args.target_nfa * 100}%")
    print(f"⚙️ Taille des patchs : {args.patch_size}x{args.patch_size}\n")

    print("🏗️ Loading models...")
    model_rec, model_unc = build_models(config, device)
    print("✅ Models loaded\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_dir = data_dir / scene
        if not scene_dir.is_dir():
            print(f"⚠️ Scene directory not found: {scene_dir}")
            continue
        
        try:
            print(f"\n🎬 Traitement de la scène : {scene}...")
            
            # 1. Chargement des données T1 et T2
            msi_before, msi_after, hsi_before, msi_interp_before, msi_interp_after, gt_mask = load_scene_data(
                scene_dir, scene, n_msi, n_hsi
            )
            
            # -----------------------------------------------------------------
            # ÉTAPE A : CALIBRATION DU SEUIL SUR LA SCÈNE T1 (Before)
            # -----------------------------------------------------------------
            print("  ↳ [T1 Calibration] Inférence sur T1 (Before)...")
            inf_res_t1 = infer_scene(msi_before, msi_interp_before, model_rec, model_unc, device)
            hsi_rec_t1 = inf_res_t1['hsi_rec']
            uncertainty_t1 = inf_res_t1['uncertainty']
            
            min_c = min(hsi_before.shape[-1], hsi_rec_t1.shape[-1])
            E_t1 = np.mean(np.abs(hsi_before[:, :, :min_c] - hsi_rec_t1[:, :, :min_c]), axis=-1)
            b_amp_t1 = np.mean(uncertainty_t1[:, :, :min_c], axis=-1) + 1e-8
            D_t1 = E_t1 / b_amp_t1
            
            # Calibration robuste par percentile (votre approche préférée)
            t_rec = compute_dynamic_nfa_threshold(E_t1, nfa_target=args.target_nfa)
            t_weighted = compute_dynamic_nfa_threshold(D_t1, nfa_target=args.target_nfa)

            # -----------------------------------------------------------------
            # ÉTAPE B : APPLICATION SUR LA SCÈNE T2 (After)
            # -----------------------------------------------------------------
            print("  ↳ [T2 Inférence] Inférence sur T2 (After)...")
            inf_res_t2 = infer_scene(msi_after, msi_interp_after, model_rec, model_unc, device)
            hsi_rec_t2 = inf_res_t2['hsi_rec']
            uncertainty_t2 = inf_res_t2['uncertainty']
            
            min_c_t2 = min(hsi_before.shape[-1], hsi_rec_t2.shape[-1])
            method_rec_t2 = np.mean(np.abs(hsi_before[:, :, :min_c_t2] - hsi_rec_t2[:, :, :min_c_t2]), axis=-1)
            b_amp_t2 = np.mean(uncertainty_t2[:, :, :min_c_t2], axis=-1) + 1e-8
            method_weighted_t2 = method_rec_t2 / b_amp_t2
            
            scene_output = output_dir / scene
            scene_output.mkdir(parents=True, exist_ok=True)
            
            # -----------------------------------------------------------------
            # ÉTAPE C : SAUVEGARDE DE LA VISUALISATION GLOBALE DE LA SCÈNE
            # -----------------------------------------------------------------
            print("  ↳ [Global Visualization] Sauvegarde de la vue globale de la scène...")
            global_out_path = scene_output / f"{scene}_global_visualization.png"
            save_scene_visualization(
                msi_after, hsi_before, hsi_rec_t2, uncertainty_t2,
                method_rec_t2, method_weighted_t2, gt_mask, global_out_path,
                t_rec, t_weighted, target_nfa=args.target_nfa
            )
            
            # -----------------------------------------------------------------
            # ÉTAPE D : DÉCOUPAGE ET SAUVEGARDE DES PATCHS
            # -----------------------------------------------------------------
            print(f"  ↳ [Patch Slicing & Visualizations] Génération des visualisations pour TOUS les patchs ({args.patch_size}x{args.patch_size})...")
            patches_vis_dir = scene_output / "patches_visualizations"
            patches_vis_dir.mkdir(parents=True, exist_ok=True)
            
            h, w = msi_after.shape[:2]
            patch_count = 0
            
            for i in range(0, h, args.patch_size):
                for j in range(0, w, args.patch_size):
                    p_msi = extract_patch(msi_after, i, j, args.patch_size)
                    p_hsi_bef = extract_patch(hsi_before, i, j, args.patch_size)
                    p_hsi_rec = extract_patch(hsi_rec_t2, i, j, args.patch_size)
                    p_unc = extract_patch(uncertainty_t2, i, j, args.patch_size)
                    p_m_rec = extract_patch(method_rec_t2, i, j, args.patch_size)
                    p_m_w = extract_patch(method_weighted_t2, i, j, args.patch_size)
                    p_gt = extract_patch(gt_mask, i, j, args.patch_size) if gt_mask is not None else None
                    
                    patch_out_path = patches_vis_dir / f"patch_y{i}_x{j}.png"
                    save_patch_visualization(
                        p_msi, p_hsi_bef, p_hsi_rec, p_unc,
                        p_m_rec, p_m_w, p_gt, patch_out_path,
                        t_rec, t_weighted,
                        target_nfa=args.target_nfa
                    )
                    patch_count += 1
            
            print(f"    ✅ Vue globale sauvegardée : {global_out_path}")
            print(f"    ✅ Tous les patchs ({patch_count} au total) sauvegardés dans : {patches_vis_dir}")
            print(f"✅ {scene} terminé avec succès.")
            
        except Exception as e:
            print(f"❌ Erreur sur la scène {scene}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Terminé ! Tous les résultats sont dans : {output_dir}")


if __name__ == "__main__":
    main()