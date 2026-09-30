"""
╔════════════════════════════════════════════════════════════════════════════╗
║         DETECTION DE CHANGEMENT MSI/HSI - COMPARAISON DE METHODES          ║
║                                                                            ║
║   Inference globale -> Calibration T1 -> Application T2 -> Visu Globale/Patch║
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

# -----------------------------------------------------------------------------
#  NFA (Number of False Alarms) & METRIQUES
# -----------------------------------------------------------------------------

def compute_dynamic_nfa_threshold(map_data, nfa_target=0.01):
    """Calcule le seuil dynamique base sur le percentile cible (ex: 1% -> 99e percentile)"""
    valid_values = map_data[np.isfinite(map_data)]
    if len(valid_values) == 0:
        return 0.0
    percentile = (1.0 - nfa_target) * 100.0
    return float(np.percentile(valid_values, percentile))


def find_optimal_threshold_nfa(map_data, target_nfa=0.01):
    """Fonction de support pour determiner le seuil base sur le NFA."""
    threshold = compute_dynamic_nfa_threshold(map_data, nfa_target=target_nfa)
    return threshold, 0.0


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
    """Extrait un patch d'un tableau 2D ou 3D en gerant les bords"""
    if array_3d is None:
        return None
    h, w = array_3d.shape[:2]
    h_end = min(i + patch_size, h)
    w_end = min(j + patch_size, w)
    if array_3d.ndim == 3:
        return array_3d[i:h_end, j:w_end, :]
    else:
        return array_3d[i:h_end, j:w_end]


# -----------------------------------------------------------------------------
#  CHARGEMENT GLOBAL DES SCENES
# -----------------------------------------------------------------------------

def load_scene_data(scene_dir, scene, n_msi=12, n_hsi=195):
    """Charge MSI, HSI, HSI interpole T1 (Before) et MSI, HSI interpole T2 (After) ainsi que le masque GT sous format (H, W, C)."""
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

    # 1. Chargement HSI Before (T1) -> Assure en (H, W, C)
    with xr.open_dataset(hsi_before_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        hsi_before = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if hsi_before.ndim == 3 and hsi_before.shape[0] < hsi_before.shape[2]:
        hsi_before = np.transpose(hsi_before, (1, 2, 0))
    if hsi_before.ndim == 3 and hsi_before.shape[2] > n_hsi:
        hsi_before = hsi_before[:, :, :n_hsi]
    
    # 2. Chargement MSI After (T2) -> Assure en (H, W, C)
    with xr.open_dataset(msi_after_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        msi_after = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if msi_after.ndim == 3 and msi_after.shape[0] < msi_after.shape[2]:
        msi_after = np.transpose(msi_after, (1, 2, 0))
    if msi_after.ndim == 3 and msi_after.shape[2] > n_msi:
        msi_after = msi_after[:, :, :n_msi]

    # 3. Chargement MSI Before (T1) -> Assure en (H, W, C)
    with xr.open_dataset(msi_before_path) as ds:
        var_name = "sr" if "sr" in ds.data_vars else list(ds.data_vars)[0]
        msi_before = np.nan_to_num(ds[var_name].values, nan=0.0).astype(np.float32)
    if msi_before.ndim == 3 and msi_before.shape[0] < msi_before.shape[2]:
        msi_before = np.transpose(msi_before, (1, 2, 0))
    if msi_before.ndim == 3 and msi_before.shape[2] > n_msi:
        msi_before = msi_before[:, :, :n_msi]
    
    h_min = min(hsi_before.shape[0], msi_after.shape[0], msi_before.shape[0])
    w_min = min(hsi_before.shape[1], msi_after.shape[1], msi_before.shape[1])
    hsi_before = hsi_before[:h_min, :w_min, :]
    msi_after = msi_after[:h_min, :w_min, :]
    msi_before = msi_before[:h_min, :w_min, :]
    
    def interpolate_msi(msi_cube):
        msi_chw = np.transpose(msi_cube, (2, 0, 1))
        c_msi, h, w = msi_chw.shape
        interp_numpy = INTERP_MATRIX @ msi_chw.reshape(c_msi, -1)
        c_hsi = interp_numpy.shape[0]
        return np.transpose(interp_numpy.reshape(c_hsi, h, w), (1, 2, 0))[:, :, :n_hsi]

    msi_interp_after = interpolate_msi(msi_after)
    msi_interp_before = interpolate_msi(msi_before)
    
    gt_mask = None
    if gt_path.exists():
        with xr.open_dataset(gt_path) as ds:
            gt = np.nan_to_num(ds["change"].values, nan=0.0).astype(np.float32)
        if gt.ndim > 2:
            gt = gt.squeeze()
        gt_mask = gt[:h_min, :w_min]
        
    return msi_before, msi_after, hsi_before, msi_interp_before, msi_interp_after, gt_mask

# -----------------------------------------------------------------------------
#  METHODES DE DETECTION
# -----------------------------------------------------------------------------

def compute_detection_map(hsi_gt, hsi_rec, uncertainty, weighted=True):
    """Calcule l'erreur brute ou ponderee par l'incertitude"""
    min_c = min(hsi_gt.shape[-1], hsi_rec.shape[-1])
    E = np.mean(np.abs(hsi_gt[:, :, :min_c] - hsi_rec[:, :, :min_c]), axis=-1)
    
    if weighted:
        b_amp = np.mean(uncertainty[:, :, :min_c], axis=-1) + 1e-8
        return E / b_amp
    return E


def compute_embedding_score(z_1, z_2):
    """Calcule la distance L2 brute entre les embeddings après projection sur la sphere unitaire."""
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

# -----------------------------------------------------------------------------
#  INFERENCE GLOBALE ET CROISEE
# -----------------------------------------------------------------------------

def infer_scene_global(msi_cube, hsi_cube, msi_interp_cube,
                       model_rec, model_unc, model_siamese, device, use_amp=True, concat_inputs=False):
    model_rec.eval()
    model_unc.eval()
    model_siamese.eval()
    
    msi_t = torch.from_numpy(np.transpose(msi_cube, (2, 0, 1))).unsqueeze(0).float().to(device)
    hsi_t = torch.from_numpy(np.transpose(hsi_cube, (2, 0, 1))).unsqueeze(0).float().to(device)
    msi_interp_t = torch.from_numpy(np.transpose(msi_interp_cube, (2, 0, 1))).unsqueeze(0).float().to(device)
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
            hsi_rec = model_rec(msi_t, msi_interp_t)
            unc_input = torch.cat([msi_t, hsi_rec], dim=1)
            unc_output = model_unc(unc_input)
            unc = unc_output[0] if isinstance(unc_output, tuple) else unc_output
                
            if concat_inputs:
                siamese_msi_input = torch.cat([msi_t, hsi_rec, unc], dim=1)
            else:
                siamese_msi_input = msi_t
                
            z_msi, z_hsi = model_siamese(siamese_msi_input, hsi_t)
            z_msi2, z_hsi_rec=model_siamese(siamese_msi_input, hsi_rec)
            
    return {
        'hsi_rec': np.transpose(hsi_rec.squeeze(0).cpu().numpy(), (1, 2, 0)),
        'uncertainty': np.transpose(unc.squeeze(0).cpu().numpy(), (1, 2, 0)) if unc.ndim == 4 else unc.squeeze(0).cpu().numpy(),
        'z_msi': z_msi,
        'z_hsi': z_hsi,
        'z_hsi_rec': z_hsi_rec
    }


def infer_croise(msi_cube2, hsi_cube1, msi_interp_cube2, model_rec, model_unc, model_siamese, device, use_amp=True, concat_inputs=False):
    """Effectue l'inference croisee entre MSI T2 et HSI T1 via le réseau siamois."""
    model_rec.eval()
    model_unc.eval()
    model_siamese.eval()
        
    msi_t = torch.from_numpy(np.transpose(msi_cube2, (2, 0, 1))).unsqueeze(0).float().to(device)
    hsi_t = torch.from_numpy(np.transpose(hsi_cube1, (2, 0, 1))).unsqueeze(0).float().to(device)
    msi_interp_t = torch.from_numpy(np.transpose(msi_interp_cube2, (2, 0, 1))).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
            if concat_inputs:
                hsi_rec = model_rec(msi_t, msi_interp_t)
                unc_input = torch.cat([msi_t, hsi_rec], dim=1)
                unc_output = model_unc(unc_input)
                unc = unc_output[0] if isinstance(unc_output, tuple) else unc_output
                siamese_msi_input = torch.cat([msi_t, hsi_rec, unc], dim=1)
            else:
                siamese_msi_input = msi_t

            z_msi, z_hsi = model_siamese(siamese_msi_input, hsi_t)
            # Si besoin de récupérer la version HSI reconstruite également :
            # _, z_hsi_rec = model_siamese(siamese_msi_input, hsi_rec)
            
    return z_msi, z_hsi

# -----------------------------------------------------------------------------
#  UTILITAIRE VISUALISATIONS AVEC MIN/MAX ET NORMALISATION
# -----------------------------------------------------------------------------

def get_rgb_indices(wvl=WVL_PRS):
    if wvl is None:
        return [0, 1, 2]
    return [
        np.argmin(np.abs(wvl - 665.0)),
        np.argmin(np.abs(wvl - 560.0)),
        np.argmin(np.abs(wvl - 490.0)),
    ]

def to_rgb(cube, indices):
    if cube is None:
        return np.zeros((100, 100, 3))
    if cube.ndim == 3 and cube.shape[0] < cube.shape[2]:
        cube = np.transpose(cube, (1, 2, 0))
    
    valid_idx = [i for i in indices if i < cube.shape[-1]]
    if len(valid_idx) == 3:
        rgb = cube[:, :, valid_idx].copy()
    else:
        rgb = cube[:, :, :3].copy()
        
    for i in range(rgb.shape[-1]):
        band = rgb[:, :, i]
        v_min = np.percentile(band, 2)
        v_max = np.percentile(band, 98)
        if v_max > v_min:
            rgb[:, :, i] = np.clip((band - v_min) / (v_max - v_min), 0, 1)
        else:
            rgb[:, :, i] = np.zeros_like(band)
    return rgb

def clip_map(m, p_low=2, p_high=98):
    if m is None or np.all(m == 0):
        return m
    v_min = np.percentile(m[~np.isnan(m)], p_low)
    v_max = np.percentile(m[~np.isnan(m)], p_high)
    if v_max > v_min:
        return np.clip((m - v_min) / (v_max - v_min), 0, 1)
    return m

def visualize_all_methods_global(
    msi_after, hsi_rec, hsi_before, score_t1, score_t2, error_normalised,
    output_path, calibrated_thresholds, target_nfa=1.0
):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    idx_hsi = get_rgb_indices()
    idx_msi = [3, 2, 1]
    
    rgb_msi_aft = to_rgb(msi_after, idx_msi)
    rgb_hsi_rec = to_rgb(hsi_rec, idx_hsi)
    rgb_hsi = to_rgb(hsi_before, idx_hsi)
    
    t_m1 = calibrated_thresholds.get('M1: Embedding Score', {}).get('threshold', 0.5)
    t_err = calibrated_thresholds.get('Reconstruction Error (Weighted)', {}).get('threshold', t_m1)

    min_t1, max_t1 = float(score_t1.min()), float(score_t1.max())
    min_t2, max_t2 = float(score_t2.min()), float(score_t2.max())
    min_err, max_err = float(error_normalised.min()), float(error_normalised.max())

    axes[0, 0].imshow(rgb_msi_aft)
    axes[0, 0].set_title("MSI After (T2)", fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_hsi_rec)
    axes[0, 1].set_title("HSI Reconstruite (T2)", fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgb_hsi)
    axes[0, 2].set_title("HSI Before (T1)", fontweight='bold')
    axes[0, 2].axis('off')

    im1 = axes[1, 0].imshow(score_t1, cmap='jet')
    axes[1, 0].set_title(f"Score T1\nmin={min_t1:.2f}, max={max_t1:.2f}", fontweight='bold', fontsize=9)
    axes[1, 0].axis('off')
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(score_t2, cmap='jet')
    axes[1, 1].set_title(f"Score T2\nmin={min_t2:.2f}, max={max_t2:.2f}", fontweight='bold', fontsize=9)
    axes[1, 1].axis('off')
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 2].imshow(clip_map(error_normalised), cmap='inferno')
    axes[1, 2].set_title(f"Erreur Norm. T2\nmin={min_err:.2f}, max={max_err:.2f}", fontweight='bold', fontsize=9)
    axes[1, 2].axis('off')
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    axes[2, 0].imshow(score_t1 > t_m1, cmap='gray')
    axes[2, 0].set_title(f"Score T1 Binary\nThresh={t_m1:.3f}", fontweight='bold', fontsize=8)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(score_t2 > t_m1, cmap='gray')
    axes[2, 1].set_title(f"Score T2 Binary (Change)\nThresh={t_m1:.3f}", fontweight='bold', fontsize=8)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(error_normalised > t_err, cmap='gray')
    axes[2, 2].set_title(f"Erreur Norm. Binary\nThresh={t_err:.3f}", fontweight='bold', fontsize=8)
    axes[2, 2].axis('off')

    plt.suptitle(f"Global Change Detection Comparison (Target NFA = {target_nfa*100}%)", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detection de Changement - Inference Globale & Visu")
    parser.add_argument("--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="results_methods_comparison")
    parser.add_argument("--scenes", type=str, nargs='+', default=None)
    parser.add_argument("--target_nfa", type=float, default=1.0)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--save_all_patches", action="store_true", help="Activer l'enregistrement de tous les patchs individuels")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    n_msi = config["data"].get("n_msi", 12)
    n_hsi = config["data"].get("n_hsi", 195)
    patch_size = args.patch_size
    
    concat_inputs = config.get("model_siamese", {}).get("concat_inputs", False)
    print(f"Concatenation des entrees MSI + HSI_REC + Uncertainty pour le modele siamois : {concat_inputs}")
    
    print("Loading models...")
    eval_cfg = config.get("evaluation", {})

    model_rec = GradualExpansionUNet_residual(
        in_msi=n_msi, in_hsi=n_hsi, base_channel=config["model"].get("base_channel", 64)
    ).to(device).eval()
    ckpt = torch.load(eval_cfg["weights_path_reconstruction"], map_location=device, weights_only=False)
    model_rec.load_state_dict(ckpt.get("model_state_dict", ckpt))

    model_unc = DualBranchNAFNet(
        n_msi=n_msi, n_hsi=n_hsi, out_channels=n_hsi, width=config["model_uncertainty"].get("base_channel", 64)
    ).to(device).eval()
    ckpt = torch.load(eval_cfg["weights_path_uncertainty"], map_location=device, weights_only=False)
    model_unc.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    
    effective_in_channel_msi = n_msi + n_hsi + n_hsi if concat_inputs else n_msi

    model_siamese = SiameseNetwork_heterogene(
        in_channel_msi=effective_in_channel_msi, in_channel_hsi=n_hsi,
        base_channels=config["model_siamese"].get("base_channels", 32),
        latent_dim=config["model_siamese"].get("latent_dim", 32)
    ).to(device).eval()
    
    ckpt_siamese = torch.load(eval_cfg["weights_path_siamese"], map_location=device, weights_only=False)
    state_dict_siamese = ckpt_siamese.get("model_state_dict", ckpt_siamese)
    
    model_state = model_siamese.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict_siamese.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state_dict[k] = v
            
    model_siamese.load_state_dict(filtered_state_dict, strict=False)
    print("Models loaded\n")
    
    TEST_SCENES = ["baltijsk", "camerino", "codigoro", "copenhagen", "cullivel", "jagersfontein", "kirtland", "lorca"]
    VAL_SCENES = ["arborea", "athens", "beer_sheva", "istanbul", "los_cabos", "taiwan", "yuen_long"]
    TRAIN_SCENES = [
        "aranjuez", "bari", "beheira", "beirut", "belgrade", "binh_dai", "brasilia", "cape_town", "copperton",
        "cukotka", "dellys", "dubai", "dublin", "elsalto", "eyjafjoll", "fontainebleau", "fukushima",
        "guantanamo", "hanging_rock", "java", "jordan", "lagos", "london", "los_angeles",
        "malindi", "mantua", "mexico_city", "montevideo", "mosul", "mrirt", "muscat", "nagaoka",
        "new_york", "nicosia", "nouakchott", "novara", "palermo", "paris", "poinciana", "port_au_prince",
        "prague", "quito", "rome", "salinas", "sanaa", "shanghai", "spinazzola", "suez", "sydney",
        "tampa_bay", "tientsin", "tijuana", "tirana", "valencia"
    ]
    ALL_SCENE = TRAIN_SCENES + TEST_SCENES + VAL_SCENES
    scenes = args.scenes if args.scenes else ALL_SCENE
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for scene in tqdm(scenes, desc="Scenes"):
        scene_dir = Path(args.data_dir) / scene
        if not scene_dir.is_dir():
            continue
        
        try:
            print(f"\nProcessing {scene}...")
            
            msi_before_cube, msi_after_cube, hsi_before_cube, msi_interp_before, msi_interp_after, gt_mask = load_scene_data(
                scene_dir, scene, n_msi=n_msi, n_hsi=n_hsi
            )
            
            print(f"  -> [{scene}] Inference & calibration T1...")
            inf_t1 = infer_scene_global(msi_before_cube, hsi_before_cube, msi_interp_before, model_rec, model_unc, model_siamese, device, concat_inputs=concat_inputs)

            #SEUILS DE DETECTION de changement 
            score_emb_t1 = compute_embedding_score(inf_t1['z_msi'], inf_t1['z_hsi_rec']) #on peut mettre z_hsi pour faire le alibrage

            t_m1, _ = find_optimal_threshold_nfa(score_emb_t1, target_nfa=args.target_nfa)
            
            error_normalised_t1 = compute_detection_map(
                hsi_before_cube, inf_t1['hsi_rec'], inf_t1['uncertainty'], weighted=True
            )
            t_err, _ = find_optimal_threshold_nfa(error_normalised_t1, target_nfa=args.target_nfa)
            
            calibrated_thresholds = {
                'M1: Embedding Score': {'threshold': t_m1},
                'Reconstruction Error (Weighted)': {'threshold': t_err}
            }
            
            print(f"  -> [{scene}] Application sur T2...")
            inf_t2 = infer_scene_global(msi_after_cube, hsi_before_cube, msi_interp_after, model_rec, model_unc, model_siamese, device, concat_inputs=concat_inputs)

            z_msi_t2, z_hsi_t1 = infer_croise(msi_after_cube, hsi_before_cube, msi_interp_after, model_rec, model_unc, model_siamese, device, concat_inputs=concat_inputs)
            score_emb_t2 = compute_embedding_score(z_msi_t2.squeeze(0).cpu().numpy(), z_hsi_t1.squeeze(0).cpu().numpy())

            error_normalised = compute_detection_map(
                hsi_before_cube, inf_t2['hsi_rec'], inf_t2['uncertainty'], weighted=True
            )
            
            scene_output = output_dir / scene
            scene_output.mkdir(parents=True, exist_ok=True)
            
            visualize_all_methods_global(
                msi_after_cube, inf_t2['hsi_rec'], hsi_before_cube, 
                score_emb_t1, score_emb_t2, error_normalised,
                scene_output / "all_methods_comparison_global.png", 
                calibrated_thresholds, target_nfa=args.target_nfa
            )

            # Option conditionnelle pour enregistrer tous les patchs
            if args.save_all_patches:
                print(f"  -> [Patch Slicing] Generation des patchs ({patch_size}x{patch_size})...")
                patches_vis_dir = scene_output / "patches_visualizations"
                patches_vis_dir.mkdir(parents=True, exist_ok=True)
                
                h, w = msi_after_cube.shape[:2]
                patch_count = 0
                
                for y in range(0, h - (h % patch_size), patch_size):
                    for x in range(0, w - (w % patch_size), patch_size):
                        p_msi_aft = extract_patch(msi_after_cube, y, x, patch_size)
                        p_hsi_rec = extract_patch(inf_t2['hsi_rec'], y, x, patch_size)
                        p_hsi_bef = extract_patch(hsi_before_cube, y, x, patch_size)
                        p_score_t1 = extract_patch(score_emb_t1, y, x, patch_size)
                        p_score_t2 = extract_patch(score_emb_t2, y, x, patch_size)
                        p_err_norm = extract_patch(error_normalised, y, x, patch_size)
                        
                        patch_out_path = patches_vis_dir / f"patch_y{y}_x{x}.png"
                        visualize_all_methods_global(
                            p_msi_aft, p_hsi_rec, p_hsi_bef, p_score_t1, p_score_t2, p_err_norm,
                            patch_out_path, calibrated_thresholds, target_nfa=args.target_nfa
                        )
                        patch_count += 1
                
                print(f"    -> {patch_count} patchs sauvegardes dans : {patches_vis_dir}")
            else:
                print(f"  -> [Info] Sauvegarde des patchs désactivée (utilisez --save_all_patches pour l'activer).")

        except Exception as e:
            print(f"Erreur lors du traitement de la scene {scene}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()