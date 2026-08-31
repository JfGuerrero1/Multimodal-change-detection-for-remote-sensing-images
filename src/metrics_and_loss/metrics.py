import numpy as np
from skimage.metrics import structural_similarity as ssim_sk
try:
    from constants import WVL_PRS
except ImportError:
    WVL_PRS = None


def ensure_chw(pred, target):
    """S'assure que les tableaux sont au format (C, H, W)."""
    if pred.ndim == 3 and pred.shape[-1] < pred.shape[0]:  # (H, W, C) -> (C, H, W)
        pred = np.moveaxis(pred, -1, 0)
        target = np.moveaxis(target, -1, 0)
    return pred, target


def ensure_hwc(pred, target):
    """S'assure que les tableaux sont au format (H, W, C)."""
    if pred.ndim == 3 and pred.shape[0] < pred.shape[-1]:  # (C, H, W) -> (H, W, C)
        pred = np.moveaxis(pred, 0, -1)
        target = np.moveaxis(target, 0, -1)
    return pred, target


def compute_mse(pred, target):
    """Calcule la MSE globale (éventuellement sur les bandes valides si WVL_PRS existe)."""
    pred, target = ensure_chw(pred, target)
    return np.mean((pred - target) ** 2)


def compute_mae(pred, target):
    pred, target = ensure_chw(pred, target)
    return np.mean(np.abs(pred - target))


import numpy as np

def compute_ergas(
    pred, target, sampling_ratio=1.0/3.0, min_val=1e-3, max_val=1.0, eval_indices=None
):
    """Calcul de l'ERGAS sécurisé pour des images (C, H, W) ou (H, W, C).

    - Nettoyage des NaN / Inf
    - Clamping strict dans [min_val, max_val]
    - Option pour filtrer/exclure certaines bandes instables (ex: bords des zones d'absorption)
    """
    pred, target = ensure_chw(pred, target)

    # 1. Cast en float64 pour éviter les erreurs de précision
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # 2. Remplacement des NaN/Inf éventuels
    pred = np.nan_to_num(pred, nan=min_val, posinf=max_val, neginf=min_val)
    target = np.nan_to_num(target, nan=min_val, posinf=max_val, neginf=min_val)

    # 3. Clamping strict
    pred = np.clip(pred, min_val, max_val)
    target = np.clip(target, min_val, max_val)

    # 4. Filtrage optionnel des bandes à évaluer (exclure les bords de bandes d'absorption)
    if eval_indices is not None:
        pred = pred[eval_indices, :, :]
        target = target[eval_indices, :, :]

    num_channels = pred.shape[0]
    if num_channels == 0:
        return 0.0

    # 5. Calcul vectorisé du RMSE et des moyennes par bande (axes H, W)
    rmse_per_band = np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))
    mean_target_per_band = np.maximum(
        np.mean(target, axis=(1, 2)), min_val
    )  # Sécurité division par zéro

    # 6. Ratio et formule ERGAS
    ratios = rmse_per_band / mean_target_per_band
    sum_ratio = np.sum(ratios**2)

    ergas = 100.0 * sampling_ratio * np.sqrt((1.0 / num_channels) * sum_ratio)

    float_ergas = float(ergas)
    
    # Sécurité supplémentaire contre les valeurs aberrantes
    if np.isnan(float_ergas) or np.isinf(float_ergas):
        return 0.0

    return float_ergas

def compute_ssim_multiband(pred, target):
    """Calcul du SSIM moyen sur l'ensemble des bandes (format scikit-image)."""
    pred, target = ensure_hwc(pred, target)
    
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        pred = pred[:, :, mask]
        target = target[:, :, mask]

    data_range = target.max() - target.min()
    if data_range == 0:
        data_range = 1.0

    score = ssim_sk(
        pred, target, channel_axis=-1, data_range=data_range, gaussian_weights=True
    )
    return score


def compute_mrae(pred, target, eps=1e-8):
    pred, target = ensure_chw(pred, target)
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        pred = pred[mask]
        target = target[mask]
    relative_error = np.abs(pred - target) / (np.abs(target) + eps)
    return np.mean(relative_error)
 
 
def compute_sam(gt, pred, eps=1e-8):
    """Calcule l'angle spectral moyen (en radians)."""
    gt, pred = ensure_hwc(gt, pred) # Convertit en (H, W, C) si besoin
    h, w, c = gt.shape
    
    gt_flat = gt.reshape(-1, c)
    pred_flat = pred.reshape(-1, c)
    
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        gt_flat = gt_flat[:, mask]
        pred_flat = pred_flat[:, mask]

    dot_product = np.sum(gt_flat * pred_flat, axis=1)
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)
    
    cos_theta = dot_product / (norm_gt * norm_pred + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.mean(np.arccos(cos_theta))
 
 
def compute_sam_map(gt, pred, eps=1e-8):
    """Renvoie la carte 2D des erreurs SAM (en radians)."""
    gt, pred = ensure_hwc(gt, pred)
    h, w, c = gt.shape
    
    gt_flat = gt.reshape(-1, c)
    pred_flat = pred.reshape(-1, c)
    
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        gt_flat = gt_flat[:, mask]
        pred_flat = pred_flat[:, mask]

    dot_product = np.sum(gt_flat * pred_flat, axis=1)
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)
    
    cos_theta = dot_product / (norm_gt * norm_pred + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
 
    sam_map = np.arccos(cos_theta).reshape(h, w)
    return sam_map

def compute_rmse(target, pred):
    """Calcule la Root Mean Square Error (RMSE)."""
    pred, target = ensure_chw(pred, target)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


import numpy as np


def compute_psnr(pred,target, data_range=1.0):
    """Calcule le PSNR (dB) sur les bandes valides.

    Args:
      pred (np.ndarray): Image prédite (C, H, W)
      target (np.ndarray): Image vérité terrain (C, H, W)
      data_range (float): Plage dynamique des réflectances (1.0 si [0, 1])
      wvl (np.ndarray, optional): Longueurs d'onde pour filtrer l'eau H2O
    """
    pred, target = ensure_chw(pred, target)


  # 2. Calcul du MSE
    mse = np.mean((pred - target) ** 2)

  # 3. Protection contre la division par zéro (évite 'inf' qui casse les moyennes)
    if mse < 1e-10:
        return 100.0  # PSNR plafonné à 100 dB si reconstruction quasi-parfaite

  # 4. Calcul du PSNR avec data_range fixe
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return float(psnr)