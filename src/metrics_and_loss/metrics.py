#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   MÉTRIQUES D'ÉVALUATION MSI / HSI                         ║
║  - Erreurs globales : MSE, MAE, MRAE, RMSE, PSNR, ERGAS                    ║
║  - Similarité structurelle : SSIM multibande                               ║
║  - Métriques spectrales : SAM (Angle Spectral) global et cartes 2D         ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim_sk

try:
    from constants import WVL_PRS
except ImportError:
    WVL_PRS = None


# ============================================================================
# UTILITAIRES DE FORMATAGE DES TENSEURS
# ============================================================================

def ensure_chw(pred: np.ndarray, target: np.ndarray):
    """S'assure que les tableaux sont au format (C, H, W)."""
    if pred.ndim == 3 and pred.shape[-1] < pred.shape[0]:  # (H, W, C) -> (C, H, W)
        pred = np.moveaxis(pred, -1, 0)
        target = np.moveaxis(target, -1, 0)
    return pred, target


def ensure_hwc(pred: np.ndarray, target: np.ndarray):
    """S'assure que les tableaux sont au format (H, W, C)."""
    if pred.ndim == 3 and pred.shape[0] < pred.shape[-1]:  # (C, H, W) -> (H, W, C)
        pred = np.moveaxis(pred, 0, -1)
        target = np.moveaxis(target, 0, -1)
    return pred, target


# ============================================================================
# MÉTRIQUES D'ERREUR GLOBALE (MSE, MAE, RMSE, MRAE)
# ============================================================================

def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Calcule la MSE globale (filtrée si WVL_PRS est disponible)."""
    pred, target = ensure_chw(pred, target)
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        pred = pred[mask]
        target = target[mask]
    return float(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Calcule la MAE globale."""
    pred, target = ensure_chw(pred, target)
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        pred = pred[mask]
        target = target[mask]
    return float(np.mean(np.abs(pred - target)))


def compute_rmse(target: np.ndarray, pred: np.ndarray) -> float:
    """Calcule la Root Mean Square Error (RMSE)."""
    pred, target = ensure_chw(pred, target)
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        pred = pred[mask]
        target = target[mask]
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_mrae(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Calcule l'erreur relative absolue moyenne (MRAE)."""
    pred, target = ensure_chw(pred, target)
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        pred = pred[mask]
        target = target[mask]
    relative_error = np.abs(pred - target) / (np.abs(target) + eps)
    return float(np.mean(relative_error))


# ============================================================================
# MÉTRIQUES DE QUALITÉ D'IMAGE (PSNR, SSIM, ERGAS)
# ============================================================================

def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Calcule le PSNR (dB) sur les bandes valides."""
    pred, target = ensure_chw(pred, target)
    if WVL_PRS is not None:
        mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000))
        pred = pred[mask]
        target = target[mask]

    mse = np.mean((pred - target) ** 2)

    # Protection contre la division par zéro (évite 'inf')
    if mse < 1e-10:
        return 100.0  # Plafonné à 100 dB si reconstruction quasi-parfaite

    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return float(psnr)


def compute_ssim_multiband(pred: np.ndarray, target: np.ndarray) -> float:
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
    return float(score)


def compute_ergas(
    pred: np.ndarray, 
    target: np.ndarray, 
    sampling_ratio: float = 1.0 / 3.0, 
    min_val: float = 1e-3, 
    max_val: float = 1.0, 
    eval_indices: list = None
) -> float:
    """Calcul de l'ERGAS sécurisé pour des images (C, H, W) ou (H, W, C)."""
    pred, target = ensure_chw(pred, target)

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    pred = np.nan_to_num(pred, nan=min_val, posinf=max_val, neginf=min_val)
    target = np.nan_to_num(target, nan=min_val, posinf=max_val, neginf=min_val)

    pred = np.clip(pred, min_val, max_val)
    target = np.clip(target, min_val, max_val)

    if eval_indices is not None:
        pred = pred[eval_indices, :, :]
        target = target[eval_indices, :, :]

    num_channels = pred.shape[0]
    if num_channels == 0:
        return 0.0

    rmse_per_band = np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))
    mean_target_per_band = np.maximum(np.mean(target, axis=(1, 2)), min_val)

    ratios = rmse_per_band / mean_target_per_band
    sum_ratio = np.sum(ratios**2)

    ergas = 100.0 * sampling_ratio * np.sqrt((1.0 / num_channels) * sum_ratio)
    float_ergas = float(ergas)
    
    if np.isnan(float_ergas) or np.isinf(float_ergas):
        return 0.0

    return float_ergas


# ============================================================================
# MÉTRIQUES SPECTRALES (SAM)
# ============================================================================

def compute_sam(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Calcule l'angle spectral moyen global (en radians)."""
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
    return float(np.mean(np.arccos(cos_theta)))
 
 
def compute_sam_map(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Renvoie la carte 2D spatiale des erreurs SAM (en radians), shape [H, W]."""
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