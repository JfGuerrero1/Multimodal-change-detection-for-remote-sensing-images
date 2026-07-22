import numpy as np
import numpy as np
from skimage.metrics import structural_similarity as ssim_sk
from constants import WVL_PRS


def compute_mse(pred, target):
    """pred, target: arrays NumPy de forme (C, H, W) ou (H, W, C)"""
    return np.mean((pred - target) ** 2)


def compute_mae(pred, target):
    return np.mean(np.abs(pred - target))


def compute_ergas(pred, target, sampling_ratio=1.0):
    """Calcul de l'ERGAS pour des images (C, H, W)

    sampling_ratio: ratio des résolutions spatiales (ex: res_MSI / res_HSI)
   
    """
    mask= ((WVL_PRS< 1350) | ((WVL_PRS> 1500) & (WVL_PRS < 1800)) | ((WVL_PRS > 2000) ))
    # On s'assure d'avoir la dimension des canaux en premier pour boucler
    if pred.shape[0] > pred.shape[2]:  # Détection si format (H, W, C)
        pred = np.moveaxis(pred, -1, 0)
        target = np.moveaxis(target, -1, 0)

    pred = pred[mask,:,:]
    target = target[mask,:,:]
    num_channels = pred.shape[0]
    rmse_per_band = []
    mean_target_per_band = []

    for c in range(num_channels):
        rmse = np.sqrt(np.mean((pred[c] - target[c]) ** 2))
        mean_tgt = np.mean(target[c])

        # Sécurité pour éviter la division par zéro sur les bandes sombres
        if mean_tgt < 1e-8:
            mean_tgt = 1e-8

        rmse_per_band.append(rmse)
        mean_target_per_band.append(mean_tgt)

    rmse_per_band = np.array(rmse_per_band)
    mean_target_per_band = np.array(mean_target_per_band)

    # Formule mathématique de l'ERGAS
    sum_ratio = np.sum((rmse_per_band / mean_target_per_band) ** 2)
    ergas = 100 * sampling_ratio * np.sqrt((1.0 / num_channels) * sum_ratio)
    return ergas


def compute_ssim_multiband(pred, target):
    """Calcul du SSIM moyen sur l'ensemble des 120 bandes."""
    # skimage attend (H, W, C) ou demande explicitement channel_axis
    if pred.shape[0] < pred.shape[2]:  # format (C, H, W) -> on passe en (H, W, C)
        pred = np.moveaxis(pred, 0, -1)
        target = np.moveaxis(target, 0, -1)

    # data_range dépend de ta normalisation (ex: 1.0 si tes réflectances sont entre 0 et 1)
    data_range = target.max() - target.min()

    # On calcule le SSIM bande par bande en spécifiant channel_axis
    score = ssim_sk(
        pred, target, channel_axis=-1, data_range=data_range, gaussian_weights=True
    )
    return score

##################Numpy for visualisation#################
def compute_mrae(pred, target, eps=1e-8):
    relative_error = np.abs(pred - target) / (np.abs(target) + eps)

    return np.mean(relative_error)
 
 
def compute_sam(gt, pred, eps=1e-8):
    c = gt.shape[-1]
    gt_flat = gt.reshape(-1, c)
    pred_flat = pred.reshape(-1, c)
    dot_product = np.sum(gt_flat * pred_flat, axis=1)
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)
    cos_theta = dot_product / (norm_gt * norm_pred + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # arcos'=-1/sqrt(1-x²)
    return np.mean(np.arccos(cos_theta))
 
 
def compute_sam_map(gt, pred, eps=1e-8):
    h, w, c = gt.shape
    gt_flat = gt.reshape(-1, c)
    pred_flat = pred.reshape(-1, c)
    dot_product = np.sum(gt_flat * pred_flat, axis=1)
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)
    cos_theta = dot_product / (norm_gt * norm_pred + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
 
    map = np.arccos(cos_theta.reshape(h, w))
    return map
 
 
def compute_mae_numpy(gt, pred):
    return np.mean(np.abs(gt - pred))