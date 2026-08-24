import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from models import GradualExpansionUNet, GradualExpansionUNet_residual
import scipy.interpolate
from matplotlib.ticker import MaxNLocator,FixedLocator
import time
import gc
from skimage.metrics import structural_similarity as ssim_sk
torch.cuda.init()
torch.cuda.set_device(0)
from constants import DEFAULT_SRF_PATH,INTERP_MATRIX,SRF_MATRIX


from src.old.metrics_and_loss.metrics import compute_sam_map,compute_mae,compute_ergas,compute_mrae,compute_ssim_multiband,compute_mse,compute_sam,compute_psnr,compute_rmse    

import gc
import os
from pathlib import Path
import time
from constants import DEFAULT_SRF_PATH, INTERP_MATRIX
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
from models import GradualExpansionUNet, GradualExpansionUNet_residual
import numpy as np
import scipy.interpolate
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim_sk
from src.old.metrics_and_loss.metrics import (
    compute_ergas,
    compute_mae,
    compute_mrae,
    compute_mse,
    compute_psnr,
    compute_rmse,
    compute_sam,
    compute_sam_map,
    compute_ssim_multiband,
)
import torch
import xarray as xr

torch.cuda.init()
torch.cuda.set_device(0)



def compute_uncertainty_metrics(unc_gt, unc_pred, num_bins=100):
  """Calcule l'AUSE et la corrélation de Spearman de manière ultra-rapide et vectorisée."""
  E = unc_gt.ravel()  # Erreur réelle |Y_pred - Y_gt|
  U = unc_pred.ravel()  # Incertitude prédite U_hat
  n_samples = len(E)

  # 1. Corrélation de Spearman (échantillonnée si > 1M pixels pour garder une vitesse max)
  if n_samples > 1_000_000:
    sub_idx = np.random.choice(n_samples, size=100_000, replace=False)
    unc_spearman, _ = spearmanr(U[sub_idx], E[sub_idx])
  else:
    unc_spearman, _ = spearmanr(U, E)

  # 2. AUSE Vectorisée (Sommes cumulées)
  idx_model = np.argsort(U)[::-1]
  idx_oracle = np.argsort(E)[::-1]

  E_model_sq = E[idx_model] ** 2
  E_oracle_sq = E[idx_oracle] ** 2

  sum_sq_model = np.cumsum(E_model_sq[::-1])[::-1]
  sum_sq_oracle = np.cumsum(E_oracle_sq[::-1])[::-1]

  k_indices = np.linspace(
      0, n_samples - 1, num_bins, endpoint=False, dtype=int
  )
  counts = n_samples - k_indices

  rmse_model = np.sqrt(sum_sq_model[k_indices] / counts)
  rmse_oracle = np.sqrt(sum_sq_oracle[k_indices] / counts)

  auc_model = np.trapz(rmse_model, dx=1.0 / num_bins)
  auc_oracle = np.trapz(rmse_oracle, dx=1.0 / num_bins)

  return {
      "unc_spearman": float(unc_spearman),
      "unc_ause": float(auc_model - auc_oracle),
  }


def compute_scene_data(
    gt_path,
    input_path,
    model,
    model_name,
    model_uncertainty=None,
    model_uncertainty_name=None,
    is_simulated=False,
    is_residual=False,
    device=None,
):
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model.to(device)
  model.eval()

  # 1. Chargement GT
  with xr.open_dataset(gt_path) as ds_gt:
    cube_gt = np.nan_to_num(ds_gt["sr"].values, nan=0.0)

  h, w, c = cube_gt.shape

  # 2. Préparation Entrée MSI
  if is_simulated:
    gt_flat = cube_gt.reshape(-1, c)
    msi_flat = np.dot(gt_flat, SRF_MATRIX)
    cube_msi = msi_flat.reshape(h, w, SRF_MATRIX.shape[1])
    input_numpy = cube_msi.transpose(2, 0, 1)
    msi_label = "MSI Simulée"
  else:
    if input_path is None:
      raise ValueError(
          "Le paramètre 'input_path' est obligatoire pour les données"
          " réelles."
      )
    with xr.open_dataset(input_path) as ds_input:
      input_raw = np.nan_to_num(ds_input["sr"].values, nan=0.0)
    cube_msi = input_raw
    input_numpy = input_raw.transpose(2, 0, 1)
    msi_label = "MSI Réelle"

  # 3. Préparation résiduelle
  cube_interp = None
  interp_tensor = None

  if is_residual:
    interp_numpy = INTERP_MATRIX @ input_numpy.reshape(12, -1)
    interp_numpy = interp_numpy.reshape(c, h, w)
    interp_tensor = torch.from_numpy(interp_numpy).unsqueeze(0).to(device)
    cube_interp = interp_numpy.transpose(1, 2, 0)

  input_tensor = torch.from_numpy(input_numpy).float().unsqueeze(0).to(device)
  cube_res = None

  # 4. Inférence Reconstruction
  with torch.no_grad():
    if is_residual:
      pred = model(input_tensor, interp_tensor)
      res = pred - interp_tensor
      cube_res = res.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    else:
      pred = model(input_tensor)

  cube_predict = pred.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

  # Métriques Reconstruction
  img_mse = compute_mse(cube_gt, cube_predict)
  img_sam = compute_sam(cube_gt, cube_predict)
  img_mae = compute_mae(cube_gt, cube_predict)
  img_ergas = compute_ergas(cube_gt, cube_predict)
  img_ssim = compute_ssim_multiband(cube_gt, cube_predict)
  img_mrae = compute_mrae(cube_gt, cube_predict)
  img_rmse = compute_rmse(cube_gt, cube_predict)
  img_psnr = compute_psnr(cube_gt, cube_predict)

  # 5. Inférence & Métriques d'Incertitude
  cube_uncertainty = None
  cube_unc_gt = None
  unc_mae = None
  unc_spearman = None
  unc_ause = None

  if model_uncertainty is not None:
    model_uncertainty.to(device)
    model_uncertainty.eval()

    with torch.no_grad():
      # Passe directement le tenseur 'pred' PyTorch
      pred_unc = model_uncertainty(pred)

    cube_uncertainty = (
        pred_unc.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    )

    # Calcul de la vraie carte d'erreur absolue
    cube_unc_gt = np.abs(cube_predict - cube_gt)

    # Métriques d'Incertitude
    unc_mae = float(np.mean(np.abs(cube_uncertainty - cube_unc_gt)))


  return {
      "cube_gt": cube_gt,
      "cube_msi": cube_msi,
      "cube_predict": cube_predict,
      "cube_interp": cube_interp,
      "cube_res": cube_res,
      "cube_uncertainty": cube_uncertainty,
      "cube_unc_gt": cube_unc_gt,
      "msi_label": msi_label,
      "is_residual": is_residual,
      "model name": model_name,
      "model_uncertainty_name": model_uncertainty_name,
      # Métriques Reconstruction
      "img_mse": img_mse,
      "img_sam": img_sam,
      "img_mae": img_mae,
      "img_ergas": img_ergas,
      "img_ssim": img_ssim,
      "img_mrae": img_mrae,
      "img_rmse": img_rmse,
      "img_psnr": img_psnr,
      # Métriques Incertitude
      "unc_mae": unc_mae,

  }