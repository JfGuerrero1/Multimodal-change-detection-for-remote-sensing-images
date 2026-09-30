import os
import time
import gc
from pathlib import Path
import sys



ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator
import numpy as np
import torch
import xarray as xr
import scipy.interpolate
from skimage.metrics import structural_similarity as ssim_sk
from scipy.stats import spearmanr
import sys
from pathlib import Path



# Maintenant tu peux importer models sans erreur
from src.models.unet_residual import GradualExpansionUNet_residual
from src.models.unet_standard import GradualExpansionUNet


# Import des constantes et métriques du projet
from src.constants import DEFAULT_SRF_PATH, INTERP_MATRIX, SRF_MATRIX, WVL_PRS, WVL_S2

from src.metrics_and_loss.metrics import (
    compute_sam_map, compute_mae, compute_ergas, compute_mrae,
    compute_ssim_multiband, compute_mse, compute_sam, compute_psnr, compute_rmse
)


def compute_uncertainty_metrics(unc_gt, unc_pred, num_bins=100):
  """Calcule l'AUSE et la corrélation de Spearman de manière vectorisée."""
  E = unc_gt.ravel()
  U = unc_pred.ravel()
  n_samples = len(E)

  if n_samples > 1_000_000:
    sub_idx = np.random.choice(n_samples, size=100_000, replace=False)
    unc_spearman, _ = spearmanr(U[sub_idx], E[sub_idx])
  else:
    unc_spearman, _ = spearmanr(U, E)

  idx_model = np.argsort(U)[::-1]
  idx_oracle = np.argsort(E)[::-1]

  E_model_sq = E[idx_model] ** 2
  E_oracle_sq = E[idx_oracle] ** 2

  sum_sq_model = np.cumsum(E_model_sq[::-1])[::-1]
  sum_sq_oracle = np.cumsum(E_oracle_sq[::-1])[::-1]

  k_indices = np.linspace(0, n_samples - 1, num_bins, endpoint=False, dtype=int)
  counts = n_samples - k_indices

  rmse_model = np.sqrt(sum_sq_model[k_indices] / counts)
  rmse_oracle = np.sqrt(sum_sq_oracle[k_indices] / counts)

  auc_model = np.trapz(rmse_model, dx=1.0 / num_bins)
  auc_oracle = np.trapz(rmse_oracle, dx=1.0 / num_bins)

  return {
      "unc_spearman": float(unc_spearman),
      "unc_ause": float(auc_model - auc_oracle),
  }


def compute_scene_data_no_mode(msi_path, hsi_path, patch_size=256):
  """Charge et extrait un patch aléatoire des cubes HSI et MSI."""
  with xr.open_dataset(hsi_path) as ds_gt:
    cube_gt = np.nan_to_num(ds_gt["sr"].values, nan=0.0)

  with xr.open_dataset(msi_path) as ds_input:
    cube_msi = np.nan_to_num(ds_input["sr"].values, nan=0.0)

  h, w, c_hsi = cube_gt.shape
  hm, wm, c_msi = cube_msi.shape

  target_h = min(h, hm)
  target_w = min(w, wm)
  cube_gt = cube_gt[:target_h, :target_w, :]
  cube_msi = cube_msi[:target_h, :target_w, :]

  # MSI simulée via SRF_MATRIX
  gt_flat = cube_gt.reshape(-1, c_hsi)
  msi_sim_flat = np.dot(gt_flat, SRF_MATRIX)
  cube_msi_sim = msi_sim_flat.reshape(target_h, target_w, SRF_MATRIX.shape[1])

  # Image interpolée via INTERP_MATRIX
  input_numpy = cube_msi.transpose(2, 0, 1)
  interp_numpy = INTERP_MATRIX @ input_numpy.reshape(c_msi, -1)
  interp_numpy = interp_numpy.reshape(c_hsi, target_h, target_w)
  cube_interp = interp_numpy.transpose(1, 2, 0)

  # Patching aléatoire
  if target_h >= patch_size and target_w >= patch_size:
    y = np.random.randint(0, target_h - patch_size + 1)
    x = np.random.randint(0, target_w - patch_size + 1)

    cube_gt = cube_gt[y : y + patch_size, x : x + patch_size, :]
    cube_msi = cube_msi[y : y + patch_size, x : x + patch_size, :]
    cube_msi_sim = cube_msi_sim[y : y + patch_size, x : x + patch_size, :]
    cube_interp = cube_interp[y : y + patch_size, x : x + patch_size, :]

  return {
      "cube_gt": cube_gt,
      "cube_msi": cube_msi,
      "cube_msi_sim": cube_msi_sim,
      "cube_interp": cube_interp,
  }


def normalize_for_display(img):
  """Normalise une image entre 0 et 1 pour l'affichage matplotlib."""
  img_min = np.percentile(img, 2)
  img_max = np.percentile(img, 98)
  img_norm = np.clip((img - img_min) / (img_max - img_min + 1e-8), 0, 1)
  return img_norm


def plot_rgb_comparison(
    data_dict,
    hsi_wavelengths=None,
    msi_wavelengths=None,
    msi_rgb_indices=None,
    with_interpolation=True,
    output_dir="comparison_plot.png"
):
  cube_gt = data_dict["cube_gt"]
  cube_msi = data_dict["cube_msi"]
  cube_msi_sim = data_dict["cube_msi_sim"]

  c_hsi = cube_gt.shape[2]
  c_msi = cube_msi.shape[2]
  
  if hsi_wavelengths is None:
    hsi_wavelengths = np.linspace(400, 2500, c_hsi)
  if msi_wavelengths is None:
    msi_wavelengths = np.array(
        [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
    )[:c_msi]

  # Extraction RGB HSI (650nm, 550nm, 470nm)
  r_val, g_val, b_val = 650, 550, 470
  r_idx = np.argmin(np.abs(hsi_wavelengths - r_val))
  g_idx = np.argmin(np.abs(hsi_wavelengths - g_val))
  b_idx = np.argmin(np.abs(hsi_wavelengths - b_val))
  hsi_rgb_bands = [r_idx, g_idx, b_idx]
  rgb_gt = normalize_for_display(cube_gt[:, :, hsi_rgb_bands])

  # Extraction RGB MSI (ex: 665nm, 560nm, 490nm pour R, G, B)
  if msi_rgb_indices is not None and isinstance(msi_rgb_indices, (list, tuple)) and len(msi_rgb_indices) == 3:
    r_m, g_m, b_m = msi_rgb_indices
    rm_val, gm_val, bm_val = msi_wavelengths[r_m], msi_wavelengths[g_m], msi_wavelengths[b_m]
  else:
    rm_val, gm_val, bm_val = 665, 560, 490
    r_m = np.argmin(np.abs(msi_wavelengths - rm_val))
    g_m = np.argmin(np.abs(msi_wavelengths - gm_val))
    b_m = np.argmin(np.abs(msi_wavelengths - bm_val))

  rgb_msi = normalize_for_display(cube_msi[:, :, [r_m, g_m, b_m]])
  rgb_msi_sim = normalize_for_display(cube_msi_sim[:, :, [r_m, g_m, b_m]])

  if with_interpolation:
    cube_interp = data_dict["cube_interp"]
    rgb_interp = normalize_for_display(cube_interp[:, :, hsi_rgb_bands])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = [
        f"HSI Réel (GT)\nRGB: {r_val}/{g_val}/{b_val}nm",
        f"MSI Simulé\nRGB: {int(rm_val)}/{int(gm_val)}/{int(bm_val)}nm",
        f"MSI Réel\nRGB: {int(rm_val)}/{int(gm_val)}/{int(bm_val)}nm",
        f"HSI Interpolé\nRGB: {r_val}/{g_val}/{b_val}nm",
    ]
    images = [rgb_gt, rgb_msi_sim, rgb_msi, rgb_interp]
  else:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [
        f"HSI Réel (GT)\nRGB: {r_val}/{g_val}/{b_val}nm",
        f"MSI Simulé\nRGB: {int(rm_val)}/{int(gm_val)}/{int(bm_val)}nm",
        f"MSI Réel\nRGB: {int(rm_val)}/{int(gm_val)}/{int(bm_val)}nm",
    ]
    images = [rgb_gt, rgb_msi_sim, rgb_msi]

  for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

  plt.tight_layout()
  plt.savefig(output_dir, dpi=300, bbox_inches='tight')
  plt.show()


def plot_multimodal_pixel_analysis(
    data_dict,
    pixel_coord=None,
    hsi_wavelengths=None,
    msi_wavelengths=None,
    output_dir="multimodal_pixel_analysis.png"
):
  """Affiche une vue 2x2 avec résidu signé (différence) et longueurs d'onde explicites."""
  cube_gt = data_dict["cube_gt"]
  cube_msi = data_dict["cube_msi"]
  cube_interp = data_dict["cube_interp"]

  h, w, c_hsi = cube_gt.shape
  _, _, c_msi = cube_msi.shape

  if pixel_coord is None:
    y, x = h // 2, w // 2
  else:
    y, x = pixel_coord

  if hsi_wavelengths is None:
    hsi_wavelengths = np.linspace(400, 2500, c_hsi)
  if msi_wavelengths is None:
    msi_wavelengths = np.array(
        [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
    )[:c_msi]

  # HSI RGB wavelengths
  r_val, g_val, b_val = 650, 550, 470
  r_idx = np.argmin(np.abs(hsi_wavelengths - r_val))
  g_idx = np.argmin(np.abs(hsi_wavelengths - g_val))
  b_idx = np.argmin(np.abs(hsi_wavelengths - b_val))
  hsi_rgb_bands = [r_idx, g_idx, b_idx]
  rgb_gt = normalize_for_display(cube_gt[:, :, hsi_rgb_bands])

  # MSI RGB wavelengths (ex: 665, 560, 490 nm)
  rm_val, gm_val, bm_val = 665, 560, 490
  r_m = np.argmin(np.abs(msi_wavelengths - rm_val))
  g_m = np.argmin(np.abs(msi_wavelengths - gm_val))
  b_m = np.argmin(np.abs(msi_wavelengths - bm_val))
  rgb_msi = normalize_for_display(cube_msi[:, :, [r_m, g_m, b_m]])

  # --- Calcul strict de la différence (Résidu Signé : HSI Réel - HSI Interpolé) ---
  cube_residu = cube_gt - cube_interp
  residu_map = np.mean(cube_residu, axis=2)  # Moyenne spatiale/spectrale des différences signées
  vmax = np.max(np.abs(residu_map))
  vmin = -vmax

  fig, axes = plt.subplots(2, 2, figsize=(14, 12))

  # Panneau 0,0 : MSI Réel
  axes[0, 0].imshow(rgb_msi)
  axes[0, 0].scatter(x, y, color="red", marker="s", s=80, facecolors="none", edgecolors="red")
  axes[0, 0].set_title(f"MSI Réel [RGB: {rm_val}/{gm_val}/{bm_val}nm] \n(Pixel : [{x}, {y}])", fontweight="bold")
  axes[0, 0].axis("off")

  # Panneau 0,1 : HSI Réel
  axes[0, 1].imshow(rgb_gt)
  axes[0, 1].scatter(x, y, color="red", marker="s", s=80, facecolors="none", edgecolors="red")
  axes[0, 1].set_title(f"HSI Réel (GT) [RGB: {r_val}/{g_val}/{b_val}nm]", fontweight="bold")
  axes[0, 1].axis("off")

  # Panneau 1,0 : Résidu (Différence Signée)
  im_res = axes[1, 0].imshow(residu_map, cmap="seismic", vmin=vmin, vmax=vmax)
  axes[1, 0].set_title("Résidu (Différence HSI Réel - Interpolé)", fontweight="bold")
  axes[1, 0].axis("off")
  fig.colorbar(im_res, ax=axes[1, 0], fraction=0.046, pad=0.04)

  # Panneau 1,1 : Profil spectral
  spec_hsi = cube_gt[y, x, :]
  spec_interp = cube_interp[y, x, :]
  spec_msi = cube_msi[y, x, :]

  axes[1, 1].plot(hsi_wavelengths, spec_hsi, label="HSI Réel", color="blue", linewidth=1.5)
  axes[1, 1].plot(hsi_wavelengths, spec_interp, label="HSI Interpolé", color="orange", linestyle="--", linewidth=1.5)
  axes[1, 1].scatter(msi_wavelengths[: len(spec_msi)], spec_msi, label="MSI Réel (Pixel)", color="red", marker="s", s=50, zorder=5)

  axes[1, 1].set_title(f"Comparaison Spectrale (Pixel [{y}, {x}])", fontweight="bold")
  axes[1, 1].set_xlabel("Longueur d'onde (nm)")
  axes[1, 1].set_ylabel("Réflectance")
  axes[1, 1].legend()
  axes[1, 1].grid(True, linestyle=":", alpha=0.6)

  plt.tight_layout()
  plt.savefig(output_dir, dpi=300, bbox_inches="tight")
  plt.show()


def plot_multimodal_simple_pixel_analysis(
    data_dict,
    pixel_coord=None,
    hsi_wavelengths=None,
    msi_wavelengths=None,
    output_dir="multimodal_simple_pixel_analysis.png"
):
  """Affiche une vue 1x3 (sans interpolation) avec longueurs d'onde explicites."""
  cube_gt = data_dict["cube_gt"]
  cube_msi = data_dict["cube_msi"]

  h, w, c_hsi = cube_gt.shape
  _, _, c_msi = cube_msi.shape

  if pixel_coord is None:
    y, x = h // 2, w // 2
  else:
    y, x = pixel_coord

  if hsi_wavelengths is None:
    hsi_wavelengths = np.linspace(400, 2500, c_hsi)
  if msi_wavelengths is None:
    msi_wavelengths = np.array(
        [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
    )[:c_msi]

  r_val, g_val, b_val = 650, 550, 470
  r_idx = np.argmin(np.abs(hsi_wavelengths - r_val))
  g_idx = np.argmin(np.abs(hsi_wavelengths - g_val))
  b_idx = np.argmin(np.abs(hsi_wavelengths - b_val))
  hsi_rgb_bands = [r_idx, g_idx, b_idx]
  rgb_gt = normalize_for_display(cube_gt[:, :, hsi_rgb_bands])

  rm_val, gm_val, bm_val = 665, 560, 490
  r_m = np.argmin(np.abs(msi_wavelengths - rm_val))
  g_m = np.argmin(np.abs(msi_wavelengths - gm_val))
  b_m = np.argmin(np.abs(msi_wavelengths - bm_val))
  rgb_msi = normalize_for_display(cube_msi[:, :, [r_m, g_m, b_m]])

  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  axes[0].imshow(rgb_msi)
  axes[0].scatter(x, y, color="red", marker="s", s=80, facecolors="none", edgecolors="red")
  axes[0].set_title(f"MSI Réel [RGB: {rm_val}/{gm_val}/{bm_val}nm] \n(Pixel : [{x}, {y}])", fontweight="bold")
  axes[0].axis("off")

  axes[1].imshow(rgb_gt)
  axes[1].scatter(x, y, color="red", marker="s", s=80, facecolors="none", edgecolors="red")
  axes[1].set_title(f"HSI Réel (GT) [RGB: {r_val}/{g_val}/{b_val}nm]", fontweight="bold")
  axes[1].axis("off")

  spec_hsi = cube_gt[y, x, :]
  spec_msi = cube_msi[y, x, :]

  axes[2].plot(hsi_wavelengths, spec_hsi, label="HSI Réel", color="blue", linewidth=1.5)
  axes[2].scatter(msi_wavelengths[: len(spec_msi)], spec_msi, label="MSI Réel (Pixel)", color="red", marker="s", s=60, zorder=5)

  axes[2].set_title(f"Comparaison Spectrale Brute (Pixel [{y}, {x}])", fontweight="bold")
  axes[2].set_xlabel("Longueur d'onde (nm)")
  axes[2].set_ylabel("Réflectance")
  axes[2].legend()
  axes[2].grid(True, linestyle=":", alpha=0.6)

  plt.tight_layout()
  plt.savefig(output_dir, dpi=300, bbox_inches="tight")
  plt.show()

def plot_random_hsi_spectra(cube_hsi, hsi_wavelengths=None, output_path="random_spectra.svg", num_spectra=10):
    """
    Prend un cube HSI en entrée (H, W, C), sélectionne aléatoirement `num_spectra` pixels,
    trace leurs signatures spectrales et sauvegarde le graphique au format SVG.
    """
    h, w, c_hsi = cube_hsi.shape
    
    if hsi_wavelengths is None:
        hsi_wavelengths = np.linspace(400, 2500, c_hsi)
        
    # Génération de coordonnées aléatoires uniques
    total_pixels = h * w
    num_spectra = min(num_spectra, total_pixels)
    random_indices = np.random.choice(total_pixels, size=num_spectra, replace=False)
    y_indices, x_indices = np.unravel_index(random_indices, (h, w))
    
    plt.figure(figsize=(12, 6))
    
    for i in range(num_spectra):
        y, x = y_indices[i], x_indices[i]
        spectrum = cube_hsi[y, x, :]
        plt.plot(hsi_wavelengths, spectrum, alpha=0.8, linewidth=1.2, label=f"Pixel {i+1}")
        
    plt.title(f"Spectres HSI de {num_spectra} pixels aléatoires (400 - 2500 nm)", fontsize=12, fontweight="bold")
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Réflectance de surface")
    
    # Légende propre sur le côté pour ne pas surcharger le graphique
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    # Sauvegarde au format SVG vectoriel
    plt.savefig(output_path, format="svg", bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique vectoriel sauvegardé avec succès dans : {output_path}")

def main():
  hsi_file = "/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/mumucd/belgrade/belgrade-after-prs.nc"
  msi_file = "/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/mumucd/belgrade/belgrade-after-s2.nc"

  print("1. Chargement et extraction des patchs...")
  data_results = compute_scene_data_no_mode(
      msi_path=msi_file, hsi_path=hsi_file, patch_size=256
  )

  print("2. Affichage et sauvegarde de l'analyse par pixel (2x2)...")
  plot_multimodal_pixel_analysis(
      data_dict=data_results,
      pixel_coord=(128, 128),
      hsi_wavelengths=WVL_PRS,
      msi_wavelengths=WVL_S2,
      output_dir="multimodal_pixel_analysis.png"
  )

  print("3. Affichage de la comparaison RGB (avec interpolation)...")
  plot_rgb_comparison(
      data_dict=data_results,
      hsi_wavelengths=WVL_PRS,
      msi_wavelengths=WVL_S2,
      msi_rgb_indices=(3, 2, 1),
      with_interpolation=True,
      output_dir="comparison_with_interpolation.png"
  )

  print("4. Affichage de la comparaison RGB (sans interpolation)...")
  plot_rgb_comparison(
      data_dict=data_results,
      hsi_wavelengths=WVL_PRS,
      msi_wavelengths=WVL_S2,
      msi_rgb_indices=(3, 2, 1),
      with_interpolation=False,
      output_dir="comparison_without_interpolation.png"
  )

  print("5. Affichage et sauvegarde de l'analyse spectrale brute (1x3)...")
  plot_multimodal_simple_pixel_analysis(
      data_dict=data_results,
      pixel_coord=(128, 128),
      hsi_wavelengths=WVL_PRS,
      msi_wavelengths=WVL_S2,
      output_dir="multimodal_simple_pixel_analysis.png"
  )

  print("6. Génération et sauvegarde des spectres aléatoires en SVG...")
  plot_random_hsi_spectra(
      cube_hsi=data_results["cube_gt"],  # Correction ici : on extrait le cube HSI du dictionnaire
      hsi_wavelengths=WVL_PRS,
      output_path="spectres_aleatoires_prisma.svg",
      num_spectra=10
  )

if __name__ == "__main__":
  main()

