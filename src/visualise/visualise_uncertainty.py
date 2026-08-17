import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path de Python
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
  sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import wandb
# Sécurité pour les serveurs sans GUI
matplotlib.use('Agg')
from src.constants import WVL_PRS, DW_INFO,WVL_S2
from src.metrics_and_loss.metrics import compute_sam_map,compute_mae,compute_ergas,compute_mrae,compute_ssim_multiband,compute_mse,compute_sam,compute_psnr,compute_rmse    
CHANNELS_ATM = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 200, 220]
CHANNELS_STD = [5, 11, 20, 32, 36, 40, 44, 50, 52, 59, 122, 187]
CHANNELS_ONLY_ATM = [98, 101, 104, 107, 110, 113, 116, 119, 122, 125]



from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def visualise_synthesis_uncertainty(
    data, save_name, plot_dir, kept_indices=None, log_to_wandb=False
):
  """Génère une planche de synthèse 2x3 avec gestion des bandes supprimées (NaN)."""
  cube_gt = data["cube_gt"]  # Shape (H, W, n_kept)
  cube_predict = data["cube_predict"]  # Shape (H, W, n_kept)
  cube_msi = data["cube_msi"]  # Shape (H, W, n_msi)
  model_name = data.get("model name", "Modèle ML")
  cube_uncertainty = data.get("cube_uncertainty", None)  # Shape (H, W, n_kept)

  # Récupération ou calcul automatique des métriques
  img_rmse = data.get("img_rmse", compute_rmse(cube_predict, cube_gt))
  img_mae = data.get("img_mae", compute_mae(cube_predict, cube_gt))
  img_sam = data.get("img_sam", compute_sam(cube_gt, cube_predict))
  img_ssim = data.get(
      "img_ssim", compute_ssim_multiband(cube_predict, cube_gt)
  )
  img_ergas = data.get("img_ergas", compute_ergas(cube_predict, cube_gt))
  img_psnr = data.get("img_psnr", compute_psnr(cube_gt, cube_predict))

  n_kept_bands = cube_gt.shape[-1]

  # 1. Reconstitution du spectre d'erreur complet (avec NaN sur les bandes retirées)
  if kept_indices is None:
    wvl_kept = (
        np.ones(len(WVL_PRS), dtype=bool)
        if WVL_PRS is not None
        else np.arange(n_kept_bands)
    )
    mae_full_spectrum = np.mean(np.abs(cube_gt - cube_predict), axis=(0, 1))
  else:
    mae_per_kept_band = np.mean(np.abs(cube_gt - cube_predict), axis=(0, 1))
    mae_full_spectrum = np.full(len(WVL_PRS), np.nan)
    mae_full_spectrum[kept_indices] = mae_per_kept_band
    wvl_kept = WVL_PRS[kept_indices]

  # 2. Indices RGB HSI basés sur les longueurs d'onde conservées
  idx_r_hsi = np.argmin(np.abs(wvl_kept - 665.0))
  idx_v_hsi = np.argmin(np.abs(wvl_kept - 560.0))
  idx_b_hsi = np.argmin(np.abs(wvl_kept - 490.0))

  idx_r_msi, idx_v_msi, idx_b_msi = 3, 2, 1

  # Fonction d'affichage RGB
  def to_rgb(cube, r, g, b, bounds=None):
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
    rgb = np.nan_to_num(rgb)
    for c in range(3):
      if bounds is None:
        v_min, v_max = (
            np.percentile(rgb[:, :, c], 2),
            np.percentile(rgb[:, :, c], 98),
        )
      else:
        v_min, v_max = bounds[c]
      rgb[:, :, c] = np.clip
      (rgb[:, :, c] - v_min) / (v_max - v_min + 1e-8), 0, 1
    return rgb

  rgb_msi = to_rgb(cube_msi, idx_r_msi, idx_v_msi, idx_b_msi)

  # Calcul des bornes communes HSI
  bounds_hsi = []
  for chan in [idx_r_hsi, idx_v_hsi, idx_b_hsi]:
    v_min = min(
        np.percentile(cube_gt[:, :, chan], 2),
        np.percentile(cube_predict[:, :, chan], 2),
    )
    v_max = max(
        np.percentile(cube_gt[:, :, chan], 98),
        np.percentile(cube_predict[:, :, chan], 98),
    )
    bounds_hsi.append((v_min, v_max))

  rgb_pred = to_rgb(
      cube_predict, idx_r_hsi, idx_v_hsi, idx_b_hsi, bounds=bounds_hsi
  )
  rgb_gt = to_rgb(cube_gt, idx_r_hsi, idx_v_hsi, idx_b_hsi, bounds=bounds_hsi)

  # 3. Calcul des cartes spatiale d'erreur et d'incertitude (Moyenne spectrale)
  mae_map = np.mean(np.abs(cube_gt - cube_predict), axis=-1)

  if cube_uncertainty is not None:
    unc_map = np.mean(cube_uncertainty, axis=-1)
    diff_map = (
        mae_map - unc_map
    )  # Positive = Surconfiance (Erreur > Incertitude)

    # Dynamique commune (0 à 98e percentile max entre Erreur et Incertitude)
    v_min_err = 0.0
    v_max_err = max(np.percentile(mae_map, 98), np.percentile(unc_map, 98))

    # Dynamique symétrique centrée sur 0 pour la différence
    v_diff_max = np.percentile(np.abs(diff_map), 98)

  # FIGURE 2x3
  fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
  clean_title = model_name.replace("\n", " ")

  # TITRE GLOBAL
  fig.suptitle(
      f"Planche de Synthèse Globale — {clean_title}\n"
      f"RMSE : {img_rmse:.4f} | MAE : {img_mae:.4f} | SAM : {img_sam:.4f} rad"
      f" ({np.degrees(img_sam):.2f}°) | PSNR : {img_psnr:.2f} dB | SSIM :"
      f" {img_ssim:.4f} | ERGAS : {img_ergas:.2f}",
      fontsize=14,
      weight="bold",
  )

  # LIGNE 1 : Images RGB
  axes[0, 0].imshow(rgb_msi)
  axes[0, 0].set_title(
      "RGB — Entrée MSI (Sentinel-2)\n Min:"
      f" {np.min(rgb_msi):.2f} | Max: {np.max(rgb_msi):.2f} | Mean:"
      f" {np.mean(rgb_msi):.2f}",
      fontsize=11,
      pad=6,
  )
  axes[0, 0].axis("off")

  axes[0, 1].imshow(rgb_pred)
  axes[0, 1].set_title(
      f"RGB — HSI Prédite ({n_kept_bands} bandes)\n Min:"
      f" {np.min(rgb_pred):.2f} | Max: {np.max(rgb_pred):.2f} | Mean:"
      f" {np.mean(rgb_pred):.2f}",
      fontsize=11,
      pad=6,
  )
  axes[0, 1].axis("off")

  axes[0, 2].imshow(rgb_gt)
  axes[0, 2].set_title(
      f"RGB — HSI Vérité Terrain ({n_kept_bands} bandes)\n Min:"
      f" {np.min(rgb_gt):.2f} | Max: {np.max(rgb_gt):.2f} | Mean:"
      f" {np.mean(rgb_gt):.2f}",
      fontsize=11,
      pad=6,
  )
  axes[0, 2].axis("off")

  # LIGNE 2 : Cartes d'Erreur, Incertitude et Résidu

  # [2, 1] Carte d'Erreur Réelle MAE
  im_err = axes[1, 0].imshow(
      mae_map, cmap="inferno", vmin=v_min_err, vmax=v_max_err
  )
  axes[1, 0].set_title(
      "Erreur Réelle — MAE Spatiale\n"
      f"Min: {np.min(mae_map):.4f} | Max: {np.max(mae_map):.4f} | Mean:"
      f" {np.mean(mae_map):.4f}",
      fontsize=11,
      pad=6,
  )
  axes[1, 0].axis("off")
  fig.colorbar(im_err, ax=axes[1, 0], fraction=0.046, pad=0.04)

  # [2, 2] Carte d'Incertitude Prédite
  if cube_uncertainty is not None:
    im_unc = axes[1, 1].imshow(
        unc_map, cmap="inferno", vmin=v_min_err, vmax=v_max_err
    )
    axes[1, 1].set_title(
        "Incertitude Prédite — $\hat{b}$\n"
        f"Min: {np.min(unc_map):.4f} | Max: {np.max(unc_map):.4f} | Mean:"
        f" {np.mean(unc_map):.4f}",
        fontsize=11,
        pad=6,
    )
    fig.colorbar(im_unc, ax=axes[1, 1], fraction=0.046, pad=0.04)
  else:
    axes[1, 1].text(
        0.5,
        0.5,
        "Incertitude non fournie",
        ha="center",
        va="center",
        fontsize=12,
    )
    axes[1, 1].set_title("Incertitude Prédite", fontsize=11, pad=6)
  axes[1, 1].axis("off")

  # [2, 3] Carte de Résidu d'Incertitude (Erreur - Incertitude)
  if cube_uncertainty is not None:
    im_diff = axes[1, 2].imshow(
        diff_map, cmap="coolwarm", vmin=-v_diff_max, vmax=v_diff_max
    )
    axes[1, 2].set_title(
        "Résidu : Erreur - Incertitude\n"
        f"Min: {np.min(diff_map):.4f} | Max: {np.max(diff_map):.4f} | Mean:"
        f" {np.mean(diff_map):.4f}",
        fontsize=11,
        pad=6,
    )
    fig.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)
  else:
    axes[1, 2].text(
        0.5,
        0.5,
        "Incertitude non fournie",
        ha="center",
        va="center",
        fontsize=12,
    )
    axes[1, 2].set_title("Différence (Erreur - $\hat{b}$)", fontsize=11, pad=6)
  axes[1, 2].axis("off")

  # Sauvegarde locale
  if plot_dir is not None:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = plot_dir / f"{save_name}.png"
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")

  # Envoi sur WandB
  if log_to_wandb and wandb is not None and wandb.run is not None:
    wandb.log({f"synthesis_plots/{save_name}": wandb.Image(fig)})

  plt.close(fig)