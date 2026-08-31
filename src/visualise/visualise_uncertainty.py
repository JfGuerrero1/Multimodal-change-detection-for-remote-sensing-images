import argparse
from pathlib import Path
import sys
import matplotlib
import torch
import xarray as xr
import yaml

# Sécurité pour les serveurs sans GUI
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Ajoute la racine du projet au sys.path de Python
root_dir = (
    Path(__file__).resolve().parents[2]
    if len(Path(__file__).resolve().parents) > 2
    else Path(__file__).resolve().parent
)
if str(root_dir) not in sys.path:
  sys.path.insert(0, str(root_dir))

import wandb
from src.constants import DW_INFO, INTERP_MATRIX, SRF_MATRIX, WVL_PRS, WVL_S2
from src.metrics_and_loss.metrics import (
    compute_ergas,
    compute_mae,
    compute_mse,
    compute_mrae,
    compute_psnr,
    compute_rmse,
    compute_sam,
    compute_sam_map,
    compute_ssim_multiband,
)
from src.models import (
    DualBranchNAFNet,
    GradualExpansionUNet,
    GradualExpansionUNet_residual,
)

import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path de Python
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib
# Sécurité pour les serveurs sans GUI
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import wandb

# Import des constantes et métriques nécessaires
from src.constants import WVL_PRS, DW_INFO, WVL_S2


def visualise_synthesis_uncertainty(
    data, save_name, plot_dir, kept_indices=None, log_to_wandb=False
):
    """Génère une planche de synthèse 3x3 avec profils spectraux complets (NaN sur bandes supprimées)."""
    
    cube_gt = data["cube_gt"]  # Shape (H, W, n_kept)
    cube_predict = data["cube_predict"]  # Shape (H, W, n_kept)
    cube_msi = data["cube_msi"]  # Shape (H, W, n_msi)
    cube_uncertainty = data["cube_uncertainty"] # Shape (H, W, n_kept)
    model_name = data.get("model name", "Modèle ML")

    # Récupération ou calcul automatique des métriques
    img_rmse = data.get("img_rmse", compute_rmse(cube_predict, cube_gt))
    img_mae = data.get("img_mae", compute_mae(cube_predict, cube_gt))
    img_sam = data.get("img_sam", compute_sam(cube_gt, cube_predict))
    img_ssim = data.get("img_ssim", compute_ssim_multiband(cube_predict, cube_gt))
    img_ergas = data.get("img_ergas", compute_ergas(cube_predict, cube_gt))
    img_psnr = data.get("img_psnr", compute_psnr(cube_gt, cube_predict))

    n_kept_bands = cube_gt.shape[-1]

    # 1. Gestion des longueurs d'onde (WVL)
    if WVL_PRS is None:
        full_wvl = np.arange(n_kept_bands)
        wvl_kept = np.arange(n_kept_bands)
        full_mask = np.zeros(n_kept_bands, dtype=bool)
    else:
        full_wvl = WVL_PRS
        if kept_indices is None:
            wvl_kept = full_wvl
            full_mask = np.zeros(len(full_wvl), dtype=bool)
        else:
            wvl_kept = full_wvl[kept_indices]
            full_mask = ~np.isin(full_wvl, wvl_kept)

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
                v_min, v_max = np.percentile(rgb[:, :, c], 2), np.percentile(rgb[:, :, c], 98)
            else:
                v_min, v_max = bounds[c]
            rgb[:, :, c] = np.clip((rgb[:, :, c] - v_min) / (v_max - v_min + 1e-8), 0, 1)
        return rgb

    rgb_msi = to_rgb(cube_msi, idx_r_msi, idx_v_msi, idx_b_msi)

    # Calcul des bornes communes HSI
    bounds_hsi = []
    for chan in [idx_r_hsi, idx_v_hsi, idx_b_hsi]:
        v_min = min(np.percentile(cube_gt[:, :, chan], 2), np.percentile(cube_predict[:, :, chan], 2))
        v_max = max(np.percentile(cube_gt[:, :, chan], 98), np.percentile(cube_predict[:, :, chan], 98))
        bounds_hsi.append((v_min, v_max))

    rgb_pred = to_rgb(cube_predict, idx_r_hsi, idx_v_hsi, idx_b_hsi, bounds=bounds_hsi)
    rgb_gt = to_rgb(cube_gt, idx_r_hsi, idx_v_hsi, idx_b_hsi, bounds=bounds_hsi)

    # --- CALCULS POUR LES CARTES (Ligne 2) ---
    unc_map = np.mean(cube_uncertainty, axis=-1)
    mae_map = np.mean(np.abs(cube_gt - cube_predict), axis=-1)
    diff_map = unc_map - mae_map

    # --- CALCULS PAR LONGUEUR D'ONDE SUR LE SPECTRE COMPLET (Ligne 3) ---
    unc_per_wvl_kept = np.mean(cube_uncertainty, axis=(0, 1))
    mae_per_wvl_kept = np.mean(np.abs(cube_gt - cube_predict), axis=(0, 1))
    diff_per_wvl_kept = unc_per_wvl_kept - mae_per_wvl_kept

    if WVL_PRS is not None and kept_indices is not None:
        unc_full = np.full(len(full_wvl), np.nan)
        unc_full[kept_indices] = unc_per_wvl_kept

        mae_full = np.full(len(full_wvl), np.nan)
        mae_full[kept_indices] = mae_per_wvl_kept

        diff_full = np.full(len(full_wvl), np.nan)
        diff_full[kept_indices] = diff_per_wvl_kept
    else:
        unc_full = unc_per_wvl_kept
        mae_full = mae_per_wvl_kept
        diff_full = diff_per_wvl_kept

    # FIGURE 3x3
    fig, axes = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True)
    clean_title = model_name.replace("\n", " ")
    
    fig.suptitle(
        f"Planche de Synthèse & Incertitude — {clean_title}\n"
        f"RMSE : {img_rmse:.4f} | MAE : {img_mae:.4f} | PSNR : {img_psnr:.2f} dB | SSIM : {img_ssim:.4f} | "
        f"SAM : {img_sam:.4f} rad ({np.degrees(img_sam):.2f}°) | ERGAS : {img_ergas:.4f}",
        fontsize=14, weight="bold",
    )

    # --- LIGNE 1 : Images RGB ---
    axes[0, 0].imshow(rgb_msi)
    axes[0, 0].set_title("RGB — Entrée MSI", fontsize=11, pad=6)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_pred)
    axes[0, 1].set_title("RGB — HSI Prédite", fontsize=11, pad=6)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb_gt)
    axes[0, 2].set_title(f"RGB — HSI Vérité Terrain\nmin: {np.min(rgb_gt):.2f}, max: {np.max(rgb_gt):.2f} | mean: {np.mean(rgb_gt):.2f}", fontsize=10, pad=6)
    axes[0, 2].axis("off")

    # --- LIGNE 2 : Cartes (Incertitude, Erreur réelle, Différence) ---
    vmax_unc = max(np.percentile(unc_map, 98), 0.01)
    vmax_mae = max(np.percentile(mae_map, 98), 0.01)
    vmax = max(vmax_unc, vmax_mae)

    # 1. Incertitude
    im_unc = axes[1, 0].imshow(unc_map, cmap="inferno", vmin=0, vmax=vmax)
    axes[1, 0].set_title(f"Incertitude Prédite\nMin: {np.min(unc_map):.2f}, Max: {np.max(unc_map):.2f} | Mean: {np.mean(unc_map):.2f}", fontsize=10, pad=6)
    axes[1, 0].axis("off")
    fig.colorbar(im_unc, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 2. Erreur Réelle (MAE)
    im_mae = axes[1, 1].imshow(mae_map, cmap="inferno", vmin=0, vmax=vmax)
    axes[1, 1].set_title(f"Erreur Réelle (MAE)\nMin: {np.min(mae_map):.2f}, Max: {np.max(mae_map):.2f} | Mean: {np.mean(mae_map):.2f}", fontsize=10, pad=6)
    axes[1, 1].axis("off")
    fig.colorbar(im_mae, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 3. Différence
    v_diff = max(np.max(np.abs(diff_map)), 0.01)
    im_diff = axes[1, 2].imshow(diff_map, cmap="seismic", vmin=-v_diff, vmax=v_diff)
    axes[1, 2].set_title(f"Différence (Incertitude - Erreur)\nMin: {-v_diff:.2f}, Max: {v_diff:.2f} | Mean: {np.mean(diff_map):.2f}", fontsize=10, pad=6)
    axes[1, 2].axis("off")
    fig.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # --- LIGNE 3 : Profils spectraux sur wvl complet (avec NaN et zones grisées) ---
    def plot_with_gray_zones(ax, x_vals, values, title, ylabel, color):
        # Tracé sur le spectre complet (les NaN créent des ruptures de lignes naturelles)
        ax.plot(x_vals, values, color=color, linewidth=2)

        # Griser les zones non utilisées si le masque global existe
        if full_mask is not None and full_wvl is not None:
            in_gray = False
            start_gray = full_wvl[0]
            for i, masked in enumerate(full_mask):
                if masked and not in_gray:
                    in_gray = True
                    start_gray = full_wvl[i]
                elif not masked and in_gray:
                    in_gray = False
                    ax.axvspan(start_gray, full_wvl[i-1], color='gray', alpha=0.3, lw=0)
            if in_gray:
                ax.axvspan(start_gray, full_wvl[-1], color='gray', alpha=0.3, lw=0)
            
            ax.set_xlim(full_wvl[0], full_wvl[-1])

        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("Longueur d'onde (nm)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Application sur la ligne 3 avec full_wvl et les vecteurs pleins de NaN
    plot_with_gray_zones(axes[2, 0], full_wvl, unc_full, "Incertitude Moyenne par WVL", "Incertitude", 'red')
    plot_with_gray_zones(axes[2, 1], full_wvl, mae_full, "MAE Moyenne par WVL", "MAE", 'red')
    plot_with_gray_zones(axes[2, 2], full_wvl, diff_full, "Différence (Unc-MAE) par WVL", "Différence", 'blue')

    # Sauvegarde et log
    # Sauvegarde sur le disque
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        # On s'assure d'ajouter .png uniquement pour la sauvegarde fichier
        file_path = plot_dir / f"{save_name}.png" if not str(save_name).endswith('.png') else plot_dir / save_name
        plt.savefig(file_path, dpi=150, bbox_inches="tight")

    # Envoi sécurisé vers Weights & Biases
    if log_to_wandb and wandb is not None and wandb.run is not None:
        # Nettoyage de la clé (suppression de .png si présent)
        clean_key = str(save_name).replace('.png', '')
        
        # Log de l'image Matplotlib
        wandb.log({f"visualisations/{clean_key}": wandb.Image(fig)})

    # Libération systématique de la mémoire
    plt.close(fig)

def compute_f1_score(pred_map, gt_map, eps=1e-8):
  """Calcule le F1-score (ou coefficient de Dice) pour une carte binaire."""
  tp = np.sum((pred_map == 1) & (gt_map == 1))
  fp = np.sum((pred_map == 1) & (gt_map == 0))
  fn = np.sum((pred_map == 0) & (gt_map == 1))

  precision = tp / (tp + fp + eps)
  recall = tp / (tp + fn + eps)

  f1 = 2 * (precision * recall) / (precision + recall + eps)
  return f1, precision, recall


def build_model_reconstruction(
    config: dict, n_msi: int, n_hsi: int
) -> torch.nn.Module:
  model_cfg = config["model"]
  name = model_cfg.get("name", "").lower()

  if name == "gradualexpansionunet":
    return GradualExpansionUNet(
        in_msi=n_msi,
        in_hsi=n_hsi,
        interpolation_mode=model_cfg.get("interpolation_mode", "Bilinear"),
        base_channel=model_cfg.get("base_channel", 64),
        activation=model_cfg.get("activation", "silu"),
        with_batch_norm=model_cfg.get("with_batch_norm", True),
        with_mlp_spectral=model_cfg.get("with_mlp_spectral", False),
        final_activation=model_cfg.get("final_activation", None),
    )
  elif name == "gradualexpansionunet_res":
    return GradualExpansionUNet_residual(
        in_msi=n_msi,
        in_hsi=n_hsi,
        interpolation_mode=model_cfg.get("interpolation_mode", "Bilinear"),
        base_channel=model_cfg.get("base_channel", 64),
        activation=model_cfg.get("activation", "silu"),
        with_batch_norm=model_cfg.get("with_batch_norm", True),
        with_mlp_spectral=model_cfg.get("with_mlp_spectral", False),
        final_activation=model_cfg.get("final_activation", None),
    )
  else:
    raise ValueError(f"❌ Modèle non reconnu : '{name}'")


def build_model_uncertainty(
    config: dict, n_msi: int, out_channels: int
) -> torch.nn.Module:
  model_cfg = config["model_uncertainty"]
  name = model_cfg.get("name", "").lower()

  if name == "dualbranchnafnet":
    return DualBranchNAFNet(
        n_msi=n_msi,
        n_hsi=out_channels,
        out_channels=out_channels,
        width=model_cfg.get("base_channel", 64),
        middle_blk_num=model_cfg.get("middle_blk_num", 1),
        enc_blk_nums=model_cfg.get("enc_blk_nums", []),
        dec_blk_nums=model_cfg.get("dec_blk_nums", []),
        drop_out_rate=model_cfg.get("drop_out_rate", 0.0),
        final_op=model_cfg.get("final_activation", "softplus"),
    )
  else:
    raise ValueError(f"❌ Modèle d'incertitude non reconnu : '{name}'")


from pathlib import Path
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import argparse
import yaml

# --- Fonctions utilitaires (supposées définies dans ton environnement) ---
# compute_mae, compute_sam, compute_sam_map, compute_f1_score, 
# build_model_reconstruction, build_model_uncertainty, SRF_MATRIX, INTERP_MATRIX, WVL_PRS, root_dir

def compute_dynamic_nfa_threshold(map_data, nfa_target=0.01):
  """Calcule le seuil dynamique pour obtenir le taux de fausse alerte (NFA) ciblé."""
  valid_values = map_data[np.isfinite(map_data)]
  percentile = (1.0 - nfa_target) * 100.0
  return np.percentile(valid_values, percentile)


def create_confusion_rgb(pred_map, gt_map):
  """Génère une carte RGB pour la confusion :
  - Vert : Vrais Positifs (TP)
  - Rouge : Faux Positifs (FP)
  - Bleu : Faux Négatifs (FN)
  - Noir : Vrais Négatifs (TN)
  """
  h, w = gt_map.shape
  rgb = np.zeros((h, w, 3), dtype=np.float32)

  tp = (pred_map == 1) & (gt_map == 1)
  fp = (pred_map == 1) & (gt_map == 0)
  fn = (pred_map == 0) & (gt_map == 1)

  rgb[tp] = [0, 1, 0]
  rgb[fp] = [1, 0, 0]
  rgb[fn] = [0, 0.4, 1]

  return rgb


def visualise_changement_complet_nfa(
    data_t1,
    data_t2,
    save_path,
    gt_change_map=None,
    metric_type="mae",
    nfa_target=0.01,
    fixed_tau_brut=None,
    fixed_tau_unc=None,
    WVL_PRS=None,
    kept_indices=None,
    calib_t1=None,
):
  cube_y1 = data_t1["cube_gt"]
  cube_y2_pred = data_t2["cube_predict"]
  cube_msi_t2 = data_t2["cube_msi"]
  cube_uncertainty_t2 = data_t2["cube_uncertainty"]
  model_name = data_t2.get("model name", "Modèle ML")

  mae_global = compute_mae(cube_y2_pred, cube_y1)
  sam_global = compute_sam(cube_y1, cube_y2_pred)

  E_l1 = np.mean(np.abs(cube_y1 - cube_y2_pred), axis=-1)
  E_sam = compute_sam_map(cube_y1, cube_y2_pred)

  if metric_type.lower() == "sam":
    E = E_sam
    b_amp = np.mean(cube_uncertainty_t2, axis=-1) / (
        np.linalg.norm(cube_y2_pred, axis=-1) + 1e-8
    ) + 1e-8
  else:
    E = E_l1
    b_amp = np.mean(cube_uncertainty_t2, axis=-1) + 1e-8

  D = E / b_amp

  if fixed_tau_brut is not None and fixed_tau_unc is not None:
    tau_brut = fixed_tau_brut
    tau_unc = fixed_tau_unc
  elif calib_t1 is not None:
    cube_y1_gt = calib_t1["cube_gt"]
    cube_y1_pred = calib_t1["cube_predict"]
    cube_uncertainty_t1 = calib_t1["cube_uncertainty"]

    if metric_type.lower() == "sam":
      E_t1 = compute_sam_map(cube_y1_gt, cube_y1_pred)
      b_amp_t1 = np.mean(cube_uncertainty_t1, axis=-1) / (
          np.linalg.norm(cube_y1_pred, axis=-1) + 1e-8
      ) + 1e-8
    else:
      E_t1 = np.mean(np.abs(cube_y1_gt - cube_y1_pred), axis=-1)
      b_amp_t1 = np.mean(cube_uncertainty_t1, axis=-1) + 1e-8

    D_t1 = E_t1 / b_amp_t1

    tau_brut = compute_dynamic_nfa_threshold(E_t1, nfa_target=nfa_target)
    tau_unc = compute_dynamic_nfa_threshold(D_t1, nfa_target=nfa_target)

  change_map_raw = E > tau_brut
  change_map_unc = D > tau_unc
  false_positives_filtered = change_map_raw.astype(int) - change_map_unc.astype(
      int
  )

  # --- Calcul des pourcentages ---
  total_px = change_map_raw.size
  pct_raw = (np.sum(change_map_raw) / total_px) * 100.0
  pct_unc = (np.sum(change_map_unc) / total_px) * 100.0
  
  gt_str = ""
  if gt_change_map is not None:
    pct_gt = (np.sum(gt_change_map) / total_px) * 100.0
    gt_str = f" | GT : {pct_gt:.2f}%"

  n_kept_bands = cube_y1.shape[-1]
  wvl_kept = WVL_PRS if WVL_PRS is not None else np.arange(n_kept_bands)
  if kept_indices is not None and WVL_PRS is not None:
    wvl_kept = WVL_PRS[kept_indices]

  idx_r_hsi = np.argmin(np.abs(wvl_kept - 665.0))
  idx_v_hsi = np.argmin(np.abs(wvl_kept - 560.0))
  idx_b_hsi = np.argmin(np.abs(wvl_kept - 490.0))
  idx_r_msi, idx_v_msi, idx_b_msi = 3, 2, 1

  def to_rgb(cube, r, g, b):
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
    rgb = np.nan_to_num(rgb)
    for c_idx in range(3):
      v_min, v_max = (
          np.percentile(rgb[:, :, c_idx], 2),
          np.percentile(rgb[:, :, c_idx], 98),
      )
      rgb[:, :, c_idx] = np.clip(
          (rgb[:, :, c_idx] - v_min) / (v_max - v_min + 1e-8), 0, 1
      )
    return rgb

  rgb_y1 = to_rgb(cube_y1, idx_r_hsi, idx_v_hsi, idx_b_hsi)
  rgb_y2_pred = to_rgb(cube_y2_pred, idx_r_hsi, idx_v_hsi, idx_b_hsi)
  rgb_msi_t2 = to_rgb(cube_msi_t2, idx_r_msi, idx_v_msi, idx_b_msi)

  fig, axes = plt.subplots(4, 3, figsize=(16, 17))

  # --- Intégration des pourcentages dans le titre global ---
  fig.suptitle(
      f"Détection de changement Y1 vs Ŷ2 ({model_name})\n"
      f"Métrique globale (Y1 - Ŷ2) -> MAE : {mae_global:.4f} | SAM :"
      f" {sam_global:.4f} rad ({np.degrees(sam_global):.2f}°)\n"
      f"NFA : {nfa_target*100:.1f}% | Mode : {metric_type.upper()}\n"
      f"Changement détecté -> Brut : {pct_raw:.2f}% | Filtré (Incert.) : {pct_unc:.2f}%{gt_str}",
      fontsize=12,
      fontweight="bold",
  )

  axes[0, 0].imshow(rgb_y1)
  axes[0, 0].set_title("1. HSI Y1 (T1 Réel)")
  axes[0, 1].imshow(rgb_msi_t2)
  axes[0, 1].set_title("2. MSI X2 (T2 Entrée)")
  axes[0, 2].imshow(rgb_y2_pred)
  axes[0, 2].set_title("3. HSI Ŷ2 (T2 Prédit)")

  vmax_b = np.percentile(b_amp[np.isfinite(b_amp)], 98)
  vmax_e = np.percentile(E[np.isfinite(E)], 98)
  vmax_d = np.percentile(D[np.isfinite(D)], 98)

  im1 = axes[1, 0].imshow(b_amp, cmap="magma", vmax=vmax_b)
  axes[1, 0].set_title("4. Incertitude b (T2)")
  plt.colorbar(im1, ax=axes[1, 0])

  im2 = axes[1, 1].imshow(E, cmap="magma", vmax=vmax_e)
  axes[1, 1].set_title(f"5. Changement brut E ({metric_type.upper()})")
  plt.colorbar(im2, ax=axes[1, 1])

  im3 = axes[1, 2].imshow(D, cmap="magma", vmax=vmax_d)
  axes[1, 2].set_title("6. Changement normalisé D (E / b)")
  plt.colorbar(im3, ax=axes[1, 2])

  axes[2, 0].imshow(change_map_raw, cmap="gray")
  axes[2, 0].set_title(f"7. Brut : E > {tau_brut:.3f}")

  axes[2, 1].imshow(change_map_unc, cmap="gray")
  axes[2, 1].set_title(f"8. Filtré : D > {tau_unc:.3f}")

  im4 = axes[2, 2].imshow(false_positives_filtered, cmap="coolwarm")
  axes[2, 2].set_title(
      "9. Impact de l'Incertitude (Brut - Filtré)\nRouge: Erreur éliminée |"
      " Bleu: Changement révélé"
  )

  if gt_change_map is not None:
    f1_raw, prec_raw, rec_raw = compute_f1_score(change_map_raw, gt_change_map)
    f1_unc, prec_unc, rec_unc = compute_f1_score(change_map_unc, gt_change_map)

    axes[3, 0].imshow(gt_change_map, cmap="gray")
    axes[3, 0].set_title("10. Carte GT (scene-cd-binary.nc)")

    rgb_raw_conf = create_confusion_rgb(change_map_raw, gt_change_map)
    axes[3, 1].imshow(rgb_raw_conf)
    axes[3, 1].set_title(
        f"11. Performance Brut (E)\nF1: {f1_raw:.4f} | Prec: {prec_raw:.3f} |"
        f" Rec: {rec_raw:.3f}\nVert: TP | Rouge: FP | Bleu: FN | Noir: TN"
    )

    rgb_unc_conf = create_confusion_rgb(change_map_unc, gt_change_map)
    axes[3, 2].imshow(rgb_unc_conf)
    axes[3, 2].set_title(
        f"12. Performance Filtré (D)\nF1: {f1_unc:.4f} | Prec: {prec_unc:.3f} |"
        f" Rec: {rec_unc:.3f}\nVert: TP | Rouge: FP | Bleu: FN | Noir: TN"
    )
  else:
    for j in range(3):
      axes[3, j].text(
          0.5,
          0.5,
          "GT non disponible",
          ha="center",
          va="center",
          fontsize=12,
      )

  for ax in axes.ravel():
    ax.axis("off")

  plt.tight_layout()

  save_path = Path(save_path)
  save_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(save_path, dpi=150, bbox_inches="tight")
  plt.close(fig)

  return {
      "E": E,
      "D": D,
      "tau_brut_dyn": tau_brut,
      "tau_unc_dyn": tau_unc,
      "change_map_raw": change_map_raw,
      "change_map_unc": change_map_unc,
  }

def get_kept_indices(wvl, bad_bands_ranges):
  if not bad_bands_ranges:
    return np.arange(len(wvl))
  mask = np.ones(len(wvl), dtype=bool)
  for b_min, b_max in bad_bands_ranges:
    mask = mask & ~((wvl >= b_min) & (wvl <= b_max))
  return np.where(mask)[0]


def process_nc_file(
    scene_dir,
    model_rec,
    model_unc,
    SRF_MATRIX,
    INTERP_MATRIX,
    device,
    kept_indices=None,
    patch_size=256,
    prefix="Brasilia",
    is_normalized=True,
):
  scene_dir = Path(scene_dir)
  scene_name = scene_dir.name

  nc_path_msi_t1 = scene_dir / f"{scene_name}-before-s2.nc"
  nc_path_msi_t2 = scene_dir / f"{scene_name}-after-s2.nc"
  nc_path_hsi_t1 = scene_dir / f"{scene_name}-before-prs.nc"
  nc_path_hsi_t2 = scene_dir / f"{scene_name}-after-prs.nc"
  nc_path_cd = scene_dir / f"{scene_name}-cd-binary.nc"

  # --- Chargement T1 ---
  ds_msi1 = xr.open_dataset(nc_path_msi_t1)
  msi_1 = ds_msi1["sr"].values
  ds_msi1.close()

  ds_hsi1 = xr.open_dataset(nc_path_hsi_t1)
  hsi_gt_1 = ds_hsi1["sr"].values
  ds_hsi1.close()

  if hsi_gt_1.shape[0] < hsi_gt_1.shape[-1]:
    hsi_gt_1 = np.moveaxis(hsi_gt_1, 0, -1)
  if msi_1.shape[0] < msi_1.shape[-1]:
    msi_1 = np.moveaxis(msi_1, 0, -1)

  # --- Chargement T2 ---
  ds_msi2 = xr.open_dataset(nc_path_msi_t2)
  msi_2 = ds_msi2["sr"].values
  ds_msi2.close()

  ds_hsi2 = xr.open_dataset(nc_path_hsi_t2)
  hsi_gt_2 = ds_hsi2["sr"].values
  ds_hsi2.close()

  if hsi_gt_2.shape[0] < hsi_gt_2.shape[-1]:
    hsi_gt_2 = np.moveaxis(hsi_gt_2, 0, -1)
  if msi_2.shape[0] < msi_2.shape[-1]:
    msi_2 = np.moveaxis(msi_2, 0, -1)

  # --- Chargement GT Change Map ---
  gt_change_map = None
  if nc_path_cd.exists():
    ds_cd = xr.open_dataset(nc_path_cd)
    gt_change_map = ds_cd["change"].values
    ds_cd.close()
    if gt_change_map.ndim == 3:
      gt_change_map = gt_change_map.squeeze()

  # --- 1. NORMALISATION GLOBALE ---
  if is_normalized:
    h_full, w_full, c_hsi_full = hsi_gt_1.shape
    c_msi = msi_1.shape[-1]

    hyper_2d_1 = hsi_gt_1.reshape(-1, c_hsi_full)
    if SRF_MATRIX.shape[1] == c_hsi_full:
      scene_multi_sim_1 = np.dot(hyper_2d_1, SRF_MATRIX.T).reshape(h_full, w_full, c_msi)
    else:
      scene_multi_sim_1 = np.dot(hyper_2d_1, SRF_MATRIX).reshape(h_full, w_full, c_msi)

    mean_scene_1 = np.mean(msi_1, axis=(0, 1), keepdims=True)
    std_scene_1 = np.std(msi_1, axis=(0, 1), keepdims=True)
    mean_scene_2 = np.mean(msi_2, axis=(0, 1), keepdims=True)
    std_scene_2 = np.std(msi_2, axis=(0, 1), keepdims=True)
    mean_sim_1 = np.mean(scene_multi_sim_1, axis=(0, 1), keepdims=True)
    std_sim_1 = np.std(scene_multi_sim_1, axis=(0, 1), keepdims=True)

    msi_1 = (msi_1 - mean_scene_1) / (std_scene_1 + 1e-8) * std_sim_1 + mean_sim_1
    msi_2 = (msi_2 - mean_scene_2) / (std_scene_2 + 1e-8) * std_sim_1 + mean_sim_1

  # --- 2. ROGNAGE ---
  h, w, c_msi = msi_2.shape
  h_crop = h - (h % patch_size)
  w_crop = w - (w % patch_size)

  msi_1 = msi_1[:h_crop, :w_crop, :]
  msi_2 = msi_2[:h_crop, :w_crop, :]

  if kept_indices is not None:
    hsi_gt_1_filtered = hsi_gt_1[:h_crop, :w_crop, kept_indices]
    hsi_gt_2_filtered = hsi_gt_2[:h_crop, :w_crop, kept_indices]
  else:
    hsi_gt_1_filtered = hsi_gt_1[:h_crop, :w_crop, :]
    hsi_gt_2_filtered = hsi_gt_2[:h_crop, :w_crop, :]

  if gt_change_map is not None:
    gt_change_map = gt_change_map[:h_crop, :w_crop]

  # --- Interpolation linéaire globale ---
  interp_matrix_filtered = (
      INTERP_MATRIX[kept_indices, :]
      if kept_indices is not None
      else INTERP_MATRIX
  )
  msi_chw = np.moveaxis(msi_2, -1, 0)
  interp_2d = interp_matrix_filtered @ msi_chw.reshape(c_msi, -1)
  hsi_interp_2 = interp_2d.reshape(-1, h_crop, w_crop).transpose(1, 2, 0)

  msi_chw = np.moveaxis(msi_1, -1, 0)
  interp_2d = interp_matrix_filtered @ msi_chw.reshape(c_msi, -1)
  hsi_interp_1 = interp_2d.reshape(-1, h_crop, w_crop).transpose(1, 2, 0)

  model_rec.eval()
  model_unc.eval()

  patches_t1 = {}
  patches_t2 = {}

  num_patches_y = h_crop // patch_size
  num_patches_x = w_crop // patch_size

  for i in range(num_patches_y):
    for j in range(num_patches_x):
      patch_idx = i * num_patches_x + j
      patch_id_str = f"{prefix.lower()}_patch_{patch_idx:04d}"

      y_start, y_end = i * patch_size, (i + 1) * patch_size
      x_start, x_end = j * patch_size, (j + 1) * patch_size

      msi1_p = msi_1[y_start:y_end, x_start:x_end, :]
      gt1_p = hsi_gt_1_filtered[y_start:y_end, x_start:x_end, :]
      interp1_p = hsi_interp_1[y_start:y_end, x_start:x_end, :]

      msi2_p = msi_2[y_start:y_end, x_start:x_end, :]
      gt2_p = hsi_gt_2_filtered[y_start:y_end, x_start:x_end, :]
      interp2_p = hsi_interp_2[y_start:y_end, x_start:x_end, :]

      msi2_t = torch.from_numpy(np.moveaxis(msi2_p, -1, 0)).float().unsqueeze(0).to(device)
      interp2_t = torch.from_numpy(np.moveaxis(interp2_p, -1, 0)).float().unsqueeze(0).to(device)
      msi1_t = torch.from_numpy(np.moveaxis(msi1_p, -1, 0)).float().unsqueeze(0).to(device)
      interp1_t = torch.from_numpy(np.moveaxis(interp1_p, -1, 0)).float().unsqueeze(0).to(device)

      with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=True):
          pred_t2_t = model_rec(msi2_t, interp2_t)
          unc_input_t2_t = torch.cat([msi2_t, pred_t2_t], dim=1)
          u_hat_t2_t = model_unc(unc_input_t2_t)

          pred_t1_t = model_rec(msi1_t, interp1_t)
          unc_input_t1_t = torch.cat([msi1_t, pred_t1_t], dim=1)
          u_hat_t1_t = model_unc(unc_input_t1_t)

      pred_hwc_2_p = pred_t2_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
      unc_hwc_2_p = u_hat_t2_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
      pred_hwc_1_p = pred_t1_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
      unc_hwc_1_p = u_hat_t1_t.squeeze(0).permute(1, 2, 0).cpu().numpy()



      patches_t1[patch_id_str] = {
          "cube_gt": gt1_p,
          "cube_predict": pred_hwc_1_p,
          "cube_msi": msi1_p,
          "cube_uncertainty": unc_hwc_1_p,
      }

      patches_t2[patch_id_str] = {
          "cube_gt": gt2_p,
          "cube_predict": pred_hwc_2_p,
          "cube_msi": msi2_p,
          "cube_uncertainty": unc_hwc_2_p,
      }

  print(f"✅ Découpage terminé : {len(patches_t2)} patchs créés.")
  return patches_t1, patches_t2, gt_change_map, (h_crop, w_crop)


def setup_parser():
  parser = argparse.ArgumentParser(
      description="Pipeline d'évaluation et détection de changement"
  )
  parser.add_argument(
      "scene", type=str, help="Nom du dossier de la scène (ex: brasilia)"
  )
  parser.add_argument(
      "--config",
      type=str,
      default="config_eval.yaml",
      help="Chemin du fichier YAML",
  )
  parser.add_argument(
      "--no-normalization",
      action="store_true",
      help="Désactive la normalisation des images MSI",
  )
  # Modification ici : ajout du type float et d'une valeur par défaut
  parser.add_argument(
      "--nfa",
      type=float,
      default=0.01,
      help="Cible NFA pour le seuillage dynamique (ex: 0.01 pour 1%%)",
  )
  return parser.parse_args()


def load_config(config_path: str) -> dict:
  with open(config_path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


if __name__ == "__main__":
  args = setup_parser()
  cfg = load_config(args.config)

  # Récupération de la valeur du NFA passée en argument
  nfa_target_val = args.nfa

  eval_cfg = cfg["evaluation"]
  data_cfg = cfg["data"]

  is_normalized = not args.no_normalization
  if "is_normalized" in data_cfg:
    is_normalized = data_cfg["is_normalized"]

  wl_filter = cfg.get("wavelength_filtering", {})
  if wl_filter.get("enabled", False):
    kept_indices = get_kept_indices(
        WVL_PRS, wl_filter.get("excluded_windows", [])
    )
  else:
    kept_indices = None

  output_dir = Path(eval_cfg["output_dir"])
  output_dir.mkdir(parents=True, exist_ok=True)

  weights_rec = Path(eval_cfg["weights_path_reconstruction"])
  weights_unc = Path(eval_cfg["weights_path_uncertainty"])

  n_msi = data_cfg["n_msi"]
  n_hsi = (
      len(kept_indices) if kept_indices is not None else data_cfg.get("n_hsi", 195)
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model_reconstruction = build_model_reconstruction(
      cfg, n_msi=n_msi, n_hsi=n_hsi
  ).to(device)
  model_uncertainty = build_model_uncertainty(
      cfg, n_msi=n_msi, out_channels=n_hsi
  ).to(device)

  if weights_rec.exists():
    state_dict_rec = torch.load(weights_rec, map_location=device)
    model_reconstruction.load_state_dict(
        state_dict_rec.get("model_state_dict", state_dict_rec)
    )

  if weights_unc.exists():
    state_dict_unc = torch.load(weights_unc, map_location=device)
    model_uncertainty.load_state_dict(
        state_dict_unc.get("model_state_dict", state_dict_unc)
    )

  scene_dir = root_dir / "data" / "mumucd" / args.scene
  if not scene_dir.exists():
    raise FileNotFoundError(
        f"❌ Le dossier de la scène n'existe pas : {scene_dir.resolve()}"
    )

  patch_size = 256
  patches_t1, patches_t2, gt_change_map_global, (h_crop, w_crop) = (
      process_nc_file(
          scene_dir=scene_dir,
          model_rec=model_reconstruction,
          model_unc=model_uncertainty,
          SRF_MATRIX=SRF_MATRIX,
          INTERP_MATRIX=INTERP_MATRIX,
          device=device,
          kept_indices=kept_indices,
          patch_size=patch_size,
          prefix=args.scene.capitalize(),
          is_normalized=is_normalized,
      )
  )

  # --- Reconstitution globale ---
  c_hsi = next(iter(patches_t2.values()))["cube_predict"].shape[-1]
  c_msi = next(iter(patches_t2.values()))["cube_msi"].shape[-1]

  pred_global_t2 = np.zeros((h_crop, w_crop, c_hsi), dtype=np.float32)
  unc_global_t2 = np.zeros((h_crop, w_crop, c_hsi), dtype=np.float32)
  gt1_global = np.zeros((h_crop, w_crop, c_hsi), dtype=np.float32)
  msi_global_t2 = np.zeros((h_crop, w_crop, c_msi), dtype=np.float32)

  pred_global_t1 = np.zeros((h_crop, w_crop, c_hsi), dtype=np.float32)
  unc_global_t1 = np.zeros((h_crop, w_crop, c_hsi), dtype=np.float32)

  num_px = w_crop // patch_size

  for patch_id, data_t2 in patches_t2.items():
    idx = int(patch_id.split("_")[-1])
    i, j = idx // num_px, idx % num_px

    y1, y2 = i * patch_size, (i + 1) * patch_size
    x1, x2 = j * patch_size, (j + 1) * patch_size

    pred_global_t2[y1:y2, x1:x2, :] = data_t2["cube_predict"]
    unc_global_t2[y1:y2, x1:x2, :] = data_t2["cube_uncertainty"]
    msi_global_t2[y1:y2, x1:x2, :] = data_t2["cube_msi"]

    gt1_global[y1:y2, x1:x2, :] = patches_t1[patch_id]["cube_gt"]
    pred_global_t1[y1:y2, x1:x2, :] = patches_t1[patch_id]["cube_predict"]
    unc_global_t1[y1:y2, x1:x2, :] = patches_t1[patch_id]["cube_uncertainty"]

  data_t1_global = {
      "cube_gt": gt1_global,
      "cube_predict": pred_global_t1,
      "cube_uncertainty": unc_global_t1,
      "model name": f"{args.scene.capitalize()} — T1 (Global Rec)",
  }
  data_t2_global = {
      "cube_predict": pred_global_t2,
      "cube_uncertainty": unc_global_t2,
      "cube_msi": msi_global_t2,
      "model name": f"{args.scene.capitalize()} — T2 (Global)",
  }

  # --- Nom du fichier incluant dynamiquement la valeur du NFA ---
  nfa_global_path = (
      output_dir / f"{args.scene}_nfa_global_map_{nfa_target_val}.png"
  )

  global_nfa_res = visualise_changement_complet_nfa(
      data_t1=data_t1_global,
      data_t2=data_t2_global,
      save_path=nfa_global_path,
      gt_change_map=gt_change_map_global,
      metric_type="mae",
      nfa_target=nfa_target_val,  # Utilisation de la variable du parser ici
      WVL_PRS=WVL_PRS,
      kept_indices=kept_indices,
      calib_t1=data_t1_global,
  )

  # --- Calcul et affichage du pourcentage de changement détecté ---
  change_raw = global_nfa_res["change_map_raw"]
  change_unc = global_nfa_res["change_map_unc"]

  total_pixels = change_raw.size
  pct_raw = (np.sum(change_raw) / total_pixels) * 100.0
  pct_unc = (np.sum(change_unc) / total_pixels) * 100.0

  print("\n📊 Statistiques de détection de changement (Globales) :")
  print(
      f"   - Pourcentage brut détecté     : {pct_raw:.2f}%"
      f" ({np.sum(change_raw)} pixels)"
  )
  print(
      f"   - Pourcentage filtré (Incert.) : {pct_unc:.2f}%"
      f" ({np.sum(change_unc)} pixels)"
  )
  if gt_change_map_global is not None:
    pct_gt = (np.sum(gt_change_map_global) / total_pixels) * 100.0
    print(
        f"   - Pourcentage Ground Truth     : {pct_gt:.2f}%"
        f" ({np.sum(gt_change_map_global)} pixels)"
    )

  print(
      f"\n🎉 Traitement global terminé avec succès ! Carte sauvegardée dans :"
      f" {nfa_global_path}"
  )