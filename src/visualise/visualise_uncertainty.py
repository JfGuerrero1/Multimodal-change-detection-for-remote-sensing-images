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
from src.old.metrics_and_loss.metrics import (
    compute_sam_map, compute_mae, compute_ergas, 
    compute_mrae, compute_ssim_multiband, compute_mse, 
    compute_sam, compute_psnr, compute_rmse
)

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

