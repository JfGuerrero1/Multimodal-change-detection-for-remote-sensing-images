import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from src.constants import WVL_PRS

def get_rgb_from_patch(patch,is_prisma):

    patch=np.nan_to_num(patch, nan=1000.0)
    if is_prisma:
        indices=[32,15,5]
    else:
        indices=[3,2,0]
    
    r, g, b = patch[indices[0]], patch[indices[1]], patch[indices[2]]
    
    if not is_prisma: 
        g = (patch[1] + patch[2]) / 2
        
    rgb = np.stack([r, g, b], axis=-1)


    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip(rgb, p2, p98)
    rgb = (rgb - p2) / (p98 - p2 + 1e-8)

    return rgb

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualise_curve(mean_error, std_error, max_error, name_curve, kept_indices=None, wvl_prs=WVL_PRS):
    """
    Trace le profil d'erreur global (Moyenne, ± Écart-type et Max) par canal spectral.
    """
    name_curve = Path(name_curve)
    name_curve.parent.mkdir(parents=True, exist_ok=True)
    
    C = len(mean_error)
    wvl_filtered = _get_filtered_wvl(wvl_prs, kept_indices, C)
 
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    # Zones d'absorption atmosphérique
    ax.axvspan(1350, 1500, color='red', alpha=0.15, label="Absorption Atmosphérique")
    ax.axvspan(1800, 2000, color='red', alpha=0.15)
 
    # Courbes principales

    ax.plot(wvl_filtered, mean_error, color="royalblue", linewidth=2, label="Erreur Moyenne (MAE")
    
    # Enveloppe écart-type
    lower_bound = np.maximum(0.0, mean_error - std_error)
    upper_bound = mean_error + std_error
    ax.fill_between(
        wvl_filtered, 
        lower_bound, 
        upper_bound, 
        color="royalblue", 
        alpha=0.2, 
        label=r"$\pm$ 1 Écart-type"
    )
 
    ax.set_title("Erreur de reconstruction par canal spectral", fontsize=12, fontweight='bold')
    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_ylabel("Erreur")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    
    fig.savefig(name_curve, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Succès] Graphique d'erreur sauvegardé : {name_curve}")
 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.constants import WVL_PRS

def _get_filtered_wvl(wvl_prs, kept_indices, target_len):
    """Fonction utilitaire interne pour filtrer automatiquement WVL_PRS."""
    if wvl_prs is None:
        return np.arange(target_len)
    
    wvl_arr = np.array(wvl_prs)
    if kept_indices is not None:
        kept_arr = np.array(kept_indices)
        try:
            if kept_arr.dtype == bool or kept_arr.dtype == np.bool_:
                wvl_arr = wvl_arr[kept_arr]
            else:
                wvl_arr = wvl_arr[kept_arr]
        except Exception:
            pass
            
    # Sécurité finale si la taille ne correspond toujours pas au cube
    if len(wvl_arr) != target_len:
        wvl_arr = np.arange(target_len)
    return wvl_arr


def supervise_analyse_spectrale(data, save_name, plot_dir, wvl_prs=WVL_PRS, kept_indices=None):
    """
    Trace le profil d'erreur global (Moyenne, Écart-type et Maximum) par longueur d'onde.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    cube_gt = data["cube_gt"]          # [H, W, C]
    cube_pred = data["cube_predict"]   # [H, W, C]
    C = cube_gt.shape[-1]
    
    wvl_prs = _get_filtered_wvl(wvl_prs, kept_indices, C)
    
    # -------------------------------------------------------------
    # CALCULS DES ERREURS (Moyenne, Std, Max par canal)
    # -------------------------------- Erreur absolue par pixel [H, W, C]
    abs_error = np.abs(cube_gt - cube_pred)
    
    mean_error = np.mean(abs_error, axis=(0, 1))
    std_error = np.std(abs_error, axis=(0, 1))
    max_error = np.max(abs_error, axis=(0, 1))
    
    # -------------------------------------------------------------
    # TRACÉ GRAPHIQUE UNIQUE OU MULTI-SUBPLOTS D'ERREUR
    # -------------------------------------------------------------
    fig1, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    fig1.suptitle(f"Profil d'Erreur Spectrale (MAE, Std, Max) — Scène : {save_name}", fontsize=14, fontweight='bold')
    
    # Zones d'absorption atmosphérique en fond (optionnel mais utile)
    ax.axvspan(1350, 1500, color='red', alpha=0.08, label="Absorption Atmosphérique")
    ax.axvspan(1800, 2000, color='red', alpha=0.08)

    # Tracé du Maximum d'erreur (en arrière-plan, plus transparent)
    ax.plot(wvl_prs, max_error, color="crimson", linestyle=":", linewidth=1.5, label="Erreur Max")
    
    # Tracé de la Moyenne (MAE)
    ax.plot(wvl_prs, mean_error, color="darkblue", linewidth=2, label="Erreur Moyenne (MAE)")
    
    # Enveloppe de l'écart-type (± std)
    ax.fill_between(
        wvl_prs, 
        np.maximum(0, mean_error - std_error), 
        mean_error + std_error, 
        color="blue", 
        alpha=0.2, 
        label="$\pm$ 1 Écart-type"
    )
    
    ax.set_ylabel("Erreur absolue de Réflectance")
    ax.set_xlabel("Longueur d'onde (nm)")
    ax.set_title("Évolution de l'erreur spatiale par canal spectral")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    
    output_path = plot_dir / f"Analyse_Spectrale_Globale_{save_name}.png"
    fig1.savefig(output_path, dpi=200)
    plt.close(fig1)
    print(f"[Succès] Analyse spectrale globale d'erreur sauvegardée : {output_path}")


def visualise_random_pixels_spectra_with_diff(data, patch_idx_name, plot_dir, num_pixels=30, wvl_prs=WVL_PRS, kept_indices=None):
    """
    Trace un 'spaghetti plot' avec filtrage automatique via kept_indices.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    cube_gt = data["cube_gt"]          # [H, W, C]
    cube_pred = data["cube_predict"]   # [H, W, C]
    H, W, C = cube_gt.shape
    
    wvl_prs = _get_filtered_wvl(wvl_prs, kept_indices, C)
    
    np.random.seed(42)
    rand_h = np.random.randint(0, H, num_pixels)
    rand_w = np.random.randint(0, W, num_pixels)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    model_name = data.get("model_name", "Modèle")
    fig.suptitle(f"Variabilité & Erreur Spectrale Locale ({num_pixels} pixels) — Patch : {patch_idx_name}", fontsize=13, fontweight='bold')
    
    for ax in (ax1, ax2):
        ax.axvspan(1350, 1500, color='gray', alpha=0.15)
        ax.axvspan(1800, 2000, color='gray', alpha=0.15)
        ax.grid(True, linestyle=":", alpha=0.6)

    all_diffs = []
    for i in range(num_pixels):
        h, w = rand_h[i], rand_w[i]
        spec_gt = cube_gt[h, w, :]
        spec_pred = cube_pred[h, w, :]
        diff = spec_pred - spec_gt
        all_diffs.append(diff)
        
        label_gt = "Pixels GT" if i == 0 else ""
        label_pred = "Pixels Prédits" if i == 0 else ""
        label_diff = "Différence par pixel (Pred - GT)" if i == 0 else ""
        
        ax1.plot(wvl_prs, spec_gt, color="black", linewidth=0.8, alpha=0.2, label=label_gt)
        ax1.plot(wvl_prs, spec_pred, color="crimson", linewidth=0.8, alpha=0.2, label=label_pred)
        ax2.plot(wvl_prs, diff, color="purple", linewidth=0.8, alpha=0.25, label=label_diff)

    all_diffs = np.array(all_diffs)
    mean_diff = np.mean(all_diffs, axis=0)
    mean_gt = np.mean(cube_gt, axis=(0, 1))
    mean_pred = np.mean(cube_pred, axis=(0, 1))
    
    ax1.plot(wvl_prs, mean_gt, color="black", linewidth=2, label="Moyenne GT")
    ax1.plot(wvl_prs, mean_pred, color="darkorange", linewidth=2, linestyle="--", label=f"Moyenne {model_name}")
    ax2.plot(wvl_prs, mean_diff, color="darkred", linewidth=2, linestyle="-", label="Erreur Moyenne (Bias)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    ax1.set_ylabel("Réflectance", fontsize=10)
    ax1.set_title("Signatures Spectrales", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    
    ax2.set_xlabel("Longueur d'onde (nm)", fontsize=10)
    ax2.set_ylabel("Différence (Pred - GT)", fontsize=10)
    ax2.set_title("Écart Spectral Résiduel par Pixel", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    
    output_path = plot_dir / f"Spaghetti_Diff_Patch_{patch_idx_name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Succès] Spaghetti plot sauvegardé : {output_path}")


def trace_spectre(patch, patch_gt, y_indices, x_indices, scene="scene_inconnue", output_dir_diag=None, wvl_prs=WVL_PRS, kept_indices=None, log_wandb=False, wandb_name="spectral_profiles"):
    """
    Trace une figure complète avec filtrage automatique via kept_indices.
    """
    if output_dir_diag is not None:
        output_dir_diag = Path(output_dir_diag)
        output_dir_diag.mkdir(parents=True, exist_ok=True)

    H, W, C = patch.shape
    num_pixels = len(y_indices)
    
    if num_pixels == 0:
        print("Aucun point à tracer (listes vides).")
        return

    bandes = _get_filtered_wvl(wvl_prs, kept_indices, C)

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.1, 1.1], hspace=0.35, wspace=0.25)
    
    ax_rgb_pred = fig.add_subplot(gs[0, 0])
    ax_rgb_gt = fig.add_subplot(gs[0, 1])
    ax_spec_pred = fig.add_subplot(gs[1, 0])
    ax_spec_gt = fig.add_subplot(gs[1, 1])
    ax_diff_indiv = fig.add_subplot(gs[2, 0])
    ax_diff_mean = fig.add_subplot(gs[2, 1])
    
    try:
        im_rgb_pred = get_rgb_from_patch(patch.transpose(2, 0, 1), is_prisma=True)
        im_rgb_gt = get_rgb_from_patch(patch_gt.transpose(2, 0, 1), is_prisma=True)
    except NameError:
        im_rgb_pred = np.clip(patch[:, :, [3, 2, 1]] / (np.percentile(patch[:, :, [3, 2, 1]], 98) + 1e-8), 0, 1)
        im_rgb_gt = np.clip(patch_gt[:, :, [3, 2, 1]] / (np.percentile(patch_gt[:, :, [3, 2, 1]], 98) + 1e-8), 0, 1)

    ax_rgb_pred.imshow(im_rgb_pred)
    ax_rgb_pred.set_title("RGB Prédit", fontsize=11, fontweight='bold')
    ax_rgb_pred.axis('on')

    ax_rgb_gt.imshow(im_rgb_gt)
    ax_rgb_gt.set_title("RGB Ground Truth", fontsize=11, fontweight='bold')
    ax_rgb_gt.axis('on')

    for ax in [ax_spec_pred, ax_spec_gt, ax_diff_indiv, ax_diff_mean]:
        ax.axvspan(1350, 1500, color='red', alpha=0.12, label="Absorption Atmosphérique")
        ax.axvspan(1800, 2000, color='red', alpha=0.12)
        ax.set_xlabel("Longueur d'onde (nm)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

    ax_spec_pred.set_title("Spectres — Modèle Prédit", fontsize=10, fontweight='bold')
    ax_spec_gt.set_title("Spectres — Ground Truth", fontsize=10, fontweight='bold')
    ax_diff_indiv.set_title("Différences (Pred - GT) par pixel", fontsize=10, fontweight='bold')
    ax_diff_mean.set_title("Erreur Moyenne $\pm$ Std sur les pixels", fontsize=10, fontweight='bold')

    for ax in [ax_spec_pred, ax_spec_gt]:
        ax.set_ylabel("Réflectance", fontsize=9)
    for ax in [ax_diff_indiv, ax_diff_mean]:
        ax.set_ylabel("Écart (Pred - GT)", fontsize=9)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

    colors = plt.cm.rainbow(np.linspace(0, 1, num_pixels))
    all_diffs = []

    for i in range(num_pixels):
        y, x = y_indices[i], x_indices[i]
        if not (0 <= y < H and 0 <= x < W):
            continue
            
        spectre_pred = patch[y, x, :].astype(float)
        spectre_gt = patch_gt[y, x, :].astype(float)
        
        spectre_pred[spectre_pred == 0] = np.nan
        spectre_gt[spectre_gt == 0] = np.nan
        
        diff = spectre_pred - spectre_gt
        all_diffs.append(diff)
        
        color = colors[i]
        ax_rgb_pred.scatter(x, y, color=color, edgecolors='black', s=80, marker='o')
        ax_rgb_gt.scatter(x, y, color=color, edgecolors='black', s=80, marker='o')
        
        ax_spec_pred.plot(bandes, spectre_pred, color=color, linestyle='-', linewidth=1.2, label=f"P{i+1} ({y},{x})")
        ax_spec_gt.plot(bandes, spectre_gt, color=color, linestyle='-', linewidth=1.2, label=f"P{i+1} ({y},{x})")
        ax_diff_indiv.plot(bandes, diff, color=color, linestyle='-', linewidth=1.0, alpha=0.6, label=f"P{i+1}" if i == 0 else "")

    if len(all_diffs) > 0:
        all_diffs_arr = np.array(all_diffs)
        mean_diff = np.nanmean(all_diffs_arr, axis=0)
        std_diff = np.nanstd(all_diffs_arr, axis=0)
        
        ax_diff_mean.plot(bandes, mean_diff, color='darkblue', linewidth=2, label="Erreur Moyenne (Bias)")
        ax_diff_mean.fill_between(bandes, mean_diff - std_diff, mean_diff + std_diff, color='blue', alpha=0.2, label="$\pm$ 1 Std")

    ax_spec_pred.legend(loc='upper right', fontsize=7)
    ax_spec_gt.legend(loc='upper right', fontsize=7)
    ax_diff_indiv.legend(loc='upper right', fontsize=7)
    ax_diff_mean.legend(loc='upper right', fontsize=8)

    if output_dir_diag is not None:
        output_plot = output_dir_diag / f"spectre_{scene}.png"
        fig.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"[Succès] Graphique sauvegardé localement : {output_plot}")

    if log_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({wandb_name: wandb.Image(fig)})
        except ImportError:
            pass
    
    plt.close()