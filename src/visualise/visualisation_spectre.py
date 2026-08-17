import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from constants import WVL_PRS

def visualise_curve(mse_path, var_path, name_curve):
    mse_per_channel = np.load(mse_path)
    var_per_channel = np.load(var_path)
 
    plt.figure(figsize=(10, 5))
    plt.axvspan(1350, 1500, color='red', alpha=0.3, label="Atmospheric Absorption")
    plt.axvspan(1800, 2000, color='red', alpha=0.3)
 
    plt.plot(WVL_PRS, mse_per_channel, color="royalblue", linewidth=2, label="Mean of MSE SAM : rad")
    plt.plot(WVL_PRS, mse_per_channel + var_per_channel, color="royalblue", linestyle="--", alpha=0.5)
 
    plt.title("Reconstruction error by each channel")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("MSE")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(name_curve, dpi=300, bbox_inches="tight")  # Correction de l'erreur de saisie
    plt.close()
    print(f"Graphique d'erreur sauvegardé : {name_curve}")
 
def supervise_analyse_spectrale(data, save_name, plot_dir, wvl_prs=WVL_PRS):
    """
    Combine les analyses spectrales avancées sans explosion mémoire :
    1. Enveloppe des Quantiles (5%-95%) globale (GT vs Prédit).
    2. Profil d'erreur RMSE global par longueur d'onde.
    3. Distribution des pixels à zéro (données manquantes).
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    cube_gt = data["cube_gt"]          # [H, W, C]
    cube_pred = data["cube_predict"]   # [H, W, C]
    
    wvl_prs = np.array(wvl_prs)
    
    # -------------------------------------------------------------
    # CALCULS GLOBAUX (QUANTILE & RMSE)
    # -------------------------------------------------------------
    mean_gt = np.mean(cube_gt, axis=(0, 1))
    q05_gt = np.percentile(cube_gt, 5, axis=(0, 1))
    q95_gt = np.percentile(cube_gt, 95, axis=(0, 1))
    
    mean_pred = np.mean(cube_pred, axis=(0, 1))
    
    # Calcul des profils d'erreur spectraux
    rmse = np.sqrt(np.mean((cube_gt - cube_pred)**2, axis=(0, 1)))
    zeros_per_wvl = np.sum(cube_gt == 0, axis=(0, 1))
    
    # -------------------------------------------------------------
    # TRACÉS GRAPHIQUES
    # -------------------------------------------------------------
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    fig1.suptitle(f"Diagnostic Global & Dispersion Spectrale — Scène : {save_name}", fontsize=14, fontweight='bold')
    
    # Subplot 1 : Enveloppes 5%-95% (GT vs Modèle)
    ax1.plot(wvl_prs, mean_gt, color="black", linewidth=2, label="GT Moyenne")
    ax1.fill_between(wvl_prs, q05_gt, q95_gt, color="black", alpha=0.1, label="GT (Quantiles 5%-95%)")
    ax1.plot(wvl_prs, mean_pred, color="crimson", linewidth=1.5, linestyle="--", label="Prédit Moyenne")
   
    ax1.set_ylabel("Réflectance")
    ax1.set_title("Comparaison des Enveloppes de Distribution Spectrale")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # Subplot 2 : RMSE par longueur d'onde
    ax2.plot(wvl_prs, rmse, color="crimson", linewidth=2, label="RMSE par Canal")
    ax2.set_ylabel("RMSE Spatiale")
    ax2.set_title("Profil de l'Erreur RMSE par Longueur d'Onde")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    # Subplot 3 : Données manquantes
    ax3.plot(wvl_prs, zeros_per_wvl, color="darkorange", linewidth=1.5, label="Pixels à Zéro")
    ax3.fill_between(wvl_prs, 0, zeros_per_wvl, color="darkorange", alpha=0.1)
    ax3.set_ylabel("Nb Zéros")
    ax3.set_xlabel("Longueur d'onde (nm)")
    ax3.set_title("Distribution des Données Manquantes")
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    output_path = plot_dir / f"Analyse_Spectrale_Globale_{save_name}.png"
    fig1.savefig(output_path, dpi=200)
    plt.close(fig1)
    print(f"[Succès] Analyse spectrale globale sauvegardée : {output_path}")

def visualise_random_pixels_spectra_with_diff(data, patch_idx_name, plot_dir, num_pixels=30, wvl_prs=WVL_PRS):
    """
    Trace un 'spaghetti plot' comparant les spectres et affiche les différences spectrales en dessous.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    cube_gt = data["cube_gt"]          # [H, W, C]
    cube_pred = data["cube_predict"]   # [H, W, C]
    H, W, C = cube_gt.shape
    wvl_prs = np.array(wvl_prs)
    
    # Sélection de pixels aléatoires (fixes via seed)
    np.random.seed(42)
    rand_h = np.random.randint(0, H, num_pixels)
    rand_w = np.random.randint(0, W, num_pixels)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    model_name = data.get("model_name", "Modèle")
    fig.suptitle(f"Variabilité & Erreur Spectrale Locale ({num_pixels} pixels) — Patch : {patch_idx_name}", fontsize=13, fontweight='bold')
    
    # Zones d'absorption atmosphérique en fond sur les deux axes
    for ax in (ax1, ax2):
        ax.axvspan(1350, 1500, color='gray', alpha=0.15)
        ax.axvspan(1800, 2000, color='gray', alpha=0.15)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Tableaux pour stocker les erreurs et calculer la moyenne globale
    all_diffs = []

    # Boucle sur les pixels aléatoires
    for i in range(num_pixels):
        h, w = rand_h[i], rand_w[i]
        spec_gt = cube_gt[h, w, :]
        spec_pred = cube_pred[h, w, :]
        diff = spec_pred - spec_gt  # ou np.abs(...) selon ta préférence
        all_diffs.append(diff)
        
        label_gt = "Pixels GT" if i == 0 else ""
        label_pred = "Pixels Prédits" if i == 0 else ""
        label_diff = "Différence par pixel (Pred - GT)" if i == 0 else ""
        
        # Subplot 1 : Spectres bruts
        ax1.plot(wvl_prs, spec_gt, color="black", linewidth=0.8, alpha=0.2, label=label_gt)
        ax1.plot(wvl_prs, spec_pred, color="crimson", linewidth=0.8, alpha=0.2, label=label_pred)
        
        # Subplot 2 : Différences individuelles
        ax2.plot(wvl_prs, diff, color="purple", linewidth=0.8, alpha=0.25, label=label_diff)

    all_diffs = np.array(all_diffs)
    mean_diff = np.mean(all_diffs, axis=0)
    
    # Moyennes sur le premier subplot
    mean_gt = np.mean(cube_gt, axis=(0, 1))
    mean_pred = np.mean(cube_pred, axis=(0, 1))
    ax1.plot(wvl_prs, mean_gt, color="black", linewidth=2, label="Moyenne GT")
    ax1.plot(wvl_prs, mean_pred, color="darkorange", linewidth=2, linestyle="--", label=f"Moyenne {model_name}")
    
    # Erreur moyenne sur le second subplot
    ax2.plot(wvl_prs, mean_diff, color="darkred", linewidth=2, linestyle="-", label="Erreur Moyenne (Bias)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Habillage Subplot 1
    ax1.set_ylabel("Réflectance", fontsize=10)
    ax1.set_title("Signatures Spectrales (Traits fins = pixels aléatoires, Traits pleins = moyennes)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    
    # Habillage Subplot 2
    ax2.set_xlabel("Longueur d'onde (nm)", fontsize=10)
    ax2.set_ylabel("Différence (Pred - GT)", fontsize=10)
    ax2.set_title("Écart Spectral Résiduel par Pixel", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    
    output_path = plot_dir / f"Spaghetti_Diff_Patch_{patch_idx_name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Succès] Spaghetti plot avec différences sauvegardé : {output_path}")

def visualise_random_pixels_spectra_with_diff(data, patch_idx_name, plot_dir, num_pixels=30, wvl_prs=WVL_PRS):
    """
    Trace un 'spaghetti plot' comparant les spectres et affiche les différences spectrales en dessous.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    cube_gt = data["cube_gt"]          # [H, W, C]
    cube_pred = data["cube_predict"]   # [H, W, C]
    H, W, C = cube_gt.shape
    wvl_prs = np.array(wvl_prs)
    
    # Sélection de pixels aléatoires (fixes via seed)
    np.random.seed(42)
    rand_h = np.random.randint(0, H, num_pixels)
    rand_w = np.random.randint(0, W, num_pixels)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    model_name = data.get("model_name", "Modèle")
    fig.suptitle(f"Variabilité & Erreur Spectrale Locale ({num_pixels} pixels) — Patch : {patch_idx_name}", fontsize=13, fontweight='bold')
    
    # Zones d'absorption atmosphérique en fond sur les deux axes
    for ax in (ax1, ax2):
        ax.axvspan(1350, 1500, color='gray', alpha=0.15)
        ax.axvspan(1800, 2000, color='gray', alpha=0.15)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Tableaux pour stocker les erreurs et calculer la moyenne globale
    all_diffs = []

    # Boucle sur les pixels aléatoires
    for i in range(num_pixels):
        h, w = rand_h[i], rand_w[i]
        spec_gt = cube_gt[h, w, :]
        spec_pred = cube_pred[h, w, :]
        diff = spec_pred - spec_gt  # ou np.abs(...) selon ta préférence
        all_diffs.append(diff)
        
        label_gt = "Pixels GT" if i == 0 else ""
        label_pred = "Pixels Prédits" if i == 0 else ""
        label_diff = "Différence par pixel (Pred - GT)" if i == 0 else ""
        
        # Subplot 1 : Spectres bruts
        ax1.plot(wvl_prs, spec_gt, color="black", linewidth=0.8, alpha=0.2, label=label_gt)
        ax1.plot(wvl_prs, spec_pred, color="crimson", linewidth=0.8, alpha=0.2, label=label_pred)
        
        # Subplot 2 : Différences individuelles
        ax2.plot(wvl_prs, diff, color="purple", linewidth=0.8, alpha=0.25, label=label_diff)

    all_diffs = np.array(all_diffs)
    mean_diff = np.mean(all_diffs, axis=0)
    
    # Moyennes sur le premier subplot
    mean_gt = np.mean(cube_gt, axis=(0, 1))
    mean_pred = np.mean(cube_pred, axis=(0, 1))
    ax1.plot(wvl_prs, mean_gt, color="black", linewidth=2, label="Moyenne GT")
    ax1.plot(wvl_prs, mean_pred, color="darkorange", linewidth=2, linestyle="--", label=f"Moyenne {model_name}")
    
    # Erreur moyenne sur le second subplot
    ax2.plot(wvl_prs, mean_diff, color="darkred", linewidth=2, linestyle="-", label="Erreur Moyenne (Bias)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Habillage Subplot 1
    ax1.set_ylabel("Réflectance", fontsize=10)
    ax1.set_title("Signatures Spectrales (Traits fins = pixels aléatoires, Traits pleins = moyennes)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    
    # Habillage Subplot 2
    ax2.set_xlabel("Longueur d'onde (nm)", fontsize=10)
    ax2.set_ylabel("Différence (Pred - GT)", fontsize=10)
    ax2.set_title("Écart Spectral Résiduel par Pixel", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    
    output_path = plot_dir / f"Spaghetti_Diff_Patch_{patch_idx_name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Succès]  plot avec différences sauvegardé : {output_path}")