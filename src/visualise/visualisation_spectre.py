import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def supervise_analyse_spectrale(data, save_name, plot_dir, wvl_prs):
    """
    Combine les analyses spectrales avancées sans explosion mémoire :
    1. Enveloppe des Quantiles (5%-95%) globale.
    2. Profil d'erreur RMSE global par longueur d'onde.
    3. Extraction et affichage automatique des signatures par classe (Ville, Eau, Forêt, Désert/Sol nu).
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    cube_gt = data["cube_gt"]          # [H, W, C]
    cube_pred = data["cube_predict"]   # [H, W, C]
    
    H, W, C = cube_gt.shape
    wvl_prs = np.array(wvl_prs)
    
    # -------------------------------------------------------------
    # PARTIE 1 : CALCULS GLOBAUX (QUANTILE & RMSE)
    # -------------------------------------------------------------
    mean_gt = np.mean(cube_gt, axis=(0, 1))
    q05_gt = np.percentile(cube_gt, 5, axis=(0, 1))
    q95_gt = np.percentile(cube_gt, 95, axis=(0, 1))
    
    
    # Calcul des profils d'erreur spectraux
    rmse= np.sqrt(np.mean((cube_gt - cube_pred)**2, axis=(0, 1)))

    zeros_per_wvl = np.sum(cube_gt == 0, axis=(0, 1))
    
    # -------------------------------------------------------------
    # PARTIE 3 : DEBUT DES TRACÉS GRAPHIQUES
    # -------------------------------------------------------------
    # Figure 1 : Enveloppes globales et Profils d'erreur
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True, constrained_layout=True)
    fig1.suptitle(f"Diagnostic Global & Dispersion Spectrale — Scène : {save_name}", fontsize=14, fontweight='bold')
    
    # Subplot 1 : Enveloppes 5%-95% (Fidélité de la distribution)
    ax1.plot(wvl_prs, mean_gt, color="black", linewidth=2, label="GT Moyenne")
    ax1.fill_between(wvl_prs, q05_gt, q95_gt, color="black", alpha=0.1, label="GT (Quantiles 5%-95%)")
   
    ax1.set_ylabel("Réflectance")
    ax1.set_title("Comparaison des Enveloppes de Distribution Spectrale")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # Subplot 2 : RMSE par longueur d'onde
    ax2.plot(wvl_prs, rmse, color="crimson", linewidth=2, label="Sans MLP (Baseline)")
   
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
    
    fig1.savefig(plot_dir / f"Analyse_Spectrale_Globale_{save_name}.png", dpi=200)
    plt.close(fig1)
