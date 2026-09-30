import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon, laplace
from tqdm import tqdm

def plot_error_uncertainty_histograms_and_fit(output_dir, max_samples_per_patch=500):
    output_path = Path(output_dir)
    
    metadata_file = output_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"❌ Le fichier 'metadata.json' est introuvable dans : {output_path}.")
    
    # 1. Chargement des métadonnées
    with open(metadata_file, "r") as f:
        meta = json.load(f)
    
    with open(output_path / "metadata_uncertainty.json", "r") as f:
        metadata_unc = json.load(f)
        
    total_patches = meta["total_patches"]
    shape_hsi = tuple(meta["hsi_shape"])
    shape_rec = (total_patches, shape_hsi[1]) + shape_hsi[2:]
    shape_unc = tuple(metadata_unc["shape"])
    
    # 2. Ouverture des memmaps
    print(f"📂 Ouverture des memmaps dans {output_path}...")
    fp_hsi = np.memmap(output_path / "hsi.bin", dtype="float32", mode="r", shape=shape_hsi)
    fp_rec = np.memmap(output_path / "reconstruction.bin", dtype="float32", mode="r", shape=shape_rec)
    fp_unc = np.memmap(output_path / "uncertainty.bin", dtype="float32", mode="r", shape=shape_unc)
    
    all_abs_errors = []
    all_signed_errors = []
    all_uncertainties = []
    
    # 3. Calcul des erreurs normalisées (absolues et signées) avec sous-échantillonnage
    print("📊 Calcul des valeurs (erreurs absolues, signées et incertitudes)...")
    for idx in tqdm(range(total_patches), desc="Parcours des patches"):
        hsi = fp_hsi[idx]
        rec = fp_rec[idx]
        unc = fp_unc[idx]
        
        unc_mean = np.mean(unc, axis=0)
        sigma_laplace = np.sqrt(2) * unc_mean
        
        diff = (hsi - rec)
        abs_diff = np.abs(diff)
        
        # Erreur absolue normalisée (pour la loi exponentielle)
        norm_abs_error = np.mean(abs_diff / (np.expand_dims(sigma_laplace, 0) + 1e-8), axis=0)
        # Erreur signée normalisée (pour la loi de Laplace)
        norm_signed_error = np.mean(diff / (np.expand_dims(sigma_laplace, 0) + 1e-8), axis=0)
        
        abs_flat = norm_abs_error.flatten()
        signed_flat = norm_signed_error.flatten()
        unc_flat = unc_mean.flatten()
        
        # Sous-échantillonnage aléatoire pour optimiser la mémoire et l'affichage
        if len(abs_flat) > max_samples_per_patch:
            indices = np.random.choice(len(abs_flat), max_samples_per_patch, replace=False)
            all_abs_errors.extend(abs_flat[indices])
            all_signed_errors.extend(signed_flat[indices])
            all_uncertainties.extend(unc_flat[indices])
        else:
            all_abs_errors.extend(abs_flat)
            all_signed_errors.extend(signed_flat)
            all_uncertainties.extend(unc_flat)
        
    all_abs_errors = np.array(all_abs_errors, dtype=np.float32)
    all_signed_errors = np.array(all_signed_errors, dtype=np.float32)
    all_uncertainties = np.array(all_uncertainties, dtype=np.float32)
    
    print(f"✨ Nombre total de points conservés : {len(all_abs_errors):,}")

    # 4. Calcul des seuils et fits statistiques
    tau_98_abs = np.quantile(all_abs_errors, 0.98)
    tau_lower_signed = np.quantile(all_signed_errors, 0.02)
    tau_upper_signed = np.quantile(all_signed_errors, 0.98)
    gamma_50 = np.quantile(all_uncertainties, 0.5)
    
    # Fit Exponentiel sur l'erreur absolue normalisée
    exp_loc, exp_scale = expon.fit(all_abs_errors, floc=0)
    print(f"📈 Fit Exponentiel (Erreur Absolue) - Échelle b : {exp_scale:.4f}")

    # Fit Laplace sur l'erreur signée normalisée
    laplace_loc, laplace_scale = laplace.fit(all_signed_errors)
    print(f"📈 Fit Laplace (Erreur Signée) - Centre (loc) : {laplace_loc:.4f}, Échelle (scale) : {laplace_scale:.4f}")

    # 5. Tracé des graphiques (Grille 3x2)
    print("📈 Génération des graphiques...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    # --- Ligne 0, Col 0 : Erreur Absolue Normalisée vs Loi Exponentielle ---
    axes[0, 0].hist(
        all_abs_errors, bins=100, alpha=0.6, edgecolor='black', color='steelblue', 
        density=True, label='Données empiriques (|HSI - Rec| / sqrt(2)*U)'
    )
    x_vals_exp = np.linspace(0, max(all_abs_errors), 500)
    y_theory_exp = expon.pdf(x_vals_exp, loc=exp_loc, scale=exp_scale)
    axes[0, 0].plot(x_vals_exp, y_theory_exp, 'r-', linewidth=2.5, label=f'Fit Exponentiel (b={exp_scale:.2f})')
    axes[0, 0].axvline(tau_98_abs, color='darkred', linestyle='--', linewidth=2, label=f'Seuil 98% = {tau_98_abs:.4f}')
    axes[0, 0].set_xlabel('Erreur absolue normalisée')
    axes[0, 0].set_ylabel('Densité de probabilité (log)')
    axes[0, 0].set_title('Erreur Absolue vs Loi Exponentielle')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # --- Ligne 0, Col 1 : CDF Erreur Absolue ---
    sorted_abs = np.sort(all_abs_errors)
    cdf_abs = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    cdf_theory_exp = expon.cdf(sorted_abs, loc=exp_loc, scale=exp_scale)
    axes[0, 1].plot(sorted_abs, cdf_abs, color='steelblue', lw=2, label='CDF Empirique')
    axes[0, 1].plot(sorted_abs, cdf_theory_exp, color='red', linestyle='--', lw=2, label='CDF Théorique (Exp)')
    axes[0, 1].axvline(tau_98_abs, color='darkred', linestyle='--', linewidth=2)
    axes[0, 1].axhline(0.98, color='darkred', linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel('Erreur absolue normalisée')
    axes[0, 1].set_ylabel('CDF')
    axes[0, 1].set_title('Fonction de Répartition - Erreur Absolue')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # --- Ligne 1, Col 0 : Erreur Signée Normalisée vs Loi de Laplace ---
    axes[1, 0].hist(
        all_signed_errors, bins=100, alpha=0.6, edgecolor='black', color='forestgreen', 
        density=True, label='Données empiriques ((HSI - Rec) / sqrt(2)*U)'
    )
    x_vals_lap = np.linspace(min(all_signed_errors), max(all_signed_errors), 500)
    y_theory_lap = laplace.pdf(x_vals_lap, loc=laplace_loc, scale=laplace_scale)
    axes[1, 0].plot(x_vals_lap, y_theory_lap, 'darkorange', linewidth=2.5, label=f'Fit Laplace (b={laplace_scale:.2f})')
    axes[1, 0].axvline(tau_lower_signed, color='purple', linestyle='--', linewidth=2, label=f'Seuil 2% = {tau_lower_signed:.4f}')
    axes[1, 0].axvline(tau_upper_signed, color='purple', linestyle='--', linewidth=2, label=f'Seuil 98% = {tau_upper_signed:.4f}')
    axes[1, 0].set_xlabel('Erreur signée normalisée')
    axes[1, 0].set_ylabel('Densité de probabilité (log)')
    axes[1, 0].set_title('Erreur Signée vs Loi de Laplace')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # --- Ligne 1, Col 1 : CDF Erreur Signée ---
    sorted_signed = np.sort(all_signed_errors)
    cdf_signed = np.arange(1, len(sorted_signed) + 1) / len(sorted_signed)
    cdf_theory_lap = laplace.cdf(sorted_signed, loc=laplace_loc, scale=laplace_scale)
    axes[1, 1].plot(sorted_signed, cdf_signed, color='forestgreen', lw=2, label='CDF Empirique')
    axes[1, 1].plot(sorted_signed, cdf_theory_lap, color='darkorange', linestyle='--', lw=2, label='CDF Théorique (Laplace)')
    axes[1, 1].axvline(tau_lower_signed, color='purple', linestyle='--', linewidth=2)
    axes[1, 1].axvline(tau_upper_signed, color='purple', linestyle='--', linewidth=2)
    axes[1, 1].axhline(0.02, color='purple', linestyle=':', alpha=0.7)
    axes[1, 1].axhline(0.98, color='purple', linestyle=':', alpha=0.7)
    axes[1, 1].set_xlabel('Erreur signée normalisée')
    axes[1, 1].set_ylabel('CDF')
    axes[1, 1].set_title('Fonction de Répartition - Erreur Signée')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # --- Ligne 2, Col 0 : Histogramme des Incertitudes U (Empirique pur) ---
    axes[2, 0].hist(all_uncertainties, bins=100, alpha=0.7, edgecolor='black', color='darkorange', label='Incertitudes empiriques U')
    axes[2, 0].axvline(gamma_50, color='darkblue', linestyle='--', linewidth=2, label=f'Médiane (50%) = {gamma_50:.4f}')
    axes[2, 0].set_xlabel('Incertitude U (paramètre b)')
    axes[2, 0].set_ylabel('Fréquence (log)')
    axes[2, 0].set_title('Histogramme - Incertitudes U')
    axes[2, 0].set_yscale('log')
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)
    
    # --- Ligne 2, Col 1 : Fonction de Répartition (CDF) Empirique des Incertitudes ---
    sorted_unc = np.sort(all_uncertainties)
    cdf_unc = np.arange(1, len(sorted_unc) + 1) / len(sorted_unc)
    axes[2, 1].plot(sorted_unc, cdf_unc, color='darkorange', lw=2, label='CDF Empirique')
    axes[2, 1].axvline(gamma_50, color='darkblue', linestyle='--', linewidth=2, label=f'Médiane (50%) = {gamma_50:.4f}')
    axes[2, 1].axhline(0.50, color='darkblue', linestyle=':', alpha=0.7)
    axes[2, 1].set_xlabel('Incertitude U')
    axes[2, 1].set_ylabel('CDF (Fonction de répartition)')
    axes[2, 1].set_title('Fonction de Répartition - Incertitudes')
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = output_path / "threshold_analysis_full_grid.png"
    plt.savefig(save_path, dpi=100)
    print(f"💾 Graphique complet sauvegardé dans : {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracer les fits et CDF pour les erreurs absolues, signées et les incertitudes.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/patches_caches/train-contrastive1/train-clean", 
        help="Dossier contenant les memmaps"
    )
    args = parser.parse_args()
    
    plot_error_uncertainty_histograms_and_fit(args.output_dir)