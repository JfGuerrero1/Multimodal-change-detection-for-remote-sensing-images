import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from src.prepare_data.utils_io import stat_dico, get_diff_for_scene,tryptique_view
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# --- Configuration des chemins ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'mumucd'
OUTPUT_DIR_DIAG = ROOT_DIR/'data'/'diag'
DATA_TIME = DATA_DIR / 'mumucd_v1_dates.txt'

DATE_DF = pd.read_csv(DATA_TIME, names=['scene', 'PRS-before', 'S2-before', 'S1-before', 'PRS-after', 'S2-after', 'S1-after'])

def genere_stat_for_scenes(scene_dir, patch_size=256,date_df=DATE_DF):
    scene_name = scene_dir.name
    print(f"Traitement {scene_name}")
    paths = {
        'after': {'hsi': scene_dir/f"{scene_name}-after-prs.nc", 
                  'msi': scene_dir/f"{scene_name}-after-s2.nc", 
                  'dw': scene_dir/f"{scene_name}-after-dw.nc"},
        'before': {'hsi': scene_dir/f"{scene_name}-before-prs.nc", 
                   'msi': scene_dir/f"{scene_name}-before-s2.nc", 
                   'dw': scene_dir/f"{scene_name}-before-dw.nc"}
    }

    all_stat_after, all_stat_before = [], []
    
    for key, files in paths.items():
        with xr.open_dataset(files['hsi']) as ds_hsi, \
             xr.open_dataset(files['msi']) as ds_msi, \
             xr.open_dataset(files['dw']) as ds_dw:
            
            h, w, _ = ds_hsi['sr'].values.shape
            patch_size = int(patch_size)
            h, w = int(h), int(w)
            h_crop, w_crop = h - (h % patch_size), w - (w % patch_size)
            
            for i in range(h_crop // patch_size):
                for j in range(w_crop // patch_size):
                    r, c = i * patch_size, j * patch_size
                    patch_hsi = ds_hsi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2,0,1)
                    patch_msi = ds_msi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2,0,1)
                    patch_dw = ds_dw["lcc"][r:r+patch_size, c:c+patch_size].values
                    patch_idx = i * (w_crop // patch_size) + j

                    
                    patch_name = f"{scene_name}_{key}_patch_{patch_idx}"
                    dico = stat_dico(patch_hsi, patch_msi, patch_dw, patch_name,date_df=DATE_DF)
                    
                    if key == 'after': all_stat_after.append(dico)
                    else: all_stat_before.append(dico)

    df_after = pd.DataFrame(all_stat_after)
    df_before = pd.DataFrame(all_stat_before)

    for df, t in [(df_after, 'after'), (df_before, 'before')]:
        df['scene_name'] = scene_name
        df['type'] = t
        df['time_diff'] = get_diff_for_scene(scene_name, DATE_DF, t)
    
    return df_after, df_before


def load_patch(patch_name, dataset_dir, patch_size=256):
    """
    Reconstitue les fenêtres HSI, MSI et DW à la volée à partir du patch_name.
    
    Exemple de patch_name : 'scene_01_after_patch_12'
    
    Returns:
        tuple: (patch_hsi, patch_msi, patch_dw)
    """
    dataset_dir = Path(dataset_dir)
    patch_size = int(patch_size)

    # 1. Extraction des composants du nom
    # 'scene_01_after_patch_12' -> scene_name='scene_01', key='after', patch_idx=12
    parts = patch_name.rsplit('_patch_', 1)
    patch_idx = int(parts[1])
    
    prefix = parts[0]
    scene_name, key = prefix.rsplit('_', 1) # 'scene_01', 'after'

    scene_dir = dataset_dir / scene_name

    # 2. Reconstitution des chemins NetCDF
    files = {
        'hsi': scene_dir / f"{scene_name}-{key}-prs.nc",
        'msi': scene_dir / f"{scene_name}-{key}-s2.nc",
        'dw': scene_dir / f"{scene_name}-{key}-dw.nc"
    }

    # 3. Ouverture des fichiers et découpage spatial
    with xr.open_dataset(files['hsi']) as ds_hsi, \
         xr.open_dataset(files['msi']) as ds_msi, \
         xr.open_dataset(files['dw']) as ds_dw:

        h, w, _ = ds_hsi['sr'].shape
        num_cols = (w - (w % patch_size)) // patch_size

        # Inversion de l'index 1D vers la grille 2D (i, j)
        i = patch_idx // num_cols
        j = patch_idx % num_cols

        r, c = i * patch_size, j * patch_size

        # Découpage et reformatage (Channel First: C, H, W)
        patch_hsi = ds_hsi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2, 0, 1)
        patch_msi = ds_msi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2, 0, 1)
        patch_dw  = ds_dw["lcc"][r:r+patch_size, c:c+patch_size].to_numpy()

    return patch_hsi, patch_msi, patch_dw 

def run_diagnostic_split(df, split_dir, type_label, suffix):
    split_name = split_dir.name.upper()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # --- LIGNE 0 : Qualité et Anomalies Brutes ---
    sns.histplot(df['hsi_zeros_vis_percentile'] * 100, bins=200, binrange=(0,1), ax=axes[0, 0], color='crimson', kde=False)
    axes[0, 0].set_title(f"Taux de zéros visible ({split_name}) {type_label}")
    
    sns.histplot(df['dw_percentile_cloud'] * 100, bins=50, ax=axes[0, 1], color='skyblue', kde=False)
    axes[0, 1].set_title(f"Couverture nuageuse ({split_name}) {type_label}")
    
    if 'max_grad_spectral_high' in df.columns:
        sns.histplot(df['max_grad_spectral_high'], bins=50, ax=axes[0, 2], color='darkmagenta', kde=False)
        axes[0, 2].set_title(f"Max Gradient Spectral ({split_name})")
    else:
        axes[0, 2].set_visible(False)

    # --- LIGNE 1 : Statistiques Radiométriques & Simulation ---
    sns.histplot(df['hsi_mean'], bins=50, binrange=(0, 0.8), color='teal', label='HSI', alpha=0.6, ax=axes[1, 0])
    sns.histplot(df['msi_mean'], bins=50, binrange=(0, 0.8), color='orange', label='MSI', alpha=0.6, ax=axes[1, 0])
    axes[1, 0].legend()
    axes[1, 0].set_title(f"Réflectances Moyennes ({split_name})")

    if 'msi_simulation_rmse' in df.columns:
        sns.histplot(df['msi_simulation_rmse'], bins=50, ax=axes[1, 1], color='gold', kde=False)
        axes[1, 1].set_title(f"RMSE Simulation MSI ({split_name})")
    else:
        axes[1, 1].set_visible(False)

    sns.histplot(df['hsi_std'], bins=50, binrange=(0, 0.3), color='teal', label='Std HSI', alpha=0.6, ax=axes[1, 2])
    sns.histplot(df['msi_std'], bins=50, binrange=(0, 0.3), color='orange', label='Std MSI', alpha=0.6, ax=axes[1, 2])
    axes[1, 2].legend()
    axes[1, 2].set_title(f"Écarts-types ({split_name})")

    # --- LIGNE 2 : Corrélations ---
    # 7. Aligment global HSI vs MSI
    sns.scatterplot(data=df, x='hsi_mean', y='msi_mean', alpha=0.5, ax=axes[2, 0], color='purple')
    axes[2, 0].plot([0, 0.6], [0, 0.6], color='red', linestyle='--')
    axes[2, 0].set_title(f"Alignement Moyen HSI vs MSI ({split_name})")

    # 8. SCATTER PLOT : Vrai MSI vs MSI Simulé (NOUVEAU)
    if 'msi_sim_mean' in df.columns:
        sns.scatterplot(data=df, x='msi_mean', y='msi_sim_mean', alpha=0.5, ax=axes[2, 1], color='dodgerblue')
        max_val = max(df['msi_mean'].max(), df['msi_sim_mean'].max())
        axes[2, 1].plot([0, max_val], [0, max_val], color='red', linestyle='--')
        axes[2, 1].set_title(f"MSI Réel vs MSI Simulé ({split_name})")
        axes[2, 1].set_xlabel("Moyenne MSI Réel (Sentinel-2)")
        axes[2, 1].set_ylabel("Moyenne MSI Simulé (depuis HSI)")
    else:
        axes[2, 1].set_visible(False)

    # Masquage de la dernière case inutilisée
    axes[2, 2].set_visible(False)

    plt.tight_layout()
    split_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_dir / f"dataset_diagnostic_plots_{split_dir.name}_{suffix}.png", dpi=300)
    plt.close()
    print(f"📊 Graphiques sauvegardés pour {split_name}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def analyze_patch_aberrant(patch_hsi, patch_msi, metrics_row, patch_name, output_dir, SRF_MATRIX=None, WVL_PRS=None):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    
    # 1. Histogramme HSI
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(patch_hsi.flatten(), bins=50, ax=ax1, color='teal')
    ax1.set_title(f"Histogramme HSI : {patch_name}")
    
    # 2. Zéros par canal
    ax2 = fig.add_subplot(gs[0, 1])
    raw_zeros = metrics_row.get('nb_zero_per_channel', [])
    zeros_per_chan = ast.literal_eval(raw_zeros) if isinstance(raw_zeros, str) else raw_zeros
    if len(zeros_per_chan) > 0:
        ax2.plot(zeros_per_chan, color='crimson')
    ax2.set_title("Zéros par Wavelength")
    
    # 3. Scatter Plot : Vrai MSI vs MSI Simulé (via SRF_MATRIX)
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        # Assurer les bonnes dimensions (C, H, W) pour le calcul
        p_hsi = np.moveaxis(patch_hsi, -1, 0) if patch_hsi.shape[0] > patch_hsi.shape[2] else patch_hsi
        p_msi = np.moveaxis(patch_msi, -1, 0) if patch_msi.shape[0] > patch_msi.shape[2] else patch_msi
        
        if SRF_MATRIX is not None:
            hsi_hwc = np.moveaxis(p_hsi, 0, -1)
            h, w, c_hsi = hsi_hwc.shape
            hyper_2d = hsi_hwc.reshape(-1, c_hsi)
            scene_multi_sim_hwc = np.dot(hyper_2d, SRF_MATRIX).reshape(h, w, -1).astype(np.float32)
            patch_msi_sim = np.moveaxis(scene_multi_sim_hwc, -1, 0)
            
            # Comparaison sur la bande Rouge (index 2) par exemple
            flat_real = p_msi[2].flatten()
            flat_sim = patch_msi_sim[2].flatten()
            
            n_samples = min(1000, len(flat_real))
            if n_samples > 0:
                idx = np.random.choice(len(flat_real), n_samples, replace=False)
                ax3.scatter(flat_real[idx], flat_sim[idx], alpha=0.3, color='purple', s=5)
                v_min = min(flat_real.min(), flat_sim.min())
                v_max = max(flat_real.max(), flat_sim.max())
                ax3.plot([v_min, v_max], [v_min, v_max], 'r--')
            ax3.set_title("MSI Réel vs Simulé (Bande Rouge)")
            ax3.set_xlabel("S2 Réel")
            ax3.set_ylabel("Simulé (HSI)")
        else:
            ax3.text(0.5, 0.5, "SRF_MATRIX non disponible", ha='center', va='center')
    except Exception as e:
        ax3.text(0.5, 0.5, f"Erreur sim: {e}", ha='center', va='center', fontsize=8)
    
    # 4. Profil Spectral HSI
    ax4 = fig.add_subplot(gs[1:, :])
    try:
        means = ast.literal_eval(metrics_row['mean_per_channel']) if isinstance(metrics_row['mean_per_channel'], str) else metrics_row['mean_per_channel']
        mins = ast.literal_eval(metrics_row['min_per_channel']) if isinstance(metrics_row['min_per_channel'], str) else metrics_row['min_per_channel']
        maxs = ast.literal_eval(metrics_row['max_per_channel']) if isinstance(metrics_row['max_per_channel'], str) else metrics_row['max_per_channel']
        
        x = range(len(means))
        ax4.plot(x, means, color='black', label='Moyenne')
        ax4.fill_between(x, mins, maxs, color='gray', alpha=0.3, label='Variation Min/Max')
        if WVL_PRS is not None and len(WVL_PRS) == len(means):
            ax4.set_xlabel("Longueur d'onde (nm)")
    except Exception as e:
        ax4.text(0.5, 0.5, f"Erreur profil spectral: {e}", ha='center', va='center')
        
    ax4.set_title("Profil Spectral HSI")
    ax4.legend()

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{patch_name}_diagnostic.png", dpi=200)
    plt.close()

def select_best_date_per_scene(df_full):
    selection_dict = {}
    for scene, group in df_full.groupby('scene_name'):
        before_data = group[group['type'] == 'before']
        after_data = group[group['type'] == 'after']
        
        ratio_b = before_data['is_aberrant'].mean() if not before_data.empty else 1.0
        ratio_a = after_data['is_aberrant'].mean() if not after_data.empty else 1.0
        
        dt_b = before_data['time_diff'].iloc[0] if not before_data.empty and 'time_diff' in before_data.columns else 999
        dt_a = after_data['time_diff'].iloc[0] if not after_data.empty and 'time_diff' in after_data.columns else 999

        if ratio_b < ratio_a:
            selection_dict[scene] = 'before'
        elif ratio_a < ratio_b:
            selection_dict[scene] = 'after'
        else:
            selection_dict[scene] = 'before' if dt_b <= dt_a else 'after'
            
    return selection_dict


def plot_exclusion_stats(df, output_dir):
    # On isole uniquement les causes d'exclusion
    causes = ["is_cloudy","is_spatially_dead", "is_inconsistent_sim", "gradient_spatial_high"]
    counts = df[causes].sum()
    
    # Création du graphique
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen', 'orange'])
    
    plt.title('Répartition des motifs d\'exclusion de patchs')
    plt.ylabel('Nombre de patchs')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Sauvegarde
    plt.tight_layout()
    plt.savefig(output_dir / "audit_exclusions.png")
    plt.close()
    print(f"Graphique d'audit sauvegardé dans {output_dir}/audit_exclusions.png")


def plot_spectral_gradients(df, output_dir):
    """
    Visualise la stabilité spectrale du dataset via les gradients.
    """
    # On ajoute les nouvelles métriques des pixels nuls si elles existent dans le DataFrame
    cols_to_plot = ['global_max', 'vis_max', 'vis_zero_max','no_atm_max']
    cols_to_plot = [c for c in cols_to_plot if c in df.columns]
    
    if not cols_to_plot:
        print("Aucune colonne de gradient trouvée dans le DataFrame.")
        return

    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3 * len(cols_to_plot)))
    if len(cols_to_plot) == 1: axes = [axes]
    
    for i, col in enumerate(cols_to_plot):
        # On utilise un échantillon si le dataset est trop gros pour la RAM
        data = df[col].dropna()

        # Utilisation d'une couleur différente pour les zéros pour les distinguer visuellement
        color = 'darkorange' if 'zero' in col else 'crimson'
        
        sns.histplot(data, kde=True, ax=axes[i], color=color)
        axes[i].set_title(f'Distribution du gradient spectral : {col}')
        axes[i].set_xlabel('Valeur du gradient')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "spectral_gradients_distribution.png")
    plt.close()
    print(f"Audit spectral généré dans {output_dir}/spectral_gradients_distribution.png")
    


def main_pipeline(dataset_dir, output_dir):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Création des dossiers de diagnostic pour les visuels
    dir_aberrant = OUTPUT_DIR_DIAG / 'patch_aberrant'
    dir_aberrant_ratio = OUTPUT_DIR_DIAG / 'patch_aberrant_ratio'
    dir_aberrant.mkdir(parents=True, exist_ok=True)
    dir_aberrant_ratio.mkdir(parents=True, exist_ok=True)
    
    print(" Début du traitement du dataset...")

    scene_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()], key=lambda x: x.name)

    # 2. Exécution parallèle de la génération des statistiques
    results = Parallel(n_jobs=-1)(
        delayed(genere_stat_for_scenes)(scene_dir, patch_size=256, date_df=DATE_DF)
        for scene_dir in tqdm(scene_dirs, desc='Calcul des métriques par scène')
    )

    all_scenes_after = [r[0] for r in results if r[0] is not None and not r[0].empty]
    all_scenes_before = [r[1] for r in results if r[1] is not None and not r[1].empty]

    # 3. Assemblage et sauvegarde des CSVs globaux
    df_full = pd.concat(all_scenes_after + all_scenes_before, ignore_index=True)
    df_full.to_csv(output_dir / "dataset_full_stats.csv", index=False)

    if all_scenes_after:
        pd.concat(all_scenes_after, ignore_index=True).to_csv(
            output_dir / "dataset_after_stats.csv", index=False
        )

    if all_scenes_before:
        pd.concat(all_scenes_before, ignore_index=True).to_csv(
            output_dir / "dataset_before_stats.csv", index=False
        )
    
    #df_full=pd.read_csv("/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/diag/dataset_full_stats.csv")
    # Diagnostics initiaux sur l'ensemble
    run_diagnostic_split(df_full, output_dir, "Dataset Global", "all_real")

    # 4. Sélection des meilleures dates par scène
    selection_dict = select_best_date_per_scene(df_full)

    output_json_path = OUTPUT_DIR_DIAG / "selected_scenes.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(selection_dict, f, indent=4, ensure_ascii=False)

    print(f" Dictionnaire de sélection des dates sauvegardé pour {len(selection_dict)} scènes.")
    
    # Application correcte de la sélection scène par scène
    selected_masks = [
        (df_full['scene_name'] == scene) & (df_full['type'] == best_type)
        for scene, best_type in selection_dict.items()
    ]
    df_full=pd.read_csv("/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/diag/dataset_full_stats.csv")
    df_selected_full = df_full[np.logical_or.reduce(selected_masks)].copy() if selected_masks else pd.DataFrame()

    run_diagnostic_split(
        df_selected_full,
        output_dir,
        "Dataset Sélectionné (avec aberrants)",
        "selected_with_aberrants2",
    )
    
   # 5. Extraction et sauvegarde de la Blacklist (UNIQUEMENT dans la sélection retenue)
    cond_aberrant = df_selected_full.get('is_aberrant', True)
    cond_ratio = df_selected_full.get('is_aberrant_ratio', True)
    
    # Masque des patchs aberrants parmi les scènes sélectionnées
    mask_aberrant = cond_aberrant 
    
    # Récupération uniquement des vrais aberrants
    df_blacklist = df_selected_full[mask_aberrant].copy()
    blacklist = df_blacklist['name'].tolist()
    
    pd.DataFrame(blacklist, columns=['name']).to_csv(
        output_dir / "blacklist.csv", index=False
    )

    run_diagnostic_split(
        df_blacklist, output_dir, "Analyse Blacklist", "blacklist_only2"
    )

    #plot_exclusion_stats(df_selected_full, output_dir)
    plot_spectral_gradients(df_full, output_dir)

    # 6. Génération DIFFÉRÉE des triptyques PNG pour les patchs aberrants uniquement
    df_aberrants = df_blacklist  # On réutilise directement les aberrants de la sélection !
    
    if not df_aberrants.empty:
        print(f" Génération des visuels pour les {len(df_aberrants)} patchs détectés comme aberrants...")
        
        for _, row in tqdm(df_aberrants.iterrows(), total=len(df_aberrants), desc="Sauvegarde PNGs aberrants"):
            patch_name = row['name']
            patch_hsi, patch_msi, patch_dw = load_patch(patch_name, dataset_dir)
            
            if row.get('is_aberrant', False):
                tryptique_view(patch_msi, patch_hsi, patch_dw, dir_aberrant / f"{patch_name}.png")
            
            if row.get('is_aberrant_ratio', False):
                tryptique_view(patch_msi, patch_hsi, patch_dw, dir_aberrant_ratio / f"{patch_name}.png")

    # 7. Dataset propre (nettoyé) : On filtre DEPUIS df_selected_full !
    df_clean = df_selected_full[~df_selected_full['name'].isin(blacklist)].copy()
    df_clean.to_csv(output_dir / "dataset_clean_stats2.csv", index=False)

    run_diagnostic_split(df_clean, output_dir, "Clean Dataset", "real_clean2")
    run_diagnostic_split(
        df_clean[df_clean['type'] == 'after'],
        output_dir,
        "Clean After",
        "clean_after2",
    )
    run_diagnostic_split(
        df_clean[df_clean['type'] == 'before'],
        output_dir,
        "Clean Before",
        "clean_before2",
    )

    print(f"🎉 Pipeline terminé avec succès ! Blacklist : {len(blacklist)} patchs exclus.")
if __name__ == "__main__":
    main_pipeline(DATA_DIR, OUTPUT_DIR_DIAG)

    """
    df=pd.read_csv("/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/diag/dataset_full_stats.csv")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution du taux de pixels zéros HSI
    sns.histplot(df['hsi_zero_percentile'], bins=50, ax=axes[0, 0], binrange=(0,0.3) ,color='royalblue', kde=True)
    axes[0, 0].set_title("Distribution du taux de pixels zéros (HSI)")
    axes[0, 0].set_xlabel("Pourcentage de zéros")
    axes[0, 0].set_ylabel("Nombre de patchs")

# 2. Distribution du gradient max / percentile de rupture global
    sns.histplot(df['global_percentile_morts'], bins=50, ax=axes[0, 1],binrange=(0,0.1), color='darkorange', kde=True)
    axes[0, 1].set_title("Distribution des pourcentages de rupture / morts (Global)")
    axes[0, 1].set_xlabel("Pourcentage de pixels en rupture")
    axes[0, 1].set_ylabel("Nombre de patchs")

# 3. Distribution du pourcentage de nuages (dw_percentile_cloud)
    sns.histplot(df['dw_percentile_cloud'], bins=50, ax=axes[1, 0], color='forestgreen', kde=True)
    axes[1, 0].set_title("Distribution du pourcentage de nuages")
    axes[1, 0].set_xlabel("Pourcentage de nuages")
    axes[1, 0].set_ylabel("Nombre de patchs")

# 4. Distribution de ratio_mean_vis_hsi_msi
    sns.histplot(df['ratio_mean_vis_hsi_msi'], bins=50, binrange=(0,1), ax=axes[1, 1], color='purple' )
    axes[1, 1].set_title("Distribution du ratio mean VIS (HSI / MSI)")
    axes[1, 1].set_xlabel("Ratio mean VIS HSI/MSI")
    axes[1, 1].set_ylabel("Nombre de patchs")

    plt.tight_layout()
    plt.savefig('histograms_analysis_ratio.png', dpi=300)
    print("Graphique mis à jour sauvegardé sous 'histograms_analysis_ratio.png'")
    """

