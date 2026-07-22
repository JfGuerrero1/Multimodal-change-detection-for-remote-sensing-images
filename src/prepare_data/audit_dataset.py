import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from src.prepare_data.utils_io import stat_dico, get_diff_for_scene

# --- Configuration des chemins ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'mumucd'
OUTPUT_DIR_DIAG = ROOT_DIR/'data'/'diag'
DATA_TIME = DATA_DIR / 'mumucd_v1_dates.txt'

DATE_DF = pd.read_csv(DATA_TIME, names=['scene', 'PRS-before', 'S2-before', 'S1-before', 'PRS-after', 'S2-after', 'S1-after'])

def genere_stat_for_scenes(scene_dir, patch_size=256):
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
            h_crop, w_crop = h - (h % patch_size), w - (w % patch_size)
            
            for i in range(h_crop // patch_size):
                for j in range(w_crop // patch_size):
                    r, c = i * patch_size, j * patch_size
                    patch_hsi = ds_hsi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2,0,1)
                    patch_msi = ds_msi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2,0,1)
                    patch_dw = ds_dw["lcc"][r:r+patch_size, c:c+patch_size]
                    patch_idx = i * (w_crop // patch_size) + j

                    
                    patch_name = f"{scene_name}_{key}_patch_{patch_idx}"
                    dico = stat_dico(patch_hsi, patch_msi, patch_dw, patch_name)
                    
                    if key == 'after': all_stat_after.append(dico)
                    else: all_stat_before.append(dico)

    df_after = pd.DataFrame(all_stat_after)
    df_before = pd.DataFrame(all_stat_before)

    for df, t in [(df_after, 'after'), (df_before, 'before')]:
        df['scene_name'] = scene_name
        df['type'] = t
        df['time_diff'] = get_diff_for_scene(scene_name, DATE_DF, t)
    
    return df_after, df_before

def run_diagnostic_split(df, split_dir, type_label, suffix):
    split_name = split_dir.name.upper()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    sns.histplot(df['hsi_zeros_vis_percentile'] * 100, bins=200, binrange=(0,1), ax=axes[0, 0], color='crimson', kde=False)
    axes[0, 0].set_title(f"Taux de zéros dans le visible ({split_name}) {type_label}")
    sns.histplot(df['dw_percentile_cloud'] * 100, bins=50, ax=axes[0, 1], color='skyblue', kde=False)
    axes[0, 1].set_title(f"Couverture nuageuse ({split_name}) {type_label}")
    
    sns.histplot(df['hsi_mean'], bins=50, binrange=(0, 0.8), color='teal', label='HSI', alpha=0.6, ax=axes[1, 0])
    sns.histplot(df['msi_mean'], bins=50, binrange=(0, 0.8), color='orange', label='MSI', alpha=0.6, ax=axes[1, 0])
    axes[1, 0].legend()
    axes[1, 0].set_title(f"Réflectances Moyennes ({split_name})")

    sns.histplot(df["hsi_brillant_vis_percentile"] * 100, bins=30, ax=axes[1, 1], color='gold', kde=False)
    axes[1, 1].set_title(f"Saturation ({split_name})")

    sns.histplot(df['hsi_std'], bins=50, binrange=(0, 0.3), color='teal', label='Std HSI', alpha=0.6, ax=axes[2, 0])
    sns.histplot(df['msi_std'], bins=50, binrange=(0, 0.3), color='orange', label='Std MSI', alpha=0.6, ax=axes[2, 0])
    axes[2, 0].legend()

    sns.scatterplot(data=df, x='hsi_mean', y='msi_mean', alpha=0.5, ax=axes[2, 1], color='purple')
    axes[2, 1].plot([0, 0.6], [0, 0.6], color='red', linestyle='--')
    axes[2, 1].set_title(f"Alignement ({split_name})")

    plt.tight_layout()
    plt.savefig(split_dir / f"dataset_diagnostic_plots_{split_dir.name}_{suffix}.png", dpi=300)
    plt.close()
    print(f"📊 Graphiques sauvegardés pour {split_name}")

def analyze_patch_aberrant(patch_hsi, patch_msi, metrics_row, patch_name, output_dir):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(patch_hsi.flatten(), bins=50, ax=ax1, color='teal')
    ax1.set_title(f"Histogramme HSI : {patch_name}")
    
    ax2 = fig.add_subplot(gs[0, 1])
    zeros_per_chan = eval(metrics_row['nb_zero_per_channel']) if isinstance(metrics_row['nb_zero_per_channel'], str) else metrics_row['nb_zero_per_channel']
    ax2.plot(zeros_per_chan, color='crimson')
    ax2.set_title("Zéros par Wavelength")
    
    ax3 = fig.add_subplot(gs[0, 2])
    idx = np.random.choice(patch_hsi.shape[-1], 1000, replace=False)
    ax3.scatter(patch_msi.flatten()[idx], patch_hsi.flatten()[idx], alpha=0.3, color='purple', s=5)
    ax3.plot([0, 1], [0, 1], 'r--')
    ax3.set_title("Corrélation Pixel MSI vs HSI")
    
    ax4 = fig.add_subplot(gs[1:, :])
    means, mins, maxs = eval(metrics_row['mean_per_channel']), eval(metrics_row['min_per_channel']), eval(metrics_row['max_per_channel'])
    x = range(len(means))
    ax4.plot(x, means, color='black', label='Moyenne')
    ax4.fill_between(x, mins, maxs, color='gray', alpha=0.3, label='Variation Min/Max')
    ax4.set_title("Profil Spectral")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"{patch_name}_diagnostic.png", dpi=200)
    plt.close()

def select_best_date_per_scene(df_full):
    selection_dict = {}
    for scene, group in df_full.groupby('scene_name'):
        before_data = group[group['type'] == 'before']
        after_data = group[group['type'] == 'after']
        
        ratio_b = before_data['is_aberrant'].mean() 
        ratio_a = after_data['is_aberrant'].mean() 
        dt_b = before_data['time_diff'].iloc[0] 
        dt_a = after_data['time_diff'].iloc[0] 
        

        if ratio_b < ratio_a: selection_dict[scene] = 'before'
        elif ratio_a < ratio_b: selection_dict[scene] = 'after'
        else: selection_dict[scene] = 'before' if dt_b <= dt_a else 'after'
    return selection_dict


def plot_exclusion_stats(df, output_dir):
    # On isole uniquement les causes d'exclusion
    causes = ['is_cloudy', 'is_spatially_dead', 'is_saturated', 'is_spectrally_dead','gradient_spatial_high']
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
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scenes_after, all_scenes_before = [], []
    print("Debut du traitement")
    scene_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
   
    """
    for scene_dir in tqdm(scene_dirs, desc='traitement des scenes'):
        df_a, df_b = genere_stat_for_scenes(scene_dir)
        all_scenes_after.append(df_a)
        all_scenes_before.append(df_b)
    

    all_scenes_after_df = pd.DataFrame(all_scenes_after)
    all_scenes_before_df=pd.DataFrame(all_scenes_before)
    all_scenes_after_df.to_csv(output_dir/"dataset_after_stats.csv",index=False)
    all_scenes_before_df.to_csv(output_dir/"dataset_before_stats.csv",index=False)
    """

    df_full = pd.concat(all_scenes_after + all_scenes_before, ignore_index=True)
    df_full.to_csv(output_dir / "dataset_full_stats.csv", index=False)
    
    #df_full=pd.read_csv("/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/diag/dataset_full_stats.csv")
    run_diagnostic_split(df_full, output_dir, "Dataset Global", "all_real")
    
    selection_dict = select_best_date_per_scene(df_full)
    
    # 1. Dataset avec les scènes choisies (incluant les aberrants)
    selected_names = []
    for scene, choice in selection_dict.items():
        subset = df_full[(df_full['scene_name'] == scene) & (df_full['type'] == choice)]
        selected_names.extend(subset['name'].tolist())
    
    df_selected_full = df_full[df_full['name'].isin(selected_names)]
    run_diagnostic_split(df_selected_full, output_dir, "Dataset Selectionné (avec aberrants)", "selected_with_aberrants2")

    
    blacklist = df_selected_full[df_selected_full['is_aberrant'] == True]['name'].tolist()
    pd.DataFrame(blacklist, columns=['name']).to_csv(output_dir / "blacklist.csv", index=False)

    df_blacklist = df_full[df_full['name'].isin(blacklist)]
    run_diagnostic_split(df_blacklist, output_dir, "Analyse Blacklist", "blacklist_only2")

    plot_exclusion_stats(df_selected_full, output_dir)
    plot_spectral_gradients(df_full,output_dir)
    
    # 4. Dataset propre (nettoyé)
    df_clean = df_full[~df_full['name'].isin(blacklist)]
    df_clean.to_csv(output_dir / "dataset_clean_stats2.csv", index=False)
    
    # 5. Diagnostics finaux
    run_diagnostic_split(df_clean, output_dir, "Clean Dataset", "real_clean2")
    run_diagnostic_split(df_full[df_full['type'] == 'after'], output_dir, " After", "clean_after2")
    run_diagnostic_split(df_full[df_full['type'] == 'before'], output_dir, " Before", "clean_before2")

    print(f" Blacklist: {len(blacklist)} patchs exclus.")

if __name__ == "__main__":
    #main_pipeline(DATA_DIR, OUTPUT_DIR_DIAG)


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

