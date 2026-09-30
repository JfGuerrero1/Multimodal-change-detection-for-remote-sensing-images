"""
|| Pipeline de diagnostic de dataset  ||
1. Chargement et parcours des scènes hyperspectrales (HSI) et multispectrales (MSI) par blocs (patchs).
2. Extraction de métriques statistiques par patch (taux de zéros, couverture nuageuse, gradients spectraux, RMSE de simulation, etc.).
3. Analyse et construction de graphiques de diagnostic globaux pour identifier les anomalies.
4. Sélection de la meilleure date par scène (mode bitemporal ou date unique) et filtrage des patchs aberrants (création d'une liste noire / blacklist).
5. Sauvegarde des statistiques finales (dataset nettoyé) et génération des visualisations pour les patchs aberrants.
"""

import json
from pathlib import Path

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import seaborn as sns
from tqdm import tqdm
import xarray as xr

from src.constants import SRF_MATRIX, WVL_PRS, WVL_S2
from src.prepare_data.utils_io import (
    get_diff_for_scene,
    stat_dico,
    tryptique_view,
)

#constantes et chemin
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "mumucd"
OUTPUT_DIR_DIAG = ROOT_DIR / "data" / "diag"
DATA_TIME = DATA_DIR / "mumucd_v1_dates.txt"
DEFAULT_SRF_PATH = ROOT_DIR / "data" / "mumucd" / "srf_matrix_norm_s2b.npy"
WVL_S2 = np.array(
    [443.0, 490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 940.0, 1610.0, 2190.0]
)
MASK = (WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | (WVL_PRS > 2000)
DATE_DF = pd.read_csv(
    DATA_TIME,
    names=['scene', 'PRS-before', 'S2-before', 'S1-before', 'PRS-after', 'S2-after', 'S1-after']
)


def build_interp_matrix(wvl_source: np.ndarray, wvl_target: np.ndarray) -> np.ndarray:
    identity = np.eye(len(wvl_source))
    interp_func = scipy.interpolate.interp1d(
        wvl_source, identity, kind="linear", axis=0, fill_value="extrapolate"
    )
    return interp_func(wvl_target).astype(np.float32)


INTERP_MATRIX = build_interp_matrix(WVL_S2, WVL_PRS)
if DEFAULT_SRF_PATH.exists():
    SRF_MATRIX = np.load(DEFAULT_SRF_PATH)


def generate_stats_for_scenes(scene_dir: Path, patch_size: int = 256, date_df: pd.DataFrame = DATE_DF):
    "pour toutes les scènes, gènére un fichier csv avec ses stats"
    scene_name = scene_dir.name
    print(f"Traitement de la scène : {scene_name}")
    paths = {
        'after': {
            'hsi': scene_dir / f"{scene_name}-after-prs.nc",
            'msi': scene_dir / f"{scene_name}-after-s2.nc",
            'dw': scene_dir / f"{scene_name}-after-dw.nc",
        },
        'before': {
            'hsi': scene_dir / f"{scene_name}-before-prs.nc",
            'msi': scene_dir / f"{scene_name}-before-s2.nc",
            'dw': scene_dir / f"{scene_name}-before-dw.nc",
        },
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
                    patch_hsi = ds_hsi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2, 0, 1)
                    patch_msi = ds_msi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2, 0, 1)
                    patch_dw = ds_dw["lcc"][r:r+patch_size, c:c+patch_size].values
                    patch_idx = i * (w_crop // patch_size) + j

                    patch_name = f"{scene_name}_{key}_patch_{patch_idx}"
                    dico = stat_dico(patch_hsi, patch_msi, patch_dw, patch_name, date_df=date_df)

                    if key == 'after':
                        all_stat_after.append(dico)
                    else:
                        all_stat_before.append(dico)

    df_after = pd.DataFrame(all_stat_after)
    df_before = pd.DataFrame(all_stat_before)

    for df, t in [(df_after, 'after'), (df_before, 'before')]:
        df['scene_name'] = scene_name
        df['type'] = t
        df['time_diff'] = get_diff_for_scene(scene_name, date_df, t)

    return df_after, df_before


def load_patch(patch_name: str, dataset_dir: Path, patch_size: int = 256):
    "permet de charger un patch a partir de son numéro, découpage 256*256 par défaut, le patch de la i-eme ligne et de la j-eme colonne, a pour numéro (i-1)*nb_col+(j-1)"
    dataset_dir = Path(dataset_dir)
    patch_size = int(patch_size)

    parts = patch_name.rsplit('_patch_', 1)
    patch_idx = int(parts[1])
    prefix = parts[0]
    scene_name, key = prefix.rsplit('_', 1)

    scene_dir = dataset_dir / scene_name

    files = {
        'hsi': scene_dir / f"{scene_name}-{key}-prs.nc",
        'msi': scene_dir / f"{scene_name}-{key}-s2.nc",
        'dw': scene_dir / f"{scene_name}-{key}-dw.nc",
    }

    with xr.open_dataset(files['hsi']) as ds_hsi, \
         xr.open_dataset(files['msi']) as ds_msi, \
         xr.open_dataset(files['dw']) as ds_dw:

        _, w, _ = ds_hsi['sr'].shape
        num_cols = (w - (w % patch_size)) // patch_size

        i = patch_idx // num_cols
        j = patch_idx % num_cols
        r, c = i * patch_size, j * patch_size

        patch_hsi = ds_hsi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2, 0, 1)
        patch_msi = ds_msi["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32).transpose(2, 0, 1)
        patch_dw = ds_dw["lcc"][r:r+patch_size, c:c+patch_size].to_numpy()

    return patch_hsi, patch_msi, patch_dw


def run_diagnostic_split(df: pd.DataFrame, split_dir: Path, type_label: str, suffix: str):
    split_name = split_dir.name.upper()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    sns.histplot(df['hsi_zeros_vis_percentile'] * 100, bins=200, binrange=(0, 1), ax=axes[0, 0], color='crimson', kde=False)
    axes[0, 0].set_title(f"Taux de zéros visibles ({split_name}) {type_label}")

    sns.histplot(df['dw_percentile_cloud'] * 100, bins=50, ax=axes[0, 1], color='skyblue', kde=False)
    axes[0, 1].set_title(f"Couverture nuageuse ({split_name}) {type_label}")

    if 'max_grad_spectral_high' in df.columns:
        sns.histplot(df['max_grad_spectral_high'], bins=50, ax=axes[0, 2], color='darkmagenta', kde=False)
        axes[0, 2].set_title(f"Gradient spectral maximal ({split_name})")
    else:
        axes[0, 2].set_visible(False)

    sns.histplot(df['hsi_mean'], bins=50, binrange=(0, 0.8), color='teal', label='HSI', alpha=0.6, ax=axes[1, 0])
    sns.histplot(df['msi_mean'], bins=50, binrange=(0, 0.8), color='orange', label='MSI', alpha=0.6, ax=axes[1, 0])
    axes[1, 0].legend()
    axes[1, 0].set_title(f"Réflectances moyennes ({split_name})")

    if 'msi_simulation_rmse' in df.columns:
        sns.histplot(df['msi_simulation_rmse'], bins=50, ax=axes[1, 1], color='gold', kde=False)
        axes[1, 1].set_title(f"RMSE de simulation MSI ({split_name})")
    else:
        axes[1, 1].set_visible(False)

    sns.histplot(df['hsi_std'], bins=50, binrange=(0, 0.3), color='teal', label='Écart-type HSI', alpha=0.6, ax=axes[1, 2])
    sns.histplot(df['msi_std'], bins=50, binrange=(0, 0.3), color='orange', label='Écart-type MSI', alpha=0.6, ax=axes[1, 2])
    axes[1, 2].legend()
    axes[1, 2].set_title(f"Écarts-types ({split_name})")

    sns.scatterplot(data=df, x='hsi_mean', y='msi_mean', alpha=0.5, ax=axes[2, 0], color='purple')
    axes[2, 0].plot([0, 0.6], [0, 0.6], color='red', linestyle='--')
    axes[2, 0].set_title(f"Alignement global HSI vs MSI ({split_name})")

    if 'msi_sim_mean' in df.columns:
        sns.scatterplot(data=df, x='msi_mean', y='msi_sim_mean', alpha=0.5, ax=axes[2, 1], color='dodgerblue')
        max_val = max(df['msi_mean'].max(), df['msi_sim_mean'].max())
        axes[2, 1].plot([0, max_val], [0, max_val], color='red', linestyle='--')
        axes[2, 1].set_title(f"MSI Réel vs MSI Simulé ({split_name})")
        axes[2, 1].set_xlabel("Moyenne MSI Réel (Sentinel-2)")
        axes[2, 1].set_ylabel("Moyenne MSI Simulé (depuis HSI)")
    else:
        axes[2, 1].set_visible(False)

    axes[2, 2].set_visible(False)

    plt.tight_layout()
    split_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_dir / f"dataset_diagnostic_plots_{split_dir.name}_{suffix}.png", dpi=300)
    plt.close()
    print(f"Graphiques de diagnostic sauvegardés pour {split_name}")


def select_best_date_per_scene(df_full: pd.DataFrame) -> dict:
    """ choisit la meilleure date pour chaque scene en fonction du nombre de patch aberrant detecté"""
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


def plot_spectral_gradients(df: pd.DataFrame, output_dir: Path):
    cols_to_plot = ['global_max', 'vis_max', 'vis_zero_max', 'no_atm_max']
    cols_to_plot = [c for c in cols_to_plot if c in df.columns]

    if not cols_to_plot:
        print("Aucune colonne de gradient trouvée dans le DataFrame.")
        return

    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3 * len(cols_to_plot)))
    if len(cols_to_plot) == 1:
        axes = [axes]

    for i, col in enumerate(cols_to_plot):
        data = df[col].dropna()
        color = 'darkorange' if 'zero' in col else 'crimson'

        sns.histplot(data, kde=True, ax=axes[i], color=color)
        axes[i].set_title(f"Distribution du gradient spectral : {col}")
        axes[i].set_xlabel("Valeur du gradient")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "spectral_gradients_distribution.png")
    plt.close()
    print(f"Audit spectral généré dans {output_dir}/spectral_gradients_distribution.png")


def main_pipeline(dataset_dir: Path, output_dir: Path, bitemporal_mode: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dir_aberrant = OUTPUT_DIR_DIAG / 'patch_aberrant'
    dir_aberrant.mkdir(parents=True, exist_ok=True)

    print("Démarrage du pipeline de traitement du dataset...")

    scene_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()], key=lambda x: x.name)

    results = Parallel(n_jobs=4)(
        delayed(generate_stats_for_scenes)(scene_dir, patch_size=256, date_df=DATE_DF)
        for scene_dir in tqdm(scene_dirs, desc='Calcul des métriques par scène')
    )

    all_scenes_after = [r[0] for r in results if r[0] is not None and not r[0].empty]
    all_scenes_before = [r[1] for r in results if r[1] is not None and not r[1].empty]

    df_full = pd.concat(all_scenes_after + all_scenes_before, ignore_index=True)
    df_full.to_csv(output_dir / "dataset_full_stats.csv", index=False)

    if all_scenes_after:
        pd.concat(all_scenes_after, ignore_index=True).to_csv(output_dir / "dataset_after_stats.csv", index=False)

    if all_scenes_before:
        pd.concat(all_scenes_before, ignore_index=True).to_csv(output_dir / "dataset_before_stats.csv", index=False)

    run_diagnostic_split(df_full, output_dir, "Dataset Global", "all_real")

    if bitemporal_mode:
        print("Mode Détection de Changement Bitemporal actif : Analyse conjointe de toutes les paires.")
        df_selected_full = df_full.copy()
    else:
        print("Mode Date Unique actif : Sélection de la meilleure date par scène.")
        selection_dict = select_best_date_per_scene(df_full)

        output_json_path = OUTPUT_DIR_DIAG / "selected_scenes.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(selection_dict, f, indent=4, ensure_ascii=False)

        selected_masks = [
            (df_full['scene_name'] == scene) & (df_full['type'] == best_type)
            for scene, best_type in selection_dict.items()
        ]
        df_selected_full = df_full[np.logical_or.reduce(selected_masks)].copy() if selected_masks else pd.DataFrame()

    run_diagnostic_split(
        df_selected_full,
        output_dir,
        "Dataset Sélectionné (avec aberrants)",
        "selected_with_aberrants",
    )

    cond_aberrant = df_selected_full.get('is_aberrant', False)
    mask_aberrant = cond_aberrant

    df_blacklist = df_selected_full[mask_aberrant].copy()
    blacklist = df_blacklist['name'].tolist()

    pd.DataFrame(blacklist, columns=['name']).to_csv(output_dir / "blacklist.csv", index=False)

    run_diagnostic_split(df_blacklist, output_dir, "Analyse de la Blacklist", "blacklist_only")
    plot_spectral_gradients(df_full, output_dir)

    if not df_blacklist.empty:
        print(f"Génération des visuels pour {len(df_blacklist)} patchs aberrants...")
        for _, row in tqdm(df_blacklist.iterrows(), total=len(df_blacklist), desc="Sauvegarde des PNGs aberrants"):
            patch_name = row['name']
            patch_hsi, patch_msi, patch_dw = load_patch(patch_name, dataset_dir)

            if row.get('is_aberrant', False):
                tryptique_view(patch_msi, patch_hsi, patch_dw, dir_aberrant / f"{patch_name}.png")

    df_clean = df_selected_full[~df_selected_full['name'].isin(blacklist)].copy()
    df_clean.to_csv(output_dir / "dataset_clean_stats.csv", index=False)

    run_diagnostic_split(df_clean, output_dir, "Dataset Nettoyé", "real_clean")
    run_diagnostic_split(df_clean[df_clean['type'] == 'after'], output_dir, "Nettoyé After", "clean_after")
    run_diagnostic_split(df_clean[df_clean['type'] == 'before'], output_dir, "Nettoyé Before", "clean_before")

    print(f"Pipeline terminé avec succès ! Blacklist globale : {len(blacklist)} patchs exclus.")


if __name__ == "__main__":
    main_pipeline(DATA_DIR, OUTPUT_DIR_DIAG)