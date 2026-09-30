import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import scipy.interpolate
from matplotlib.ticker import MaxNLocator
import time
import gc
import imageio
import h5py
from src.visualise.save_fig import store_data_png, save_data_with_ext

# ==========================================
# CONSTANTES ET CONFIGURATION DES CHEMINS
# ==========================================

# Longueurs d'onde de référence pour Sentinel-2 (MSI)
wvl_s2 = np.array([443., 490., 560., 665., 705., 740., 783., 842., 865., 940., 1610., 2190.])

# Longueurs d'onde de référence pour PRISMA (HSI)
WVL_PRS = np.array([
    406.9934, 415.839, 423.78476, 431.3347, 438.6569, 446.0147, 453.38947, 460.73175, 468.09842, 475.31885,
    482.54816, 489.79486, 497.05865, 504.51172, 512.0464, 519.54376, 527.3053, 535.05255, 542.88513, 550.9146,
    559.02026, 567.2061, 575.4868, 583.8441, 592.339, 601.0144, 609.9582, 618.72, 627.77844, 636.6763,
    645.9638, 655.41876, 664.8941, 674.46436, 684.13727, 694.12836, 703.737, 713.72687, 723.87994, 733.9552,
    744.14954, 754.4696, 764.85645, 775.2735, 785.65955, 796.127, 806.71106, 817.31104, 827.9195, 838.5272,
    849.20996, 859.97314, 870.74255, 881.45605, 892.08093, 902.80164, 913.44507, 923.9502, 934.11206, 944.6273,
    956.2715, 967.0267, 977.3654, 979.224, 988.9179, 998.9082, 1008.6443, 1018.5357, 1029.344, 1037.9878,
    1047.675, 1057.5737, 1067.7948, 1078.2161, 1088.761, 1099.2776, 1109.8894, 1120.6759, 1131.3048, 1142.0703,
    1152.6501, 1163.676, 1174.7142, 1185.5884, 1196.3394, 1207.2737, 1217.8635, 1229.1852, 1240.2145, 1250.9799,
    1262.5322, 1273.4963, 1284.4878, 1295.4218, 1306.218, 1317.2566, 1328.2993, 1339.1294, 1349.7877, 1361.0531,
    1372.9117, 1383.2798, 1394.754, 1405.6268, 1416.5374, 1427.3748, 1438.466, 1449.1888, 1459.3157, 1469.9308,
    1480.8422, 1491.4292, 1502.0236, 1512.6333, 1523.2222, 1533.7764, 1544.2262, 1554.8168, 1565.3688, 1575.6274,
    1585.8597, 1596.2454, 1606.4913, 1616.8336, 1627.021, 1637.0919, 1647.2316, 1656.933, 1667.185, 1677.3193,
    1687.4269, 1697.2943, 1707.0945, 1716.8589, 1726.6516, 1736.4883, 1746.2192, 1755.833, 1765.5127, 1775.1178,
    1784.7173, 1793.9531, 1803.5902, 1813.0514, 1822.4413, 1832.0272, 1841.3256, 1850.5543, 1859.5587, 1868.1732,
    1878.7426, 1887.081, 1896.0913, 1904.9347, 1914.3015, 1923.3857, 1932.2599, 1941.1107, 1949.9008, 1958.6244,
    1967.3418, 1976.013, 1984.853, 1993.5482, 2002.1106, 2010.6614, 2019.3214, 2027.7267, 2036.2607, 2044.6809,
    2053.0078, 2061.3787, 2069.7957, 2077.9915, 2086.3823, 2094.6252, 2102.8213, 2111.039, 2119.2314, 2127.3372,
    2135.5103, 2143.4656, 2151.3862, 2159.564, 2167.4849, 2175.3442, 2183.4202, 2191.1003, 2199.1353, 2206.843,
    2214.625, 2222.4263, 2230.0076, 2237.904, 2245.4485, 2253.1104, 2260.8665, 2268.2883, 2276.0537, 2283.4934,
    2290.8267, 2298.6094, 2305.7227, 2313.2007, 2320.8955, 2327.8242, 2335.5264, 2342.8228, 2349.7915, 2357.2937,
    2364.5945, 2371.5522, 2378.771, 2386.0618, 2393.0388, 2400.036, 2407.6045, 2414.3567, 2421.2373, 2428.6677,
    2435.5442, 2442.403, 2449.1423, 2456.5857, 2463.0303, 2469.6272, 2477.055, 2483.793, 2490.2192, 2497.1155
])

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = Path("/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images")
 
DATA_DIR = ROOT_DIR / "data" / "mumucd"
CACHE_DIR = DATA_DIR / "patches_cache"
DEFAULT_SRF_PATH = DATA_DIR / "srf_matrix_norm_s2b.npy"
RESULT_DIR = ROOT_DIR / "results"
PLOT_DIR = RESULT_DIR / "Result_plot"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT_DIR / "models"
RGB_DIR = DATA_DIR / 'images_rgb'

# ==========================================
# FONCTIONS DE TRAITEMENT ET VISUALISATION
# ==========================================

def get_rgb(data, is_prisma=False, save_path="output.png", title="VUE RGB", add_stat=False, return_only=False):
    data_tensor = data

    
    # PRISMA : indice 32 (~665nm), indice 20 (~560nm), indice 11 (~490nm)
    indices = [32, 20, 11] if is_prisma else [3, 2, 1]

    if any(i >= data_tensor.shape[0] for i in indices):
        raise IndexError("Index de bande hors limites pour ce tenseur.")

    r, g, b = data_tensor[indices[0]], data_tensor[indices[1]], data_tensor[indices[2]]
        
    rgb = np.stack([r, g, b], axis=-1)

    # 2. Normalisation par percentiles (2% - 98%) pour éviter la saturation
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip(rgb, p2, p98)
    rgb = (rgb - p2) / (p98 - p2 + 1e-8)

    #SI on veut juste le tableua numpy sans eneregistrer la figure
    if return_only:
        return rgb

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

    return rgb

def build_interp_matrix(wvl_source, wvl_target):
    """Construit la matrice d'interpolation linéaire entre les longueurs d'onde source et cible."""
    identity = np.eye(len(wvl_source))
    interp_func = scipy.interpolate.interp1d(
        wvl_source, identity, kind="linear", axis=0, fill_value="extrapolate"
    )
    return interp_func(wvl_target)

# Initialisation globale de la matrice d'interpolation
INTERP_MATRIX = build_interp_matrix(wvl_s2, WVL_PRS).astype(np.float32)


def genere_grille_before_after(scene, plot_dir):
    """
    Charge les données HSI et MSI (avant/après), effectue les simulations, 
    les interpolations, les normalisations, et génère les grilles de visualisations RGB.
    """
    print(f"\n--- Traitement de la scène : {scene} ---")
    
    hsi_after = DATA_DIR / f"{scene}" / f"{scene}-after-prs.nc"
    msi_after = DATA_DIR / f"{scene}" / f"{scene}-after-s2.nc"
    hsi_before = DATA_DIR / f"{scene}" / f"{scene}-before-prs.nc"
    msi_before = DATA_DIR / f"{scene}" / f"{scene}-before-s2.nc"

    # 1. Chargement
    with xr.open_dataset(hsi_after) as ds:
        data_hsi_after = np.nan_to_num(ds["sr"].values, nan=1000.0)
        if data_hsi_after.shape[-1] < data_hsi_after.shape[0]:
            data_hsi_after = np.transpose(data_hsi_after, (2, 0, 1))
    
    with xr.open_dataset(hsi_before) as ds:
        data_hsi_before = np.nan_to_num(ds["sr"].values, nan=1000.0)
        if data_hsi_before.shape[-1] < data_hsi_before.shape[0]:
            data_hsi_before = np.transpose(data_hsi_before, (2, 0, 1))

    with xr.open_dataset(msi_after) as ds:
        data_msi_after = np.nan_to_num(ds["sr"].values, nan=1000.0)
        if data_msi_after.shape[-1] < data_msi_after.shape[0]:
            data_msi_after = np.transpose(data_msi_after, (2, 0, 1))
    
    with xr.open_dataset(msi_before) as ds:
        data_msi_before = np.nan_to_num(ds["sr"].values, nan=1000.0)
        if data_msi_before.shape[-1] < data_msi_before.shape[0]:
            data_msi_before = np.transpose(data_msi_before, (2, 0, 1))
    
    c_hsi, h, w = data_hsi_after.shape
    c_msi, h, w = data_msi_after.shape

    # 2. Simulation MSI
    srf = np.load(DEFAULT_SRF_PATH)
    hsi_before_perm = data_hsi_before.transpose(1, 2, 0)
    hsi_before_flat = hsi_before_perm.reshape(-1, c_hsi)
    hsi_after_perm = data_hsi_after.transpose(1, 2, 0)
    hsi_after_flat = hsi_after_perm.reshape(-1, c_hsi)

    msi_sim_flat_before = np.dot(hsi_before_flat, srf)
    msi_sim_flat_after = np.dot(hsi_after_flat, srf)

    data_msi_sim_before = msi_sim_flat_before.reshape(h, w, c_msi).transpose(2, 0, 1)
    data_msi_sim_after = msi_sim_flat_after.reshape(h, w, c_msi).transpose(2, 0, 1)

    # 3. Normalisation MSI
    mean_sim_a = np.mean(data_msi_sim_after, axis=(1, 2), keepdims=True)
    std_sim_a = np.std(data_msi_sim_after, axis=(1, 2), keepdims=True)
    mean_a = np.mean(data_msi_after, axis=(1, 2), keepdims=True)
    std_a = np.std(data_msi_after, axis=(1, 2), keepdims=True)
    data_msi_after_norm = (data_msi_after - mean_a) / (std_a + 1e-8) * std_sim_a + mean_sim_a

    # 4. Génération RGB
    rgb_msi_after = get_rgb(data_msi_after, is_prisma=False, save_path=plot_dir / f"{scene}_after_msi.png", title="AFTER - MSI")
    rgb_hsi_after = get_rgb(data_hsi_after, is_prisma=True, save_path=plot_dir / f"{scene}_after_hsi.png", title="AFTER - HSI")
    rgb_msi_sim_after = get_rgb(data_msi_sim_after, is_prisma=False, save_path=plot_dir / f"{scene}_after_msi_sim.png", title="AFTER - MSI Simulé")
    rgb_msi_after_norm = get_rgb(data_msi_after_norm, is_prisma=False, save_path=plot_dir / f"{scene}_after_msi_norm.png", title="AFTER - MSI Normalisé")


# ==========================================
# BLOC PRINCIPAL (MAIN) - SAUVEGARDE CIBLÉE
# ==========================================
if __name__ == "__main__":
    print(f"Forme de la matrice d'interpolation : {INTERP_MATRIX.shape}")

    scene = "belgrade"
    plot_dir = RGB_DIR / "val"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Traitement spécifique de la scène : {scene} (AFTER) ---")
    
    hsi_after = DATA_DIR / scene / f"{scene}-after-prs.nc"
    msi_after = DATA_DIR / scene / f"{scene}-after-s2.nc"

    # Chargement
    with xr.open_dataset(hsi_after) as ds:
        data_hsi_after = np.nan_to_num(ds["sr"].values, nan=1000.0)
        if data_hsi_after.shape[-1] < data_hsi_after.shape[0]:
            data_hsi_after = np.transpose(data_hsi_after, (2, 0, 1))
            
    with xr.open_dataset(msi_after) as ds:
        data_msi_after = np.nan_to_num(ds["sr"].values, nan=1000.0)
        if data_msi_after.shape[-1] < data_msi_after.shape[0]:
            data_msi_after = np.transpose(data_msi_after, (2, 0, 1))
            
    c_hsi, h, w = data_hsi_after.shape
    c_msi, h, w = data_msi_after.shape

    # Simulation MSI (HSI x SRF)
    srf = np.load(DEFAULT_SRF_PATH)
    hsi_after_perm = data_hsi_after.transpose(1, 2, 0)
    hsi_after_flat = hsi_after_perm.reshape(-1, c_hsi)
    msi_sim_flat_after = np.dot(hsi_after_flat, srf)
    data_msi_sim_after = msi_sim_flat_after.reshape(h, w, c_msi).transpose(2, 0, 1)

    # Normalisation MSI
    mean_sim = np.mean(data_msi_sim_after, axis=(1, 2), keepdims=True)
    std_sim = np.std(data_msi_sim_after, axis=(1, 2), keepdims=True)
    mean = np.mean(data_msi_after, axis=(1, 2), keepdims=True)
    std = np.std(data_msi_after, axis=(1, 2), keepdims=True)
    data_msi_after_norm = (data_msi_after - mean) / (std + 1e-8) * std_sim + mean_sim

    # Récupération des tableaux RGB en mémoire (normalisés entre 0 et 1)
    # Récupération des tableaux RGB en mémoire
    rgb_msi = get_rgb(data_msi_after, is_prisma=False, return_only=True)
    rgb_hsi = get_rgb(data_hsi_after, is_prisma=True, return_only=True)
    rgb_msi_sim = get_rgb(data_msi_sim_after, is_prisma=False, return_only=True)
    rgb_msi_norm = get_rgb(data_msi_after_norm, is_prisma=False, return_only=True)

    # Inverser l'ordre des canaux (passe de RGB à BGR, ou inversement) et convertir en uint8 [0-255]
    rgb_msi = (np.clip(rgb_msi, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    rgb_hsi = (np.clip(rgb_hsi, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    rgb_msi_sim = (np.clip(rgb_msi_sim, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    rgb_msi_norm = (np.clip(rgb_msi_norm, 0, 1) * 255).astype(np.uint8)[..., ::-1]

    # Dossier d'export dédié
    dossier_export = plot_dir / "belgrade_export_after"
    dossier_export.mkdir(parents=True, exist_ok=True)

    # Sauvegarde avec save_data_with_ext
    save_data_with_ext(rgb_msi, dossier_export / "belgrade_after_msi", ext=".png", isrgb=True, m255=False)
    save_data_with_ext(rgb_hsi, dossier_export / "belgrade_after_hsi", ext=".png", isrgb=True, m255=False)
    save_data_with_ext(rgb_msi_norm, dossier_export / "belgrade_after_msi_norm", ext=".png", isrgb=True, m255=False)
    save_data_with_ext(rgb_msi_sim, dossier_export / "belgrade_after_msi_sim", ext=".png", isrgb=True, m255=False)

    # Dossier d'export dédié
    dossier_export = plot_dir / "belgrade_export_after"
    dossier_export.mkdir(parents=True, exist_ok=True)

    # Sauvegarde avec save_data_with_ext (laisse m255=False puisque nos données sont déjà en 0-255)
    save_data_with_ext(rgb_msi, dossier_export / "belgrade_after_msi", ext=".png", isrgb=True, m255=False)
    save_data_with_ext(rgb_hsi, dossier_export / "belgrade_after_hsi", ext=".png", isrgb=True, m255=False)
    save_data_with_ext(rgb_msi_norm, dossier_export / "belgrade_after_msi_norm", ext=".png", isrgb=True, m255=False)
    save_data_with_ext(rgb_msi_sim, dossier_export / "belgrade_after_msi_sim", ext=".png", isrgb=True, m255=False)

    
   

    print("✅ Sauvegarde ciblée (belgrade after) réalisée avec succès dans :", dossier_export)