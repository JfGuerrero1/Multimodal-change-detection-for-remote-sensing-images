
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import csv
from matplotlib.colors import LogNorm
from pathlib import Path
import pandas as pd
import xarray as xr
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import datetime
import numpy as np
from skimage.morphology import remove_small_objects
from src.constants import SRF_MATRIX

# --- Configuration des chemins ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'mumucd'
OUTPUT_DIR_DIAG = ROOT_DIR/'data'/'diag'
DATA_TIME = DATA_DIR / 'mumucd_v1_dates.txt'
CLOUD_CLASS_ID=8
WATER_CLASS_ID=0

WVL_PRS =  np.array([
    406.9934, 415.839 , 423.78476, 431.3347 , 438.6569 , 446.0147 , 453.38947, 460.73175, 468.09842, 475.31885,
    482.54816, 489.79486, 497.05865, 504.51172, 512.0464 , 519.54376, 527.3053 , 535.05255, 542.88513, 550.9146 ,
    559.02026, 567.2061 , 575.4868 , 583.8441 , 592.339  , 601.0144 , 609.9582 , 618.72   , 627.77844, 636.6763 ,
    645.9638 , 655.41876, 664.8941 , 674.46436, 684.13727, 694.12836, 703.737  , 713.72687, 723.87994, 733.9552 ,
    744.14954, 754.4696 , 764.85645, 775.2735 , 785.65955, 796.127  , 806.71106, 817.31104, 827.9195 , 838.5272 ,
    849.20996, 859.97314, 870.74255, 881.45605, 892.08093, 902.80164, 913.44507, 923.9502 , 934.11206, 944.6273 ,
    956.2715 , 967.0267 , 977.3654 , 979.224  , 988.9179 , 998.9082 , 1008.6443, 1018.5357, 1029.344  , 1037.9878,
    1047.675 , 1057.5737, 1067.7948, 1078.2161, 1088.761 , 1099.2776, 1109.8894, 1120.6759, 1131.3048, 1142.0703,
    1152.6501, 1163.676 , 1174.7142, 1185.5884, 1196.3394, 1207.2737, 1217.8635, 1229.1852, 1240.2145, 1250.9799,
    1262.5322, 1273.4963, 1284.4878, 1295.4218, 1306.218 , 1317.2566, 1328.2993, 1339.1294, 1349.7877, 1361.0531,
    1372.9117, 1383.2798, 1394.754 , 1405.6268, 1416.5374, 1427.3748, 1438.466 , 1449.1888, 1459.3157, 1469.9308,
    1480.8422, 1491.4292, 1502.0236, 1512.6333, 1523.2222, 1533.7764, 1544.2262, 1554.8168, 1565.3688, 1575.6274,
    1585.8597, 1596.2454, 1606.4913, 1616.8336, 1627.021 , 1637.0919, 1647.2316, 1656.933 , 1667.185 , 1677.3193,
    1687.4269, 1697.2943, 1707.0945, 1716.8589, 1726.6516, 1736.4883, 1746.2192, 1755.833 , 1765.5127, 1775.1178,
    1784.7173, 1793.9531, 1803.5902, 1813.0514, 1822.4413, 1832.0272, 1841.3256, 1850.5543, 1859.5587, 1868.1732,
    1878.7426, 1887.081 , 1896.0913, 1904.9347, 1914.3015, 1923.3857, 1932.2599, 1941.1107, 1949.9008, 1958.6244,
    1967.3418, 1976.013 , 1984.853 , 1993.5482, 2002.1106, 2010.6614, 2019.3214, 2027.7267, 2036.2607, 2044.6809,
    2053.0078, 2061.3787, 2069.7957, 2077.9915, 2086.3823, 2094.6252, 2102.8213, 2111.039 , 2119.2314, 2127.3372,
    2135.5103, 2143.4656, 2151.3862, 2159.564 , 2167.4849, 2175.3442, 2183.4202, 2191.1003, 2199.1353, 2206.843 ,
    2214.625 , 2222.4263, 2230.0076, 2237.904 , 2245.4485, 2253.1104, 2260.8665, 2268.2883, 2276.0537, 2283.4934,
    2290.8267, 2298.6094, 2305.7227, 2313.2007, 2320.8955, 2327.8242, 2335.5264, 2342.8228, 2349.7915, 2357.2937,
    2364.5945, 2371.5522, 2378.771 , 2386.0618, 2393.0388, 2400.036 , 2407.6045, 2414.3567, 2421.2373, 2428.6677,
    2435.5442, 2442.403 , 2449.1423, 2456.5857, 2463.0303, 2469.6272, 2477.055 , 2483.793 , 2490.2192, 2497.1155
])

WVL_S2= np.array([443., 490., 560., 665., 705., 740., 783., 842., 865., 940., 1610., 2190.])

def get_diff_for_scene(scene_name, date_df, type_key):
    """Calcule l'écart temporel pour une scène et un type (   / ) donné."""
    row = date_df[date_df['scene'] == scene_name]
    if row.empty: return 0
    
    # Choix des colonnes selon le type
    col_prs = f'PRS-{type_key}'
    col_s2 = f'S2-{type_key}'
    
    d1 = pd.to_datetime(str(row[col_prs].values[0]), format='%Y%m%d')
    d2 = pd.to_datetime(str(row[col_s2].values[0]), format='%Y%m%d')
    return abs((d1 - d2).days)

CLOUD_CLASS_ID=8
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


def tryptique_view(patch_msi,patch_hsi,patch_dw,output_path):
    rgb_msi=get_rgb_from_patch(patch_msi,is_prisma=False)
    rgb_hsi=get_rgb_from_patch(patch_hsi,is_prisma=True)
    DW_INFO = {
        0: {"name": "Water", "color": '#419BDF'},
        1: {"name": "Trees", "color": '#397D49'},
        2: {"name": "Grass", "color": '#88B053'},
        3: {"name": "Flooded vegetation", "color": '#7A87C6'},
        4: {"name": "Crops", "color": '#E49635'},
        5: {"name": "Shrub", "color": '#DFC35A'},
        6: {"name": "Built area", "color": '#C4281B'},
        7: {"name": "Bare ground", "color": '#A59B8F'},
        8: {"name": "Snow / Clouds", "color": '#B39FE1'}
    }

    dw_colors = [DW_INFO[k]["color"] for k in sorted(DW_INFO.keys())]
    cmap_dw = mcolors.ListedColormap(dw_colors)
    norm_dw = mcolors.BoundaryNorm(list(range(10)), cmap_dw.N)

    fig,axes=plt.subplots(1,3,figsize=(15,5))

    # --- Affichage MSI (Gauche) ---
    axes[0].imshow(rgb_msi)
    axes[0].set_title("Vue MSI (Sentinel-2 RGB)", fontsize=12)
    axes[0].axis('off')
    
    # --- Affichage HSI (Milieu) ---
    axes[1].imshow(rgb_hsi)
    axes[1].set_title("Vue Hyperspectrale (PRISMA RGB)", fontsize=12)
    axes[1].axis('off')
    
    # --- Affichage Dynamic World (Droite) ---
    axes[2].imshow(patch_dw, cmap=cmap_dw, norm=norm_dw, interpolation='nearest')
    axes[2].set_title("Classification Sémantique (DW)", fontsize=12)
    axes[2].axis('off')
    
    # 4. Génération de la légende personnalisée
    legend_handles = []
    for k in sorted(DW_INFO.keys()):
        # On crée un petit carré de couleur pour chaque classe
        patch = mpatches.Patch(color=DW_INFO[k]["color"], label=DW_INFO[k]["name"])
        legend_handles.append(patch)
    
    # On place la légende à droite du 3ème axe (Dynamic World)
    axes[2].legend(
        handles=legend_handles, 
        loc='center left', 
        bbox_to_anchor=(1.05, 0.5), # (1.05, 0.5) signifie juste à l'extérieur droit du carré
        borderaxespad=0.,
        fontsize=10,
        frameon=True
    )
    
    # 5. Sauvegarde
    plt.tight_layout() # Évite que la légende soit coupée au moment du save
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


import numpy as np
from skimage.morphology import remove_small_objects

def stat_dico(patch_hsi, patch_msi, patch_dw, patch_name, date_df,
              rupture_threshold=0.2, cloud_threshold=0.1, 
              spatial_threshold=0.5,
              bright_thresh=0.35, ratio_thresh=0.8,
              sim_rmse_threshold=0.15):
    """
    Calcule les métriques de qualité HSI/MSI avec simulation MSI physique 
    par projection spectrale (SRF_MATRIX).
    """
    # 1. Alignement initial des dimensions (C, H, W) pour les traitements spectraux/spatiaux
    if patch_hsi.shape[0] > patch_hsi.shape[2]: patch_hsi = np.moveaxis(patch_hsi, -1, 0)
    if patch_msi.shape[0] > patch_msi.shape[2]: patch_msi = np.moveaxis(patch_msi, -1, 0)

    eps = 1e-3

    # 2. Masquage des bandes d'absorption atmosphérique HSI
    mask_atm = ((WVL_PRS > 1300) & (WVL_PRS < 1500)) | ((WVL_PRS > 1800) & (WVL_PRS < 2000))
    no_atm_indices = np.where(~mask_atm)[0]
    patch_hsi_no_atm = patch_hsi[no_atm_indices, :, :]

    # 3. Gradients Spectraux HSI
    diff_no_atm = np.abs(np.diff(patch_hsi_no_atm, axis=0)) / (patch_hsi_no_atm[:-1, :, :] + eps)
    max_diff_no_atm = np.max(diff_no_atm, axis=0)

    diff_spectrale = np.abs(np.diff(patch_hsi, axis=0))
    max_diff = np.max(diff_spectrale, axis=0)

    # 4. Plage Visible & SWIR pour HSI
    patch_hsi_visible = patch_hsi[5:40, :, :]
    patch_hsi_swir = patch_hsi[51:100, :, :]

    diff_visible = np.abs(np.diff(patch_hsi_visible, axis=0)) / (patch_hsi_visible[:-1, :, :] + eps)
    max_diff_visible = np.max(diff_visible, axis=0)

    # Détection des lignes/colonnes mortes
    hsi_sum_c = np.sum(patch_hsi, axis=0)
    hsi_has_dead_row = 1 if np.any(np.all(hsi_sum_c == 0, axis=1)) else 0
    hsi_has_dead_col = 1 if np.any(np.all(hsi_sum_c == 0, axis=0)) else 0

    msi_sum_c = np.sum(patch_msi, axis=0)
    msi_has_dead_row = 1 if np.any(np.all(msi_sum_c == 0, axis=1)) else 0
    msi_has_dead_col = 1 if np.any(np.all(msi_sum_c == 0, axis=0)) else 0

    # 5. Gradient Spatial HSI
    grad_y, grad_x = np.gradient(patch_hsi, axis=(1, 2))
    spatial_gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_spatial_grad = np.max(spatial_gradient_magnitude, axis=0) 

    # 6. DÉTECTION SPECTRALE DES NUAGES / OMBRES / NEIGE / EAU (HSI)
    idx_blue_hsi  = np.argmin(np.abs(WVL_PRS - 490))
    idx_green_hsi = np.argmin(np.abs(WVL_PRS - 560))
    idx_red_hsi   = np.argmin(np.abs(WVL_PRS - 665))
    idx_nir_hsi   = np.argmin(np.abs(WVL_PRS - 842))
    idx_swir1_hsi = np.argmin(np.abs(WVL_PRS - 1690))

    blue_hsi  = patch_hsi[idx_blue_hsi]
    green_hsi = patch_hsi[idx_green_hsi]
    red_hsi   = patch_hsi[idx_red_hsi]
    nir_hsi   = patch_hsi[idx_nir_hsi]
    swir1_hsi = patch_hsi[idx_swir1_hsi]

    ndsi_hsi = (green_hsi - swir1_hsi) / (green_hsi + swir1_hsi + eps)
    is_snow_hsi = (ndsi_hsi > 0.42) & (nir_hsi > 0.11)

    mndwi_hsi = (green_hsi - swir1_hsi) / (green_hsi + swir1_hsi + eps)
    is_water_hsi = mndwi_hsi > 0.0

    visible_brightness_hsi = (red_hsi + green_hsi + blue_hsi) / 3.0
    is_bright_hsi = visible_brightness_hsi > bright_thresh

    cloud_ratio_hsi = green_hsi / (swir1_hsi + eps)
    has_cloud_sig_hsi = cloud_ratio_hsi > ratio_thresh

    cloud_mask_hsi = is_bright_hsi & has_cloud_sig_hsi & (~is_snow_hsi)
    shadow_mask_hsi = (nir_hsi < 0.1) & (visible_brightness_hsi < 0.12) & (~cloud_mask_hsi) & (~is_water_hsi)


    cloud_pct_hsi = float(np.mean(cloud_mask_hsi) * 100.0)
    shadow_pct_hsi = float(np.mean(shadow_mask_hsi) * 100.0)

    # 7. DÉTECTION SPECTRALE DES NUAGES / OMBRES / NEIGE / EAU (MSI - Sentinel-2)
    idx_blue_msi  = 0
    idx_green_msi = 1
    idx_red_msi   = 2
    idx_nir_msi   = 3
    idx_swir1_msi = 4 if patch_msi.shape[0] > 4 else 3

    blue_msi  = patch_msi[idx_blue_msi]
    green_msi = patch_msi[idx_green_msi]
    red_msi   = patch_msi[idx_red_msi]
    nir_msi   = patch_msi[idx_nir_msi]
    swir1_msi = patch_msi[idx_swir1_msi]

    ndsi_msi = (green_msi - swir1_msi) / (green_msi + swir1_msi + eps)
    is_snow_msi = (ndsi_msi > 0.42) & (nir_msi > 0.11)

    mndwi_msi = (green_msi - swir1_msi) / (green_msi + swir1_msi + eps)
    is_water_msi = mndwi_msi > 0.0

    visible_brightness_msi = (red_msi + green_msi + blue_msi) / 3.0
    is_bright_msi = visible_brightness_msi > bright_thresh

    cloud_ratio_msi = green_msi / (swir1_msi + eps)
    has_cloud_sig_msi = cloud_ratio_msi > ratio_thresh

    cloud_mask_msi = is_bright_msi & has_cloud_sig_msi & (~is_snow_msi)
    shadow_mask_msi = (nir_msi < 0.1) & (visible_brightness_msi < 0.12) & (~cloud_mask_msi) & (~is_water_msi)

    cloud_pct_msi = float(np.mean(cloud_mask_msi) * 100.0)
    shadow_pct_msi = float(np.mean(shadow_mask_msi) * 100.0)

    # 8. SIMULATION MSI VIA SRF_MATRIX (Projection HSI -> MSI)
    # Passage temporaire en format (H, W, C) pour la multiplication matricielle
    hsi_hwc = np.moveaxis(patch_hsi, 0, -1)
    h, w, c_hsi = hsi_hwc.shape
    hyper_2d = hsi_hwc.reshape(-1, c_hsi)
    
    # Simulation via SRF_MATRIX globale
    scene_multi_sim_hwc = np.dot(hyper_2d, SRF_MATRIX).reshape(h, w, -1).astype(np.float32)
    patch_msi_sim = np.moveaxis(scene_multi_sim_hwc, -1, 0) # Remise en (C, H, W)

    # Calcul du RMSE entre le vrai MSI et le MSI simulé
    min_bands = min(patch_msi.shape[0], patch_msi_sim.shape[0])
    msi_simulation_rmse = float(np.sqrt(np.mean((patch_msi[:min_bands] - patch_msi_sim[:min_bands])**2)))
    # Dans stat_dico, après la simulation :
    patch_msi_sim = np.moveaxis(scene_multi_sim_hwc, -1, 0)
    
    msi_simulation_rmse = float(np.sqrt(np.mean((patch_msi[:min_bands] - patch_msi_sim[:min_bands])**2)))
    msi_sim_mean = float(np.mean(patch_msi_sim)) # <-- AJOUT ICI

    # --- METRIQUES DICTIONNAIRES ---
    rupture_metrics = {
        "global_max": float(np.max(max_diff)),
        "global_mean": float(np.mean(max_diff)),
        "global_percentile_morts": float(np.mean(max_diff > rupture_threshold)),
        "no_atm_max": float(np.max(max_diff_no_atm)),
        "no_atm_mean": float(np.mean(max_diff_no_atm)),
        "no_atm_percentile_morts": float(np.mean(max_diff_no_atm > rupture_threshold)),
        "max_grad_spectral_high": float(np.max(max_diff_visible))
    }

    hsi_metrics = {
        "hsi_min": float(np.min(patch_hsi)),
        "hsi_max": float(np.max(patch_hsi)),
        "hsi_mean": float(np.mean(patch_hsi)),
        "hsi_std": float(np.std(patch_hsi)),
        "hsi_nb_zero": int(np.sum(patch_hsi == 0)),
        "hsi_zero_percentile": float(np.mean(patch_hsi == 0)),
        "hsi_zeros_visible": int(np.sum(patch_hsi_visible == 0)),
        "hsi_zeros_vis_percentile": float(np.mean(patch_hsi_visible == 0)),
        "hsi_zeros_swir": float(np.sum(patch_hsi_swir == 0)),
        "hsi_brillant_visible": float(np.sum(patch_hsi_visible > 0.9)),
        "hsi_brillant_vis_percentile": float(np.mean(patch_hsi_visible > 0.9)),
        "hsi_has_dead_row": hsi_has_dead_row,
        "hsi_has_dead_col": hsi_has_dead_col,
        "hsi_cloud_pct": cloud_pct_hsi,
        "hsi_shadow_pct": shadow_pct_hsi
    }

    msi_metrics = {
        "msi_min": float(np.min(patch_msi)),
        "msi_max": float(np.max(patch_msi)),
        "msi_mean": float(np.mean(patch_msi)),
        "msi_std": float(np.std(patch_msi)),
        "msi_mean_vis": float(np.mean(patch_msi[:3])), 
        "msi_zero": int(np.sum(patch_msi == 0)),
        "msi_zero_percentile": float(np.mean(patch_msi == 0)),
        "msi_brillant": float(np.sum(patch_msi > 0.9)),
        "msi_brillant_percentile": float(np.mean(patch_msi > 0.9)),
        "msi_has_dead_row": msi_has_dead_row,
        "msi_has_dead_col": msi_has_dead_col,
        "msi_cloud_pct": cloud_pct_msi,
        "msi_shadow_pct": shadow_pct_msi,
        "msi_simulation_rmse": msi_simulation_rmse,
        "msi_sim_mean": msi_sim_mean
    }

    dw_metrics = {
        "dw_dominant_class_id": int(np.bincount(patch_dw.ravel()).argmax()) if patch_dw.size > 0 else -1,
        "dw_percentile_cloud": float(np.mean(patch_dw == CLOUD_CLASS_ID)),
        "water_percentile": float(np.mean(patch_dw == WATER_CLASS_ID))
    }

    ratio = {
        "ratio_mean_vis_hsi_msi": float(np.mean(patch_hsi_visible) / (np.mean(patch_msi[:3]) + eps)),
        "ratio_std_vis_hsi_msi": float(np.std(patch_hsi_visible) / (np.std(patch_msi[:3]) + eps)),
    }

    spatial_metrics = {
        "spatial_grad_max": float(np.max(max_spatial_grad)),
        "spatial_grad_mean": float(np.mean(max_spatial_grad)),
        "spatial_rupture_percentile": float(np.mean(max_spatial_grad > spatial_threshold))
    }

    # --- DECISIONS & DIAGNOSTICS ---
    is_cloudy = (dw_metrics['dw_percentile_cloud'] > 5.0) or (cloud_pct_hsi > 5.0) or (cloud_pct_msi > 5.0)
    if is_cloudy:
        print(f"Patch {patch_name} is cloudy: DW cloud {dw_metrics['dw_percentile_cloud']:.2f}%, HSI cloud {cloud_pct_hsi:.2f}%, MSI cloud {cloud_pct_msi:.2f}%")
    is_spatially_dead = (hsi_metrics['hsi_has_dead_row'] > 0) or (hsi_metrics['hsi_has_dead_col'] > 0)   
    gradient_spatial_high = spatial_metrics['spatial_rupture_percentile'] > 0
    gradient_spectral_high = rupture_metrics['max_grad_spectral_high'] > 100
    is_inconsistent_sim = msi_simulation_rmse > sim_rmse_threshold
    
    is_aberrant = (gradient_spatial_high or is_cloudy or gradient_spectral_high or is_inconsistent_sim)

    diagnostics = {
        "is_cloudy": is_cloudy,
        "is_spatially_dead": is_spatially_dead,
        "is_inconsistent_sim": is_inconsistent_sim,
        "gradient_spatial_high": gradient_spatial_high,
        "is_aberrant_ratio": (ratio["ratio_mean_vis_hsi_msi"] > 1.2) or (ratio["ratio_mean_vis_hsi_msi"] < 0.2),
        "is_aberrant": is_aberrant,
    }

    scene_name, key = patch_name.rsplit('_patch_', 1)[0].rsplit('_', 1)
    time_diff = get_diff_for_scene(scene_name, date_df, key)

    return {
        "name": patch_name, 
        "time_diff": time_diff, 
        **rupture_metrics, 
        **hsi_metrics, 
        **msi_metrics, 
        **dw_metrics, 
        **diagnostics, 
        **ratio
    }

def trace_spectre(patch):
    # 1. On s'assure d'avoir le format (H, W, C) pour extraire les pixels proprement


    # Détection des pixels morts (où la réflectance vaut 0 sur les bandes 5 à 40)
    zeros_mask_visible = (patch[:, :, 5:40] == 0)
    # On cherche les coordonnées (y, x) où AU MOINS une bande est à 0 dans cette zone
    y_indices, x_indices = np.where(np.any(zeros_mask_visible, axis=2))
    
    if len(y_indices) == 0:
        print("Aucun pixel mort détecté dans la zone visible spécifiée.")
    
    y_indices=[i for i in range(450,400,-10)]
    x_indices=[i for i in range(180,230,10)]
    # Palette de couleurs distinctes pour faire correspondre la carte et le graphique
    colors = ['#FF1493', '#00FFFF', '#FFFF00', '#00FF00', '#FF4500'] # Rose, Cyan, Jaune, Vert, Orange

    # 2. Création de la figure avec 2 sous-graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.2]})
    
    
    ax_rgb = axes[0]
    im_rgb = get_rgb_from_patch(patch.transpose(2,0,1), is_prisma=True) # Adapte selon ta fonction
    ax_rgb.imshow(im_rgb)
    ax_rgb.set_title("Vue RGB & Localisation des Pixels ", fontsize=12, fontweight='bold')
    ax_rgb.axis('on') # 'on' permet de voir les coordonnées (y, x) sur les axes pour se repérer

    
    ax_spec = axes[1]
    bandes = WVL_PRS 

    # 3. Boucle pour tracer les spectres et les points correspondants
    num_pixels_to_show=len(y_indices)
    for i in range(num_pixels_to_show):
        y, x = y_indices[i], x_indices[i]
        spectre = patch[y, x, :]
        color = colors[i % len(colors)]
        
        # Sur la vue RGB, on place un marqueur coloré sur le pixel mort
        ax_rgb.scatter(x, y, color=color, edgecolors='black', s=100, marker='o', 
                       label=f"Pixel {i+1} (y={y}, x={x})")
        
        # Sur le graphique, on trace la courbe spectrale avec la MEME couleur
        ax_spec.plot(bandes, spectre, color=color, linestyle='-', linewidth=2,
                     label=f"Pixel {i+1} (y={y}, x={x})")

    # 4. Habillage et mise en forme du graphique spectral
    
    wvl_start, wvl_end = bandes[5], bandes[40]
    ax_spec.axvspan(wvl_start, wvl_end, color='gray', alpha=0.15, label="Zone Visible Analysée")

    ax_spec.set_title("Signatures Spectrales des Pixels Morts", fontsize=12, fontweight='bold')
    ax_spec.set_xlabel("Longueur d'onde (nm)", fontsize=11)
    ax_spec.set_ylabel("Réflectance", fontsize=11)
    ax_spec.set_ylim(-0.05, 1.05)
    ax_spec.grid(True, linestyle='--', alpha=0.5)
    
    # Gestion des légendes uniques pour les deux subplots
    ax_rgb.legend(loc='upper right', fontsize=9)
    ax_spec.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    
    # Sauvegarde et affichage
    output_plot = OUTPUT_DIR_DIAG / f"spectre_{scene}.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(output_plot)
    plt.show()
    plt.close() # Remplaçant de plt.clean() qui n'existe pas en matplotlib standard


def plot_patch_spatial_gradient(patch_hsi,scene):
    """
    Calcule et affiche le gradient spatial maximal d'un patch HSI de forme (C, H, W).
    """
    # 1. Gestion automatique du format si besoin (H, W, C) -> (C, H, W)
    if patch_hsi.shape[0] > patch_hsi.shape[2]: 
        patch_hsi = np.moveaxis(patch_hsi, -1, 0)

    # 2. Calcul du gradient spatial 
    grad_y, grad_x = np.gradient(patch_hsi, axis=(1, 2)) #(C)
    
    # 3. Norme du gradient spatial combinée, puis max sur toutes les bandes spectrales
    spatial_gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_spatial_grad = np.max(spatial_gradient_magnitude, axis=0) # Forme (H, W)

    # 4. Affichage de la carte de gradient spatial
    output_plot=OUTPUT_DIR_DIAG/f'gradient_spatial_{scene}'
    plt.figure(figsize=(8, 6))
    im = plt.imshow(max_spatial_grad, cmap='inferno')
    plt.colorbar(im, label="Intensité du gradient spatial")
    plt.title("Carte des ruptures spatiales (max sur les bandes)")
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    plt.grid(False)
    plt
    plt.savefig(output_plot,dpi=300)


def plot_patch_spectral_gradient_visible(patch_hsi,scene, use_relative=True, eps=1e-3):
    """
    Calcule et affiche le gradient spectral maximal d'un patch HSI 
    sur le domaine visible uniquement (< 800 nm).
    """
    # 1. Gestion du format (H, W, C) -> (C, H, W)
    if patch_hsi.shape[0] > patch_hsi.shape[2]: 
        patch_hsi = np.moveaxis(patch_hsi, -1, 0)

    # 2. Extraction du domaine visible (< 800 nm)
    vis_indices = np.where(WVL_PRS < 800)[0]
    patch_vis = patch_hsi[vis_indices, :, :]

    # 3. Calcul du gradient spectral (Relatif ou Absolu)
    diff_spectrale = np.abs(np.diff(patch_vis, axis=0)) # Forme (C_vis-1, H, W)

    if use_relative:
        denom = patch_vis[:-1, :, :] + eps
        gradient_3d = diff_spectrale / denom
        title_type = "relatif"
    else:
        gradient_3d = diff_spectrale
        title_type = "absolu"

    # 4. Max sur les canaux du visible -> carte 2D (H, W)
    max_spectral_grad = np.max(gradient_3d, axis=0)

    # 5. Affichage
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(max_spectral_grad, cmap='inferno')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Gradient spectral ({scene})")
    
    ax.set_title(f"Gradient Spectral ({title_type}) — Visible (< 800 nm)")
    ax.set_xlabel("X (Pixels)")
    ax.set_ylabel("Y (Pixels)")
    
    plt.tight_layout()

    # 6. Export et nettoyage
    output_plot = OUTPUT_DIR_DIAG /'gradient_spectral'/ f'gradient_spectral_visible_{scene}.png'
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.tight_layout()

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

def process_and_save_clouds(
    img,patch_name,scene,
    wavelengths=None, 
    bright_thresh=0.35, 
    ratio_thresh=0.8, 
    save_dir=OUTPUT_DIR_DIAG/'clouds2',

    
):
    """
    Fonction unique pour détecter et sauvegarder les nuages/ombres sur MSI ou HSI.
    
    img         : ndarray (H, W, C) normalisé [0.0, 1.0]
    wavelengths : list/array des longueurs d'onde en nm (Optionnel si MSI Sentinel-2)
    save_dir    : Dossier de destination pour les visuels et masques
    prefix      : Identifiant du patch (ex: 'patch_0012')
    """
    os.makedirs(save_dir, exist_ok=True)
    

    print(f'Traitement_{scene}')

    # Cas HSI : Recherche dynamique des longueurs d'onde les plus proches
    wl = np.array(wavelengths)
    idx_blue  = np.argmin(np.abs(wl - 490))
    idx_green = np.argmin(np.abs(wl - 560))
    idx_red   = np.argmin(np.abs(wl - 665))
    idx_nir   = np.argmin(np.abs(wl - 842))
    idx_swir1 = np.argmin(np.abs(wl -1690 ))  # SWIR1 idéal autour de 1600nm (pour NDSI)

 # 1. EXTRACTION DES BANDES & RGB
    blue  = img[:, :, idx_blue]
    green = img[:, :, idx_green]
    red   = img[:, :, idx_red]
    nir   = img[:, :, idx_nir]
    swir1 = img[:, :, idx_swir1]

    rgb = np.clip(np.stack([red, green, blue], axis=-1)*2.5 , 0, 1)


# NDSI : Dépiste la neige (Neige = NDSI fort | Nuage = NDSI faible)
    ndsi = (green - swir1) / (green + swir1 + 1e-6)
    is_snow = (ndsi > 0.42) & (nir > 0.11)

# 3. DÉTECTION DES NUAGES
    visible_brightness = (red + green + blue) / 3.0
    is_bright = visible_brightness > bright_thresh

    # MNDWI est bien plus robuste pour l'eau peu profonde / urbaine / portuaire
    mndwi = (green - swir1) / (green + swir1 + 1e-6)
    is_water = mndwi > 0.0  # Seuil souvent autour de 0.0



    cloud_ratio = green / (swir1 + 1e-6)
    has_cloud_sig = cloud_ratio > ratio_thresh

    cloud_mask = is_bright & has_cloud_sig & (~is_snow)
    shadow_mask = (nir < 0.1) & (visible_brightness < 0.12) & (~cloud_mask) & (~is_water)
    

    # 4. MÉTRIQUES ET METRIC D'ACCEPTATION
    cloud_pct = np.mean(cloud_mask) * 100.0
    shadow_pct = np.mean(shadow_mask) * 100.0

    # 5. VISUALISATION ET SAUVEGARDE
    rgb_overlay = rgb.copy()
    rgb_overlay[cloud_mask] = [1, 0, 0]      # Rouge = Nuage
    rgb_overlay[shadow_mask] = [0, 0.4, 1]   # Bleu = Ombre

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb);          axes[0].set_title(f"RGB - {patch_name}")
    axes[1].imshow(cloud_mask,  cmap='gray'); axes[1].set_title(f"Nuages ({cloud_pct:.1f}%)")
    axes[2].imshow(shadow_mask, cmap='gray'); axes[2].set_title(f"Ombres ({shadow_pct:.1f}%)")
    axes[3].imshow(rgb_overlay);  axes[3].set_title("Superposition (R=Nuage, B=Ombre)")
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{patch_name}_summary.png"), dpi=150)
    plt.close(fig) # Libération de la mémoire
    
    # Masque binaire PNG
    plt.imsave(os.path.join(save_dir, f"{patch_name}_cloud_mask.png"), cloud_mask, cmap='gray')

    return {
        "cloud_mask": cloud_mask,
        "shadow_mask": shadow_mask,
        "cloud_pct": cloud_pct,
        "shadow_pct": shadow_pct,
        "is_valid": (cloud_pct <= 5.0)   # Booléen direct pour garder/rejeter
    }

if __name__ == "__main__":
    """
    L_scene=[
    "aranjuez",
    "arborea",
    "baltijsk",
    "bari",
    "beheira",
    "beirut",
    "belgrade",
    "binh_dai",
    "brasilia",
    "camerino",
    "cape_town",
    "codigoro",
    "copenhagen",
    "copperton",
    "cukotka",
    "cullivel",
    "dellys",
    "dubai",
    "dublin",
    "elsalto",
    "eyjafjoll",
    "fontainebleau",
    "fukushima",
    "guantanamo",
    "hanging_rock",
    "istanbul",
    "jagersfontein",
    "java",
    "jordan",
    "kirtland",
    "kitami",
    "lagos",
    "london",
    "lorca",
    "los_angeles",
    "los_cabos",
    "malindi",
    "mantua",
    "mexico_city",
    "montevideo",
    "mosul",
    "mrirt",
    "muscat",
    "nagaoka",
    "new_york",
    "nicosia",
    "nouakchott",
    "novara",
    "palermo",
    "paris",
    "poinciana",
    "port_au_prince",
    "prague",
    "quito",
    "rome",
    "salinas",
    "sanaa",
    "shanghai",
    "spinazzola",
    "suez",
    "sydney",
    "taiwan",
    "tampa_bay",
    "tientsin",
    "tijuana",
    "tirana",
    "valencia",
    "yuen_long"]

    for scene in L_scene:
        
        scene_path=f'/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/mumucd/{scene}/{scene}-after-prs.nc'
        with  xr.open_dataset(scene_path) as ds:
            patch=ds["sr"].values
            patch=patch[:,:,:]
            process_and_save_clouds(patch,scene+"_hsi",WVL_PRS)
    """

    














