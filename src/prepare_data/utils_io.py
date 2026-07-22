
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import csv
from pathlib import Path
import pandas as pd
import xarray as xr

import datetime

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

def get_diff_for_scene(scene_name, date_df, type_key):
    """Calcule l'écart temporel pour une scène et un type (before/after) donné."""
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



def stat_dico(patch_hsi, patch_msi, patch_dw, patch_name,rupture_threshold=0.2,cloud_threshold=0.1,sature_threshold=0.1,spatial_threshold=0.5):
    """
    Calcule les métriques de qualité HSI/MSI avec gestion automatique du format d'entrée.
    Intègre une détection de pixel mort par différence de longueurs d'onde (gradient spectral).
    """


   
    if patch_hsi.shape[0] > patch_hsi.shape[2]: patch_hsi = np.moveaxis(patch_hsi, -1, 0)
    if patch_msi.shape[0] > patch_msi.shape[2]: patch_msi = np.moveaxis(patch_msi, -1, 0)

    mask_atm=((WVL_PRS > 1300) & (WVL_PRS < 1500)) | ((WVL_PRS > 1800) & (WVL_PRS < 2000))
    no_atm_indices = np.where(~mask_atm)[0]
    patch_hsi_no_atm = patch_hsi[no_atm_indices, :, :]

    diff_no_atm = np.abs(np.diff(patch_hsi_no_atm, axis=0))
    max_diff_no_atm = np.max(diff_no_atm, axis=0)


    rupture_threshold=float(rupture_threshold)
    cloud_threshold=float(cloud_threshold)
    sature_threshold=float(sature_threshold)
    #Gradient
    diff_spectrale = np.abs(np.diff(patch_hsi, axis=0))
    max_diff = np.max(diff_spectrale, axis=0)
    
    # Plage Visible (5:35)
    vis_hsi = patch_hsi[5:40, :, :]
    diff_vis = np.abs(np.diff(vis_hsi, axis=0))
    max_diff_vis = np.max(diff_vis, axis=0)
    patch_hsi_visible=patch_hsi[5:40,:,:]
    patch_hsi_swir=patch_hsi[51:100,:,:]

    #sans_atm
   

    pixels_with_zero_mask = np.any(vis_hsi == 0, axis=0) # Forme : (H, W) -> matrice de booléens
    diff_vis = np.abs(np.diff(vis_hsi, axis=0)) # Forme : (C_vis - 1, H, W)
    max_diff_vis = np.max(diff_vis, axis=0)     # Forme : (H, W)
    gradients_of_zero_pixels = max_diff_vis[pixels_with_zero_mask]
    if gradients_of_zero_pixels.size > 0:
        vis_zero_max = float(np.max(gradients_of_zero_pixels))
        vis_zero_mean = float(np.mean(gradients_of_zero_pixels))
    else:
        vis_zero_max = 0.0
        vis_zero_mean = 0.0
    


    grad_y, grad_x = np.gradient(patch_hsi, axis=(1, 2))

    spatial_gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2) # Forme (C, H, W)

    max_spatial_grad = np.max(spatial_gradient_magnitude, axis=0) # Forme (H, W)

    rupture_metrics = {
    "global_max": float(np.max(max_diff)),
    "global_mean": float(np.mean(max_diff)),
    "global_percentile_morts": float(np.sum(max_diff > rupture_threshold) / max_diff.size),
    "vis_max": float(np.max(max_diff_vis)),
    "vis_mean": float(np.mean(max_diff_vis)),
    "vis_percentile_morts": float(np.sum(max_diff_vis > rupture_threshold) / max_diff_vis.size),
    "vis_zero_max": vis_zero_max,
    "vis_zero_mean": vis_zero_mean,
    "no_atm_max": float(np.max(max_diff_no_atm)),
    "no_atm_mean": float(np.mean(max_diff_no_atm)),
    "no_atm_percentile_morts": float(np.sum(max_diff_no_atm > rupture_threshold) / max_diff_no_atm.size)
    }
    hsi_metrics = {
        "hsi_min": float(np.min(patch_hsi)),
        "hsi_max": float(np.max(patch_hsi)),
        "hsi_mean": float(np.mean(patch_hsi)),
        "hsi_std": float(np.std(patch_hsi)),
        "hsi_nb_zero": int(np.sum(patch_hsi == 0)),
        "hsi_zero_percentile": float(np.sum(patch_hsi == 0) / patch_hsi.size),
        "hsi_zeros_visible": int(np.sum((patch_hsi_visible == 0))),
        "hsi_zero_percentile": float(np.sum(patch_hsi == 0) / patch_hsi.size),
        "hsi_zeros_vis_percentile": float(np.sum(patch_hsi_visible == 0) / patch_hsi_visible.size),
        "hsi_zeros_swir": float(np.sum((patch_hsi_swir == 0))),
        "hsi_brillant_visible": float(np.sum((patch_hsi_visible > 0.9))),
        "hsi_brillant_vis_percentile": float(np.sum((patch_hsi_visible > 0.9)) / patch_hsi_visible.size),
        "hsi_has_dead_row": 1 if np.sum(np.all(np.sum(patch_hsi == 0, axis=0) == patch_hsi.shape[0], axis=1)) > 0 else 0,
        "hsi_has_dead_col": 1 if np.sum(np.all(np.sum(patch_hsi == 0, axis=0) == patch_hsi.shape[0], axis=0)) > 0 else 0
    }

    msi_metrics = {
        "msi_min": float(np.min(patch_msi)),
        "msi_max": float(np.max(patch_msi)),
        "msi_mean": float(np.mean(patch_msi)),
        "msi_std": float(np.std(patch_msi)),
        "msi_mean_vis": float(np.std(patch_msi[:])),
        "msi_zero":float(np.sum(patch_msi == 0)),
        "msi_zero_percentile": float(np.sum(patch_msi == 0) / patch_msi.size),
        "msi_brillant":float(np.sum(patch_msi > 0.9)),
        "msi_brillant_percentile": float(np.sum(patch_msi > 0.9) / patch_msi.size),
        "msi_has_dead_row": 1 if np.sum(np.all(np.sum(patch_msi == 0, axis=0) == patch_msi.shape[0], axis=1)) > 0 else 0,
        "msi_has_dead_col": 1 if np.sum(np.all(np.sum(patch_msi == 0, axis=0) == patch_msi.shape[0], axis=0)) > 0 else 0
    }

    values, counts = np.unique(patch_dw, return_counts=True)
    dw_metrics = {
        "dw_dominant_class_id": int(values[np.argmax(counts)]) if len(values) > 0 else -1,
        "dw_percentile_cloud": float(np.sum(patch_dw == CLOUD_CLASS_ID) / patch_dw.size),
        "water_percentile": float(np.sum(patch_dw == WATER_CLASS_ID) / patch_dw.size)
    }

    ratio = {
        "ratio_mean_vis_hsi_msi": float(np.mean(patch_hsi_visible) / (np.mean(patch_msi) + 1e-6)),
        "ratio_std_vis_hsi_msi": float(np.std(patch_hsi_visible) / (np.std(patch_msi) + 1e-6)),
        
    }
    spatial_metrics = {
    "spatial_grad_max": float(np.max(max_spatial_grad)),
    "spatial_grad_mean": float(np.mean(max_spatial_grad)),
    # Pourcentage de pixels qui dépassent un certain seuil de rupture spatiale
    "spatial_rupture_percentile": float(np.sum(max_spatial_grad > spatial_threshold) / max_spatial_grad.size)
    }

    
    is_cloudy = dw_metrics['dw_percentile_cloud'] > cloud_threshold
    
    is_spectrally_dead = rupture_metrics['vis_percentile_morts'] > rupture_threshold 
    is_spatially_dead = (hsi_metrics['hsi_has_dead_row'] > 0) or (hsi_metrics['hsi_has_dead_col'] > 0)   
    is_saturated = hsi_metrics['hsi_brillant_vis_percentile'] > sature_threshold 
    gradient_spatial_high=spatial_metrics['spatial_rupture_percentile']>0
    
    is_aberrant = is_cloudy or is_spatially_dead or is_saturated or is_spectrally_dead or gradient_spatial_high

    diagnostics = {
        "is_cloudy": is_cloudy,
        "is_spectrally_dead": is_spectrally_dead,
        "is_spatially_dead": is_spatially_dead,
        "is_saturated": is_saturated,
        "is_aberrant": is_aberrant,
        "gradient_spatial_high":gradient_spatial_high
    }
    date_df=pd.read_csv(DATA_TIME)
    scene_name, key = patch_name.rsplit('_patch_', 1)[0].rsplit('_', 1)
    time_diff=get_diff_for_scene(scene_name,date_df,key)

    return {"name":patch_name,"time_diff":time_diff,**rupture_metrics, **hsi_metrics, **msi_metrics, **dw_metrics, **diagnostics, **ratio}
  

def trace_spectre(patch):
    # 1. On s'assure d'avoir le format (H, W, C) pour extraire les pixels proprement


    # Détection des pixels morts (où la réflectance vaut 0 sur les bandes 5 à 40)
    zeros_mask_visible = (patch[:, :, 5:40] == 0)
    # On cherche les coordonnées (y, x) où AU MOINS une bande est à 0 dans cette zone
    y_indices, x_indices = np.where(np.any(zeros_mask_visible, axis=2))
    
    if len(y_indices) == 0:
        print("Aucun pixel mort détecté dans la zone visible spécifiée.")
        

    # Nombre de pixels à afficher (on limite à 5 pour que le graphique reste lisible)
    
    y_indices=[300 for i in range(580,700,20)]
    x_indices=[i for i in range(580,700,20)]
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
    output_plot = OUTPUT_DIR_DIAG / "spectre_java_foret.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(output_plot)
    plt.show()
    plt.close() # Remplaçant de plt.clean() qui n'existe pas en matplotlib standard


def plot_patch_spatial_gradient(patch_hsi):
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
    output_plot=OUTPUT_DIR_DIAG/'gradient_spatial_java'
    plt.figure(figsize=(8, 6))
    im = plt.imshow(max_spatial_grad, cmap='inferno')
    plt.colorbar(im, label="Intensité du gradient spatial")
    plt.title("Carte des ruptures spatiales (max sur les bandes)")
    plt.xlabel("X (Pixels)")
    plt.ylabel("Y (Pixels)")
    plt.grid(False)
    plt
    plt.savefig(output_plot,dpi=300)



if __name__ == "__main__":
    java_path='/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/mumucd/java/java-after-prs.nc'
    with  xr.open_dataset(java_path) as ds:
        patch=ds["sr"].values
        patch=patch[:,:,:]
        print(patch.shape)
        trace_spectre(patch)
        plot_patch_spatial_gradient(patch)

    














