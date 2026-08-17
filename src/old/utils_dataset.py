import os
import glob
import random
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from einops import rearrange
from tqdm import tqdm
from pathlib import Path
import scipy
import matplotlib.pyplot as plt 
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import csv


CURRENT_FILE = Path(__file__).resolve()
# utils_dataset.py is in "project/src/", .parent.parent links to "project/"
ROOT_DIR = CURRENT_FILE.parent.parent
DATA_DIR = ROOT_DIR / 'data' /'dataset'
CACHE_DIR = DATA_DIR / 'patches_caches'
TRAIN_DIR=CACHE_DIR/'train'
TEST_DIR=CACHE_DIR/'test'
VAL_DIR=CACHE_DIR/'val'
DEFAULT_SRF_PATH = DATA_DIR / 'srf_matrix_norm_s2b.npy'


def build_interp_matrix(wvl_source, wvl_target):
    identity = np.eye(len(wvl_source))
    interp_func = scipy.interpolate.interp1d(
        wvl_source, identity, kind="linear", axis=0, fill_value="extrapolate"
    )
    return interp_func(wvl_target)

WVL_S2 = np.array([443., 490., 560., 665., 705., 740., 783., 842., 865., 940., 1610., 2190.])
WVL_PRS = np.array([
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
mask= ((WVL_PRS< 1350) | ((WVL_PRS> 1500) & (WVL_PRS < 1800)) | ((WVL_PRS > 2000) ))

INTERP_MATRIX = build_interp_matrix(WVL_S2, WVL_PRS) 
INTERP_MATRIX = INTERP_MATRIX.astype(np.float32) #C_hsi, C_msi
#MSI -> MSI_interpoled
SRF_MATRIX=np.load(DEFAULT_SRF_PATH)
#For HSI-> MSI_simuled


def get_rgb_from_patch(patch,is_prisma):

    patch=np.nan_to_num(patch, nan=1000.0)
    if is_prisma:
        indices=[32,15,5]
    else:
        indices=[3,2,0]
    
    r, g, b = patch[indices[0]], patch[indices[1]], patch[indices[2]]
    
    if not is_prisma: # Si c'est du MSI réel à 12 bandes
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





def get_stat_from_patch(patch, patch_name):
    #Numpy C,H,W

    min_patch=np.min(patch)
    max_patch=np.max(patch)
    mean_patch=np.mean(patch)
    std_patch=np.std(patch)

    zeros_mask=(patch==0)
    nb_zero=np.sum(zeros_mask)
    nb_zero_per_channel=np.sum(zeros_mask, axis=(1,2)) #C
    nb_zero_per_row=np.sum(zeros_mask, axis=(0,2) ) #H
    nb_zero_per_col=np.sum(zeros_mask, axis=(0,1) ) #W

    min_per_channel=np.min(patch,axis=(1,2))
    max_per_channel=np.max(patch,axis=(1,2))
    mean_per_channel=np.mean(patch,axis=(1,2))
    std_per_channel=np.std(patch,axis=(1,2))

    patch_metric = {
        "name": patch_name,

        
        "min_patch": float(min_patch),
        "max_patch": float(max_patch),
        "mean_patch": float(mean_patch),
        "std_patch": float(std_patch),
        "nb_zero": int(nb_zero),
        
        "nb_zero_per_channel": [int(x) for x in nb_zero_per_channel],
        "nb_zero_per_row": [int(x) for x in nb_zero_per_row],
        "nb_zero_per_col": [int(x) for x in nb_zero_per_col],
        "min_per_channel": [float(x) for x in min_per_channel],
        "max_per_channel": [float(x) for x in max_per_channel],
        "mean_per_channel": [float(x) for x in mean_per_channel],
        "std_per_channel": [float(x) for x in std_per_channel],
    }

    return patch_metric


import numpy as np
def stat_csv(patch_hsi, patch_msi, patch_dw, patch_name, preview_dir):
    """
    Calcule les métriques de qualité HSI/MSI, extrait le mode sémantique de DW,
    et enregistre un fichier CSV unique '{patch_name}_metrics.csv' dans le dossier cible.
    """
    # HSI
    min_hsi, max_hsi = float(np.min(patch_hsi)), float(np.max(patch_hsi))
    mean_hsi, std_hsi = float(np.mean(patch_hsi)), float(np.std(patch_hsi))

    zeros_mask_hsi = (patch_hsi == 0)
    nb_zero_hsi = int(np.sum(zeros_mask_hsi))
    nb_zeros_visible_hsi = int(np.sum(zeros_mask_hsi[5:35, :, :])) 
    nb_zeros_swir_hsi = int(np.sum(zeros_mask_hsi[51:100, :, :]))
    
    brillant_mask_hsi = (patch_hsi > 0.8)
    nb_brillant_hsi = int(np.sum(brillant_mask_hsi))
    nb_brillant_visible_hsi = int(np.sum(brillant_mask_hsi[5:35, :, :]))
    
    zero_percentile_hsi = float(nb_zero_hsi / patch_hsi.size)
    zero_visible_percentile_hsi = float(nb_zeros_visible_hsi / patch_hsi[5:35, :, :].size)
    brillant_visible_percentile_hsi = float(nb_brillant_visible_hsi / patch_hsi[5:35, :, :].size)

    # Détection des lignes et colonnes mortes (Capteur HSI)
    dead_rows_per_pixel = np.any(np.sum(zeros_mask_hsi, axis=2) == patch_hsi.shape[2], axis=0)
    dead_cols_per_pixel = np.any(np.sum(zeros_mask_hsi, axis=1) == patch_hsi.shape[1], axis=0)

    dead_rows_count = int(np.sum(dead_rows_per_pixel))
    dead_cols_count = int(np.sum(dead_cols_per_pixel))
    has_dead_row = 1 if dead_rows_count > 0 else 0
    has_dead_col = 1 if dead_cols_count > 0 else 0

    #MSI
    min_msi, max_msi = float(np.min(patch_msi)), float(np.max(patch_msi))
    mean_msi, std_msi = float(np.mean(patch_msi)), float(np.std(patch_msi))

    zeros_mask_msi = (patch_msi == 0)
    nb_zero_msi = int(np.sum(zeros_mask_msi))
    brillant_mask_msi = (patch_msi > 0.8)
    nb_brillant_msi = int(np.sum(brillant_mask_msi))
    
    zero_percentile_msi = float(nb_zero_msi / patch_msi.size)
    brillant_percentile_msi = float(nb_brillant_msi / patch_msi.size)

    # DW
    values, counts = np.unique(patch_dw, return_counts=True)
    dw_dominant_class_id = int(values[np.argmax(counts)])
    percentile_of_cloud_dw = float(np.sum(patch_dw == 8) / patch_dw.size)

    # --- 4. PACAKGING DES DONNÉES ---
    row_data = {
        "name": patch_name,
        # Métriques HSI
        "min_hsi": min_hsi, "max_hsi": max_hsi, "mean_hsi": mean_hsi, "std_hsi": std_hsi,
        "nb_zero_hsi": nb_zero_hsi, "zeros_in_visible_hsi": nb_zeros_visible_hsi, "zeros_in_swir_hsi": nb_zeros_swir_hsi,
        "zero_percentile_hsi": zero_percentile_hsi, "zero_visible_percentile_hsi": zero_visible_percentile_hsi,
        "nb_brillant_hsi": nb_brillant_hsi, "brillant_visible_percentile_hsi": brillant_visible_percentile_hsi,
        "has_dead_row_hsi": has_dead_row, "has_dead_col_hsi": has_dead_col, 
        "dead_rows_count_hsi": dead_rows_count, "dead_cols_count_hsi": dead_cols_count,
        # Métriques MSI
        "min_msi": min_msi, "max_msi": max_msi, "mean_msi": mean_msi, "std_msi": std_msi,
        "zero_percentile_msi": zero_percentile_msi, "brillant_percentile_msi": brillant_percentile_msi,
        # Métriques DW
        "dw_dominant_class_id": dw_dominant_class_id, "percentile_of_cloud_dw": percentile_of_cloud_dw
    }

    # --- 5. ÉCRITURE DU CSV INDIVIDUEL ---
    target_dir = Path(preview_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    patch_csv_path = target_dir / f"{patch_name}_metrics.csv"
    
    headers = list(row_data.keys())

    with open(patch_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row_data)












    
def prepare_dataset_offline(scene_pairs, split_name, patch_size=256, use_simulated_msi=False):


    mode_suffix = "simulated" if use_simulated_msi else "real"
    split_cache_dir = CACHE_DIR / mode_suffix / split_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    sim_str = 'sim' if use_simulated_msi else 'no_sim'
    
    if use_simulated_msi:
        c_multi = SRF_MATRIX.shape[1] #12
    
    c_hsi = INTERP_MATRIX.shape[0]  # 230

    all_stats_msi=[]
    all_stats_hsi=[]
    for x_file, y_file in tqdm(scene_pairs, desc=f"Patching {split_name}"):
        base_name = y_file.name.replace('-prs.nc', '')
        scene_name=base_name.replace('-after','')
        print(scene_name)
        print(base_name)
        print(DATA_DIR/scene_name)
        
        x_path = split_cache_dir / f'{base_name}_X_patches.npy'
        y_path = split_cache_dir / f'{base_name}_y_patches.npy'
        dw_path=DATA_DIR/scene_name/f'{base_name}-dw.nc'
        interp_path = split_cache_dir / f'{base_name}_interp_patches.npy'  

        #if x_path.exists() and y_path.exists() and interp_path.exists():  
            #continue

            
        with xr.open_dataset(x_file) as ds_x, xr.open_dataset(y_file) as ds_y,xr.open_dataset(dw_path)as ds_dw:
            h, w, c_hyper = ds_y["sr"].shape
            map=ds_dw["lcc"]
            print(map.shape)
            if not use_simulated_msi:
                c_multi = ds_x["sr"].shape[2] #12
            
            h_crop = h - (h % patch_size)
            w_crop = w - (w % patch_size)
            total_patches = (h_crop // patch_size) * (w_crop // patch_size)
            
            X_mmap = np.lib.format.open_memmap(str(x_path), mode='w+', dtype=np.float32, shape=(total_patches, c_multi, patch_size, patch_size))
            y_mmap = np.lib.format.open_memmap(str(y_path), mode='w+', dtype=np.float32, shape=(total_patches, c_hyper, patch_size, patch_size))
            interp_mmap = np.lib.format.open_memmap(str(interp_path), mode='w+', dtype=np.float32, shape=(total_patches, c_hsi, patch_size, patch_size))

            scene_preview_dir = split_cache_dir / f"{base_name}_previews"
            scene_preview_dir.mkdir(parents=True, exist_ok=True)  
            
            patch_idx = 0
            for i in range(h_crop // patch_size):
                for j in range(w_crop // patch_size):
                    r = i * patch_size
                    c = j * patch_size
                    
                    patch_hyper = ds_y["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32) #h,w,c
                    patch_dw=map[r:r+patch_size,c:c+patch_size]

                    if use_simulated_msi:
                        hyper_2d = patch_hyper.reshape(-1, c_hyper) #h*w,c_hsi
                        patch_multi = np.dot(hyper_2d, SRF_MATRIX).reshape(patch_size, patch_size, c_multi).astype(np.float32)
                        #                    h*w,c_hsi * c_hsi,c_msi
                        #finally, h,w,c_multi
                    else:
                        patch_multi = ds_x["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32)
                    
                    patch_multi_chw = np.transpose(patch_multi, (2, 0, 1))  # [C_msi, H, W]
                    X_mmap[patch_idx] = patch_multi_chw
                    patch_hyper_chw= np.transpose(patch_hyper, (2, 0, 1)) # [C_hsi,H,W]
                    y_mmap[patch_idx] = patch_hyper_chw
              
                    interp_numpy = INTERP_MATRIX @ patch_multi_chw.reshape(c_multi, -1)  # [C_hsi, H*W]
                    #              C_HSI,C_MSI @ C_MSI, H*W
                
                    
                    interp_mmap[patch_idx] = interp_numpy.reshape(c_hsi, patch_size, patch_size) #[C_HSI_,H,W]

                   

                    #affichage, MSI,HSI,DW
                    
                    name_id = f"{base_name}_patch_{patch_idx:04d}_{sim_str}"

                    #tryptique_view(patch_multi_chw,patch_hyper_chw,patch_dw,scene_preview_dir/f'{name_id}_tryptique.png')

                    #fichier json
                    metrics = get_stat_from_patch(patch_multi_chw, name_id)
                    all_stats_msi.append(metrics)
                    metrics = get_stat_from_patch(patch_hyper_chw, name_id)
                    all_stats_hsi.append(metrics)

                    stat_csv(patch_hyper_chw,patch_multi_chw,patch_dw,name_id,scene_preview_dir)



                    patch_idx += 1


    json_path = split_cache_dir / f"dataset_health_metrics_msi_{sim_str}.json"
    with open(json_path, "w") as f:
        json.dump(all_stats_msi, f, indent=4)
    print(f" Bilan statistique global sauvegardé dans : {json_path}")

    json_path = split_cache_dir / f"dataset_health_metrics_hsi_{sim_str}.json"
    with open(json_path, "w") as f:
        json.dump(all_stats_msi, f, indent=4)
    print(f" Bilan statistique global sauvegardé dans : {json_path}")
            
            
          




class SpectralDataset(Dataset):
    def __init__(self, dataset_dir, augment=False,is_residual=False, keep_atm_wave=True,is_normalised=False):
        """
        cache_dir: pointera vers CACHE_DIR/'simulated' ou 'real'/'train' ou 'test' ou 'val'
        donc pas besoin d'utiliser d'argument simulé ou non 

        augment: True pour activer les flips/rotations (uniquement pour le train en général)
        """
        #############################################################################################################
        self.cache_dir = Path(dataset_dir)
        self.augment = augment
        self.is_residual=is_residual
        self.keep_atm_wave=keep_atm_wave
        self.is_normalised=is_normalised
        self.srf_matrix=np.load(DEFAULT_SRF_PATH)


        if keep_atm_wave:
            self.indices_to_use = slice(None)  
        else:
            self.indices_to_use = np.where(mask)[0]

        print("Le script utilise les données dans ")
        print(self.cache_dir)

        self.x_files = sorted(glob.glob(str(self.cache_dir / "*X_patches.npy")))
        self.y_files = sorted(glob.glob(str(self.cache_dir / "*y_patches.npy")))
        
        assert len(self.x_files) > 0, f"No patches."
        assert len(self.x_files) == len(self.y_files), "Not same number of x and y files"
        
        self.index_map = []
        self.mmap_x_arrays = []
        self.mmap_y_arrays = []
        self.mmap_interp_arrays=[]
        
        for file_idx, (x_path, y_path) in enumerate(zip(self.x_files, self.y_files)):
           
            x_mmap = np.load(x_path, mmap_mode='r')
            y_mmap = np.load(y_path, mmap_mode='r')

            if self.is_residual:
                interp_path = str(x_path).replace('_X_patches.npy', '_interp_patches.npy')
                interp_mmap = np.load(interp_path, mmap_mode='r')
                self.mmap_interp_arrays.append(interp_mmap)
            
            self.mmap_x_arrays.append(x_mmap)
            self.mmap_y_arrays.append(y_mmap)
            num_patches = x_mmap.shape[0]
                  
            for patch_idx in range(num_patches):
                self.index_map.append((file_idx, patch_idx))
                
        print(f"[{self.cache_dir.parent.name.upper()} - {self.cache_dir.name.upper()}] load {len(self.index_map)} patches from {len(self.x_files)} scenes.")

    def __len__(self):
        return len(self.index_map)
    
    def augment_color_jittering(self,x):
        

        return x
    

    def augment_pair(self, x, y):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])

        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])

        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[1,2])
        y = torch.rot90(y, k, dims=[1,2])

        return x, y

    def augment_triplet(self, x, x_interp, y):
        if random.random() < 0.5:
            x, x_interp, y = (
                torch.flip(x, dims=[2]),
                torch.flip(x_interp, dims=[2]),
                torch.flip(y, dims=[2]),
            )
        if random.random() < 0.5:
            x, x_interp, y = (
                torch.flip(x, dims=[1]),
                torch.flip(x_interp, dims=[1]),
                torch.flip(y, dims=[1]),
            )

        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=[1, 2])
        x_interp = torch.rot90(x_interp, k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[1, 2])
        return x, x_interp, y
    
    def __getitem__(self, idx):

        
        # file with patch
        file_idx, patch_idx = self.index_map[idx]
        
        # copy
        x_patch = self.mmap_x_arrays[file_idx][patch_idx].copy()
        y_patch = self.mmap_y_arrays[file_idx][patch_idx].copy()
        c_msi,h,w=x_patch.shape
        c_hsi,h,w=y_patch.shape
        if self.is_normalised:
            y_patch_flat=y_patch.reshape(-1,c_hsi) #h*w,c_hsi
            x_sim=np.dot(y_patch_flat,SRF_MATRIX) #h*w,c_msi
            x_sim=x_sim.reshape(c_msi,h,w)

            mean_sim = np.mean(x_sim, axis=(1, 2), keepdims=True)
            std_sim = np.std(x_sim, axis=(1, 2), keepdims=True)

            mean = np.mean(x_patch, axis=(1, 2), keepdims=True)
            std=np.std(x_patch,axis=(1,2),keepdims=True)
            x_patch=(x_patch-mean)/(std+1e-8)*std_sim+mean_sim


        x = torch.from_numpy(x_patch)
        y = torch.from_numpy(y_patch)
    
        y=y[self.indices_to_use, :,:]

        if self.is_residual:

            #interpolation
            x_interp = (self.mmap_interp_arrays[file_idx][patch_idx].copy())
            x_interp = x_interp[self.indices_to_use, :, :]

            x_interp = torch.from_numpy(x_interp).float()

            if self.augment:
                x, x_interp, y = self.augment_triplet(x, x_interp, y)
            return x,x_interp,y
       
        else:
           
            if self.augment:               
                x, y = self.augment_pair(x,  y)
            return x,y
##################################################################################################################

 ################################################################################################################   
class SpectralDataset(Dataset):
    def __init__(self, cache_dir, augment=False):
        self.augment = augment

        self.msi_files = sorted(glob.glob(str(Path(cache_dir) / "*_MSI_simulated.npy")))
        self.hsi_true_files = sorted(glob.glob(str(Path(cache_dir) / "*_HSI_true.npy")))
        self.hsi_sim_files = sorted(glob.glob(str(Path(cache_dir) / "*_HSI_simulated.npy")))
        
        assert len(self.msi_files) == len(self.hsi_true_files) == len(self.hsi_sim_files), "Mismatch between MSI, HSI true and HSI simulated files."
        
        self.index_map = []
        self.mmap_msi = []
        self.mmap_hsi_true = []
        self.mmap_hsi_sim = []
        
        for f_idx, (f_msi, f_hsi_t, f_hsi_s) in enumerate(zip(self.msi_files, self.hsi_true_files, self.hsi_sim_files)):
            msi = np.load(f_msi, mmap_mode='r')
            hsi_t = np.load(f_hsi_t, mmap_mode='r')
            hsi_s = np.load(f_hsi_s, mmap_mode='r')
            
            self.mmap_msi.append(msi)
            self.mmap_hsi_true.append(hsi_t)
            self.mmap_hsi_sim.append(hsi_s)
            
            for p_idx in range(msi.shape[0]):
                self.index_map.append((f_idx, p_idx))

    def __len__(self):
        return len(self.index_map)

    def augment_triplet(self, x_msi, x_hsi, y_res):
        if random.random() < 0.5:
            x_msi = torch.flip(x_msi, dims=[2])
            x_hsi = torch.flip(x_hsi, dims=[2])
            y_res = torch.flip(y_res, dims=[2])

        if random.random() < 0.5:
            x_msi = torch.flip(x_msi, dims=[1])
            x_hsi = torch.flip(x_hsi, dims=[1])
            y_res = torch.flip(y_res, dims=[1])

        k = torch.randint(0, 4, (1,)).item()
        x_msi = torch.rot90(x_msi, k, dims=[1,2])
        x_hsi = torch.rot90(x_hsi, k, dims=[1,2])
        y_res = torch.rot90(y_res, k, dims=[1,2])

        return x_msi, x_hsi, y_res

    def __getitem__(self, idx):
        f_idx, p_idx = self.index_map[idx]
        
        msi = torch.from_numpy(self.mmap_msi[f_idx][p_idx].copy())
        hsi_true = torch.from_numpy(self.mmap_hsi_true[f_idx][p_idx].copy())
        hsi_sim = torch.from_numpy(self.mmap_hsi_sim[f_idx][p_idx].copy())

        X = torch.cat([msi, hsi_sim], dim=0)
        
        y = torch.abs(hsi_true - hsi_sim)

        if self.augment:
            c_msi = msi.shape[0]
            msi_aug, hsi_sim_aug, y = self.augment_triplet(X[:c_msi], X[c_msi:], y)
            X = torch.cat([msi_aug, hsi_sim_aug], dim=0)

        return X , y    

##################################################################################################################################""""""""""""""""
def create_data_loaders_spectral( use_simulated_msi, augment, batch_size=8,  num_workers=4, is_residual=False,keep_atm_wave=True,is_normalised=False):
    

    print(f" Chargement direct des patches pré-calculés depuis : {CACHE_DIR.resolve()}")
    
    mode_suffix = "simulated" if use_simulated_msi else "real"
    
    # Chemins directs vers tes dossiers déjà splittés
    train_dir = CACHE_DIR / mode_suffix / 'train'
    val_dir = CACHE_DIR / mode_suffix / 'val'
    test_dir = CACHE_DIR / mode_suffix / 'test'
    
    # Instanciation directe des datasets sans refaire de découpage
    train_dataset = SpectralDataset(train_dir, augment=augment, is_residual=is_residual,keep_atm_wave=keep_atm_wave,is_normalised=is_normalised)
    val_dataset = SpectralDataset(val_dir, augment=False, is_residual=is_residual, keep_atm_wave=keep_atm_wave,is_normalised=is_normalised)
    test_dataset = SpectralDataset(test_dir, augment=False, is_residual=is_residual,keep_atm_wave=keep_atm_wave,is_normalised=is_normalised)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    )
