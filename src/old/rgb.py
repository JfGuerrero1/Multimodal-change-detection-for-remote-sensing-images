import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from models import GradualExpansionUNet, GradualExpansionUNet_residual
import scipy.interpolate
from matplotlib.ticker import MaxNLocator
import time
import gc
import imageio
import h5py

# longuers d'onde

wvl_s2 = np.array([443., 490., 560., 665., 705., 740., 783., 842., 865., 940., 1610., 2190.])
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
 
DATA_DIR = ROOT_DIR / "data" / "dataset"
CACHE_DIR = DATA_DIR / "patches_cache"
DEFAULT_SRF_PATH = DATA_DIR / "srf_matrix_norm_s2b.npy"
RESULT_DIR = ROOT_DIR / "results"
PLOT_DIR = RESULT_DIR / "Result_plot"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT_DIR / "models"
RGB_DIR=DATA_DIR/'images_rgb'

def get_rgb(data, is_prisma=False, save_path="output.png", title="VUE RGB", add_stat=False, return_only=False):
    """Calcule et normalise un rendu RGB à partir d'un tenseur de données spectrales.
    
    Si return_only=True, calcule l'image RGB normalisée en mémoire sans sauvegarder de fichiers.
    """
    data_tensor = data

    # 1. Sélection des bandes (Indices PRISMA vs Sentinel-2)
    indices = [32, 15, 5] if is_prisma else [3, 1, 0]

    if any(i >= data_tensor.shape[0] for i in indices):
        raise IndexError("Index de bande hors limites pour ce tenseur.")

    r, g, b = data_tensor[indices[0]], data_tensor[indices[1]], data_tensor[indices[2]]
    if not is_prisma:
        # Moyenne pour recréer le canal vert propre (Sentinel-2)
        g = (data_tensor[1] + data_tensor[2]) / 2
        
    rgb = np.stack([r, g, b], axis=-1)
    rgb_nonorm = rgb.copy()

    # 2. Normalisation par percentiles (0-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip(rgb, p2, p98)
    rgb = (rgb - p2) / (p98 - p2 + 1e-8)

    # --- SI ON VEUT JUSTE LE TABLEAU NUMPY POUR NOTRE GRILLE DE PRÉDICTIONS ---
    if return_only:
        return rgb

    print("raw_values shape:", data.shape, "dtype:", data.dtype, "min:", np.nanmin(data), "max:", np.nanmax(data))
    print("number nan raw:", np.sum(np.isnan(data)))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb, vmin=0, vmax=1)
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path)
    print(f"Sauvegarde sous le nom: {save_path}")
    plt.close()

    if add_stat:
        # Ligne 1 : Canaux R, G, B individuels en niveaux de gris
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1), plt.imshow(r, vmin=0, vmax=1, cmap="grey"), plt.axis('off'), plt.title("R")
        plt.subplot(1, 3, 2), plt.imshow(g, vmin=0, vmax=1, cmap="grey"), plt.axis('off'), plt.title("G")
        plt.subplot(1, 3, 3), plt.imshow(b, vmin=0, vmax=1, cmap="grey"), plt.axis('off'), plt.title("B")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path.replace(".png", "_chan01.png"))
        plt.close()

        # Ligne 2 : Analyse poussée des zéros et histogrammes de distribution locale
        plt.figure(figsize=(15, 10))
        # Rouge
        plt.subplot(3, 4, 1), plt.imshow(r, cmap="grey"), plt.colorbar(), plt.axis('off'), plt.title("R")
        plt.subplot(3, 4, 5), plt.imshow(r==0, cmap="grey", vmin=0, vmax=1), plt.colorbar(), plt.axis('off')
        plt.title(f"R==0   ({100*np.sum(r==0)/r.size:.1f}%)")
        plt.subplot(3, 4, 9), plt.hist(r.flatten(), bins=256, range=(0, 1.5), density=True), plt.xlim(-0.1, 1.5), plt.ylim(0, 12), plt.title("R")

        # Vert
        plt.subplot(3, 4, 2), plt.imshow(g, cmap="grey"), plt.colorbar(), plt.axis('off'), plt.title("G")
        plt.subplot(3, 4, 6), plt.imshow(g==0, cmap="grey", vmin=0, vmax=1), plt.colorbar(), plt.axis('off')
        plt.title(f"G==0   ({100*np.sum(g==0)/g.size:.1f}%)")
        plt.subplot(3, 4, 10), plt.hist(g.flatten(), bins=256, range=(0, 1.5), density=True), plt.xlim(-0.1, 1.5), plt.ylim(0, 12), plt.title("G")

        # Bleu
        plt.subplot(3, 4, 3), plt.imshow(b, cmap="grey"), plt.colorbar(), plt.axis('off'), plt.title("B")
        plt.subplot(3, 4, 7), plt.imshow(b==0, cmap="grey", vmin=0, vmax=1), plt.colorbar(), plt.axis('off')
        plt.title(f"B==0   ({100*np.sum(b==0)/b.size:.1f}%)")
        plt.subplot(3, 4, 11), plt.hist(b.flatten(), bins=256, range=(0, 1.5), density=True), plt.xlim(-0.1, 1.5), plt.ylim(0, 12), plt.title("B")

        # Synthèse non-normalisée et masques cumulés
        plt.subplot(3, 4, 4), plt.imshow(rgb_nonorm, vmin=0, vmax=1), plt.axis('off'), plt.title("RGB nonorm")
        any_zero = np.any(rgb_nonorm == 0, axis=-1)
        plt.subplot(3, 4, 8), plt.imshow(any_zero, vmin=0, vmax=1, cmap="grey"), plt.axis('off')
        plt.title(f"(R or G or B) == 0   ({100*np.sum(any_zero)/r.size:.1f}%)")
        plt.subplot(3, 4, 12), plt.imshow(np.sum(rgb_nonorm==0, axis=-1), cmap="grey"), plt.colorbar(), plt.axis('off')
        plt.title("Nombre de canaux à 0")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path.replace(".png", "_chanminmax.png"))
        plt.close()

    return rgb
def gif_from_path(path):
    print("1. Ouverture du fichier...")
    with xr.open_dataset(path) as ds:
        
        total_bandes = ds["sr"].shape[2] 
        frames = []
        
        print("2. Extraction et normalisation robuste par bande...")
        for i in range(0, total_bandes, 1):
     
            bande = np.nan_to_num(ds["sr"][:, :, i].values, nan=0.0)
            
            # Calcul des percentiles pour éliminer les valeurs aberrantes de CETTE bande
            p2, p98 = np.percentile(bande, (2, 98))
            
            if p98 > p2:
                # On encadre les valeurs pour éliminer le bruit
                bande = np.clip(bande, p2, p98)
                # Étalement parfait de la dynamique entre 0 et 255
                bande = ((bande - p2) / (p98 - p2) * 255).astype(np.uint8)
            else:
                bande = np.zeros_like(bande, dtype=np.uint8)
                
            frames.append(bande)
            
        print(f"3. Génération du GIF avec {len(frames)} bandes...")
        output_path = "output.gif"
        
        # Sauvegarde au format uint8 avec le plugin pillow
        imageio.mimsave(output_path, frames, plugin='pillow', duration=0.3, loop=0)
        
        del frames
        print(f" Terminé ! Le GIF contrasté est disponible sous : {output_path}")


def afficher_image_prisma(chemin_h5):
    # 1. Ouvrir le fichier HDF5 en mode lecture ('r')
    with h5py.File(chemin_h5, "r") as f:

        def inspecter(name):
            print(name)
        

        print(f['HDFEOS/SWATHS/PRS_L1_PCO/Geolocation Fields/Latitude'][0,:])
        print(f['HDFEOS/SWATHS/PRS_L1_PCO/Geolocation Fields/Longitude'][0,:])

        
        longueurs_onde_swir = f["KDP_AUX/Cw_Swir_Matrix"]
        print(
        longueurs_onde_swir[0,:]
        )  

        longueurs_onde_vnir = f["KDP_AUX/Cw_Vnir_Matrix"]
        print(
        longueurs_onde_vnir[0,:]
        ) 

        wl_vnir = np.array( [981.6649 , 971.3399  ,960.9087 , 949.432  , 938.33093, 928.1533,
 917.82996 ,907.2473 , 896.5753  ,885.9016 , 875.2759 , 864.522 ,   853.74316,
 843.03064 , 832.3985 , 821.8129 , 811.2137 , 800.62103 , 790.0769 , 779.68396,
 769.30316 , 758.8678 , 748.54456  , 738.26 ,   728.1448 , 718.08594 , 707.88965,
 698.24664 , 688.32904 , 678.5385 ,  668.91504 , 659.42267 , 649.9523 ,  640.48737,
 631.5421 , 622.49066 , 613.62775  ,604.7203  , 595.9254 ,  587.33124 , 578.90704,
 570.5919 ,  562.3614  , 554.2172  , 546.1432 , 538.2549  , 530.4708 , 522.69,
 515.0587  , 507.49118 , 499.9806  , 492.52676 , 485.213   , 477.9621  , 470.72733,
 463.49054 , 456.16    , 448.75095 , 441.3433  , 433.99976 , 426.5187  , 418.70532,
 410.10828])


        f.visit(inspecter)


        chemin_vnir = "HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube"
        vnir_cube = f[chemin_vnir]

        print(f"Structure du cube VNIR : {vnir_cube.shape}")
        target_r = 650.0
        target_g = 550.0
        target_b = 470.0


        idx_r = np.abs(wl_vnir - target_r).argmin()
        idx_g = np.abs(wl_vnir - target_g).argmin()
        idx_b = np.abs(wl_vnir - target_b).argmin()

        print(f"Vrais indices trouvés -> Rouge: {idx_r}, Vert: {idx_g}, Bleu: {idx_b}")


       

  
        r = vnir_cube[:,idx_r, :]
        g = vnir_cube[:,idx_g, :]
        b = vnir_cube[:,idx_b, :]
    img_rgb = np.dstack((r, g, b))


    img_normalisee = np.zeros_like(img_rgb, dtype=np.float32)
    for i in range(3):
        bande = img_rgb[:, :, i]
        p2, p98 = np.percentile(bande, (2, 98))
        # On évite la division par zéro et on scale entre 0 et 1
        if p98 - p2 > 0:
            img_normalisee[:, :, i] = np.clip((bande - p2) / (p98 - p2), 0, 1)
    
    frames = []
    _,total_bandes,_=vnir_cube.shape
   
    for i in range(0, total_bandes, 1):
     
            
            # Calcul des percentiles pour éliminer les valeurs aberrantes de CETTE bande
        p2, p98 = np.percentile(bande, (2, 98))
            
        if p98 > p2:
                # On encadre les valeurs pour éliminer le bruit
            bande = np.clip(bande, p2, p98)
                # Étalement parfait de la dynamique entre 0 et 255
            bande = ((bande - p2) / (p98 - p2) * 255).astype(np.uint8)
        else:
            bande = np.zeros_like(bande, dtype=np.uint8)
                
        frames.append(bande)

    # 5. Affichage avec Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_normalisee)
    plt.title("Image Satellite  ")
    plt.savefig("Image prisma")
    plt.axis("off")  # Masquer les axes de pixels
    plt.show()
    


def build_interp_matrix(wvl_source, wvl_target):
    identity = np.eye(len(wvl_source))
    interp_func = scipy.interpolate.interp1d(
        wvl_source, identity, kind="linear", axis=0, fill_value="extrapolate"
    )
    return interp_func(wvl_target)

INTERP_MATRIX = build_interp_matrix(wvl_s2, WVL_PRS)
INTERP_MATRIX = INTERP_MATRIX.astype(np.float32)

def add_colorbar(fig,im, ax):
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.05, extend='both')
    cbar.locator = MaxNLocator(nbins=3)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=7)
    return cbar



 
L_scene = [
    "aranjuez", "arborea", "belgrade", "copperton", "eyjafjoll", "java", "los_angeles", "mrirt", "nouakchott", "prague", "tirana",
    "athens", "binh_dai", "cukotka", "fontainebleau", "jordan", "los_cabos", "novara", "quito", "suez", "valencia",
    "baltijsk", "brasilia", "cullivel", "fukushima", "kirtland", "malindi", "palermo", "rome", "sydney", "yuen_long",
    "bari", "camerino", "dellys", "guantanamo", "kitami", "mantua", "muscat", "paris", "salinas", "taiwan",
    "beer_sheva", "cape_town", "dubai", "hanging_rock", "lagos", "mexico_city", "nagaoka", "sanaa", "tampa_bay",
    "beheira", "codigoro", "dublin", "istanbul", "london", "montevideo", "new_york", "poinciana", "shanghai", "tientsin",
    "beirut", "copenhagen", "elsalto", "jagersfontein", "lorca", "mosul", "nicosia", "port_au_prince", "spinazzola", "tijuana"
]
 
L_test = ["baltijsk", "camerino", "codigoro", "copenhagen", "cullivel", "jagersfontein", "kirtland", "lorca"]
L_val = ["arborea", "athens", "beer_sheva", "istanbul", "los_cabos", "taiwan", "yuen_long"]
L_train = [
    "aranjuez","bari", "beheira", "beirut", "belgrade", "binh_dai", "brasilia", "cape_town", "copperton",
    "cukotka", "dellys", "dubai", "dublin", "elsalto", "eyjafjoll", "fontainebleau", "fukushima",
    "guantanamo", "hanging_rock", "java", "jordan", "kitami", "lagos", "london", "los_angeles",
    "malindi", "mantua", "mexico_city", "montevideo", "mosul", "mrirt", "muscat", "nagaoka",
    "new_york", "nicosia", "nouakchott", "novara", "palermo", "paris", "poinciana", "port_au_prince",
    "prague", "quito", "rome", "salinas", "sanaa", "shanghai", "spinazzola", "suez", "sydney",
    "tampa_bay", "tientsin", "tijuana", "tirana", "valencia"
]
L_train_a=["java", "jordan", "kitami", "lagos", "london", "los_angeles",
    "malindi", "mantua", "mexico_city", "montevideo", "mosul", "mrirt", "muscat", "nagaoka",
    "new_york", "nicosia", "nouakchott", "novara", "palermo", "paris", "poinciana", "port_au_prince",
    "prague", "quito", "rome", "salinas", "sanaa", "shanghai", "spinazzola", "suez", "sydney",
    "tampa_bay", "tientsin", "tijuana", "tirana", "valencia"]
if __name__ == "__main__":

    
    
    print(INTERP_MATRIX.shape)

    L_ens_scene = [ L_train_a, L_val]


    L_plot_dir = [
     
        RGB_DIR/ "train",
        RGB_DIR/ "val"]
            
    for i in range(len(L_plot_dir)):
        plot_dir = L_plot_dir[i]
        plot_dir.mkdir(parents=True, exist_ok=True)
        scenes = L_ens_scene[i]
     
 
        for scene in scenes:
            hsi_after = DATA_DIR / f"{scene}" / f"{scene}-after-prs.nc"
            msi_after = DATA_DIR / f"{scene}" / f"{scene}-after-s2.nc"

            hsi_before = DATA_DIR / f"{scene}" / f"{scene}-before-prs.nc"
            msi_before = DATA_DIR / f"{scene}" / f"{scene}-before-s2.nc"

            #MSI,HSI,HSI_interp,MSI_simule, MSI_normalise

            with xr.open_dataset(hsi_after) as ds:
                data_hsi_after = np.nan_to_num(ds["sr"].values, nan=1000.0)
                raw_values = ds["sr"].values
                if data_hsi_after.shape[-1] < data_hsi_after.shape[0]:
                    data_hsi_after= np.transpose(data_hsi_after, (2, 0, 1))
            
            with xr.open_dataset(hsi_before) as ds:
                data_hsi_before = np.nan_to_num(ds["sr"].values, nan=1000.0)
                raw_values = ds["sr"].values
                if data_hsi_before.shape[-1] < data_hsi_before.shape[0]:
                    data_hsi_before= np.transpose(data_hsi_before, (2, 0, 1))

            with xr.open_dataset(msi_after) as ds:
                data_msi_after = np.nan_to_num(ds["sr"].values, nan=1000.0)
                raw_values = ds["sr"].values
                if data_msi_after.shape[-1] < data_msi_after.shape[0]:
                    data_msi_after= np.transpose(data_msi_after, (2, 0, 1)) #c_hsi,h,w
            
            with xr.open_dataset(msi_before) as ds:
                data_msi_before = np.nan_to_num(ds["sr"].values, nan=1000.0)
                raw_values = ds["sr"].values
                if data_msi_before.shape[-1] < data_msi_before.shape[0]:
                    data_msi_before= np.transpose(data_msi_before, (2, 0, 1)) #c_msi,h,w
            
            c_hsi,h,w=data_hsi_after.shape
            c_msi,h,w=data_msi_after.shape


            #MSI Simule
            srf=np.load(DEFAULT_SRF_PATH)
            
            hsi_before_perm = data_hsi_before.transpose(1, 2, 0)
            hsi_before_flat = hsi_before_perm.reshape(-1, c_hsi)

            hsi_after_perm = data_hsi_after.transpose(1, 2, 0)
            hsi_after_flat = hsi_after_perm.reshape(-1, c_hsi)
     
            # Produit scalaire : (h*w, c_hsi) x (c_hsi, c_msi) -> (h*w, c_msi)
            msi_sim_flat_before = np.dot(hsi_before_flat, srf)
            msi_sim_flat_after = np.dot(hsi_after_flat, srf)

            # Re-shape spatial direct : (h*w, c_msi) -> (h, w, c_msi)
            # Puis remise en ordre standard pour tes fonctions : (c_msi, h, w)
            data_msi_sim_before = msi_sim_flat_before.reshape(h, w, c_msi).transpose(2, 0, 1)
            data_msi_sim_after = msi_sim_flat_after.reshape(h, w, c_msi).transpose(2, 0, 1)


            #HSI_interp

            msi_after_flat=data_msi_after.reshape(c_msi,h*w)
            msi_before_flat=data_msi_before.reshape(c_msi,h*w)

            data_msi_interp_before=np.dot(INTERP_MATRIX,msi_before_flat)

            data_msi_interp_before=data_msi_interp_before.reshape(c_hsi,h,w)

            data_msi_interp_after=np.dot(INTERP_MATRIX,msi_after_flat)
            data_msi_interp_after=data_msi_interp_after.reshape(c_hsi,h,w)

            #MSI_renormalise
            mean_sim = np.mean(data_msi_sim_after, axis=(1, 2), keepdims=True)

            std_sim = np.std(data_msi_sim_after, axis=(1, 2), keepdims=True)

            mean = np.mean(data_msi_after, axis=(1, 2), keepdims=True)
            std=np.std(data_msi_after,axis=(1,2),keepdims=True)
            data_msi_after_norm=(data_msi_after-mean)/(std+1e-8)*std_sim+mean_sim

            mean_sim = np.mean(data_msi_sim_before, axis=(1, 2), keepdims=True)
            std_sim = np.std(data_msi_sim_before, axis=(1, 2), keepdims=True)

            mean = np.mean(data_msi_before, axis=(1, 2), keepdims=True)
            print(data_msi_before.shape)
            std=np.std(data_msi_before,axis=(1,2),keepdims=True)
            data_msi_before_norm=(data_msi_before-mean)/(std+1e-8)*std_sim+mean_sim

            # 1. Génération des arrays RGB via TA fonction existante (sans stats individuelles)
            rgb_msi_before = get_rgb(data_msi_before, is_prisma=False, save_path=plot_dir / f"{scene}_before_msi.png", title="BEFORE - MSI", add_stat=False)
            rgb_hsi_before = get_rgb(data_hsi_before, is_prisma=True, save_path=plot_dir / f"{scene}_before_hsi.png", title="BEFORE - HSI", add_stat=False)
            rgb_msi_sim_before = get_rgb(data_msi_sim_before, is_prisma=False, save_path=plot_dir / f"{scene}_before_msi_sim.png", title="BEFORE - MSI Simulé", add_stat=False)
            rgb_hsi_interp_before = get_rgb(data_msi_interp_before, is_prisma=True, save_path=plot_dir / f"{scene}_before_hsi_interp.png", title="BEFORE - HSI Interpolé", add_stat=False)
            rgb_msi_before_norm = get_rgb(data_msi_before_norm, is_prisma=False, save_path=plot_dir / f"{scene}_before_msi_norm.png", title="BEFORE - MSI Normalisé", add_stat=False)

            rgb_msi_after = get_rgb(data_msi_after, is_prisma=False, save_path=plot_dir / f"{scene}_after_msi.png", title="AFTER - MSI", add_stat=False)
            rgb_hsi_after = get_rgb(data_hsi_after, is_prisma=True, save_path=plot_dir / f"{scene}_after_hsi.png", title="AFTER - HSI", add_stat=False)
            rgb_msi_sim_after = get_rgb(data_msi_sim_after, is_prisma=False, save_path=plot_dir / f"{scene}_after_msi_sim.png", title="AFTER - MSI Simulé", add_stat=False)
            rgb_hsi_interp_after = get_rgb(data_msi_interp_after, is_prisma=True, save_path=plot_dir / f"{scene}_after_hsi_interp.png", title="AFTER - HSI Interpolé", add_stat=False)
            rgb_msi_after_norm = get_rgb(data_msi_after_norm, is_prisma=False, save_path=plot_dir / f"{scene}_after_msi_norm.png", title="AFTER - MSI Normalisé", add_stat=False)

            titles = [
                "1. MSI (Sentinel-2)", "2. HSI (PRISMA)", "3. MSI Simulé (HSI x SRF)",
                "4. HSI Interpolé", "5. MSI Normalisé"
            ]


            fig_b, axes_b = plt.subplots(2, 3, figsize=(15, 10))
            before_row = [rgb_msi_before, rgb_hsi_before, rgb_msi_sim_before, rgb_hsi_interp_before, rgb_msi_before_norm]
            
            for idx in range(6):
                row, col = idx // 3, idx % 3
                if idx < 5:
                    axes_b[row, col].imshow(before_row[idx], interpolation='nearest')
                    axes_b[row, col].set_title(titles[idx], fontsize=11, fontweight='bold')
                axes_b[row, col].axis('off') # On éteint aussi la 6ème case vide
                
            plt.suptitle(f"Etat : BEFORE — Scène : {scene}", fontsize=14, fontweight='bold', y=0.96)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{scene}_GRID_BEFORE.png", bbox_inches='tight', dpi=80)
            plt.close(fig_b)


            fig_a, axes_a = plt.subplots(2, 3, figsize=(15, 10))
            after_row = [rgb_msi_after, rgb_hsi_after, rgb_msi_sim_after, rgb_hsi_interp_after, rgb_msi_after_norm]
            
            for idx in range(6):
                row, col = idx // 3, idx % 3
                if idx < 5:
                    axes_a[row, col].imshow(after_row[idx], interpolation='nearest')
                    axes_a[row, col].set_title(titles[idx], fontsize=11, fontweight='bold')
                axes_a[row, col].axis('off') 
                
            plt.suptitle(f"Etat : AFTER — Scène : {scene}", fontsize=14, fontweight='bold', y=0.96)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{scene}_GRID_AFTER.png", bbox_inches='tight', dpi=80)
            plt.close(fig_a)

            print(f" Grilles 2x3 BEFORE et AFTER sauvegardées pour la scène {scene}")
            
    
    
    #afficher_image_prisma('/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/PRS_L1_STD_OFFL_20200917103641_20200917103645_0001/PRS_L1_STD_OFFL_20210312174127_20210312174132_0001 (10).he5')


    

    




     



            




    

    
    