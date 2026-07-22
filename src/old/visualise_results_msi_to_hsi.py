import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from models import GradualExpansionUNet, GradualExpansionUNet_residual
import scipy.interpolate
from matplotlib.ticker import MaxNLocator,FixedLocator
import time
import gc
from skimage.metrics import structural_similarity as ssim_sk
torch.cuda.init()
torch.cuda.set_device(0)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- 1. CONFIGURATION DES CHEMINS ---
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = Path("/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images")
 
DATA_DIR = ROOT_DIR / "data" / "mumucd"
CACHE_DIR = DATA_DIR / "patches_cache"
DEFAULT_SRF_PATH = DATA_DIR / "srf_matrix_norm_s2b.npy"
RESULT_DIR = ROOT_DIR / "results"
PLOT_DIR = RESULT_DIR / "Result_plot"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT_DIR / "checkpoints" / "models"
RGB_DIR = DATA_DIR / 'images_rgb2'
 
matplotlib.use('Agg')
 
# --- 2. TRAITEMENTS SPECTRAUX ET CORRESPONDANCES ---

def build_interp_matrix(wvl_source, wvl_target):
    identity = np.eye(len(wvl_source))
    interp_func = scipy.interpolate.interp1d(
        wvl_source, identity, kind="linear", axis=0, fill_value="extrapolate"
    )
    return interp_func(wvl_target)





wvl_s2 = np.array([443., 490., 560., 665., 705., 740., 783., 842., 865., 940., 1610., 2190.])
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

INTERP_MATRIX = build_interp_matrix(wvl_s2, WVL_PRS).astype(np.float32)
 
L_test = ["baltijsk", "camerino", "codigoro", "copenhagen", "cullivel", "jagersfontein", "kirtland", "lorca"]
L_val = ["arborea", "athens", "beer_sheva", "istanbul", "los_cabos", "taiwan", "yuen_long"]
L_train = [
    "aranjuez", "bari", "beheira", "beirut", "belgrade", "binh_dai", "brasilia", "cape_town", "copperton",
    "cukotka", "dellys", "dubai", "dublin", "elsalto", "eyjafjoll", "fontainebleau", "fukushima",
    "guantanamo", "hanging_rock", "java", "jordan", "kitami", "lagos", "london", "los_angeles",
    "malindi", "mantua", "mexico_city", "montevideo", "mosul", "mrirt", "muscat", "nagaoka",
    "new_york", "nicosia", "nouakchott", "novara", "palermo", "paris", "poinciana", "port_au_prince",
    "prague", "quito", "rome", "salinas", "sanaa", "shanghai", "spinazzola", "suez", "sydney",
    "tampa_bay", "tientsin", "tijuana", "tirana", "valencia"
]

CHANNELS_ATM = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 200, 220]
CHANNELS_STD = [5, 11, 20, 32, 36, 40, 44, 50, 52, 59, 122, 187]
CHANNELS_ONLY_ATM = [98, 101, 104, 107, 110, 113, 116, 119, 122, 125]
 
# --- 3. METRIQUES ---
def compute_ergas(pred, target, sampling_ratio=1.0, epsilon=1e-6):
    """Calcul de l'ERGAS robuste pour des images (H, W, C) ou (C, H, W)"""
    mask = ((WVL_PRS < 1350) | ((WVL_PRS > 1500) & (WVL_PRS < 1800)) | ((WVL_PRS > 2000)))
    
    # 1. On applique le masque d'abord sur l'axe des canaux
    if pred.shape[0] < pred.shape[2]:  # Si c'est du (C, H, W)
        pred = pred[mask, :, :]
        target = target[mask, :, :]
    else:                              # Si c'est du (H, W, C)
        pred = pred[:, :, mask]
        target = target[:, :, mask]

    # 2. On passe impérativement en (C, H, W) pour la suite de ton calcul vectorisé
    if pred.shape[0] > pred.shape[2]:  # Si (H, W, C) -> (C, H, W)
        pred = np.moveaxis(pred, -1, 0)
        target = np.moveaxis(target, -1, 0)

    # 3. Maintenant num_channels est exact
    num_channels = pred.shape[0]

    # Calcul vectorisé sur les axes spatiaux (H, W) qui sont désormais les axes (1, 2)
    rmse_per_band = np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))
    mean_target_per_band = np.mean(target, axis=(1, 2))

    sum_ratio = np.sum((rmse_per_band / (mean_target_per_band + epsilon)) ** 2)
    ergas = 100 * sampling_ratio * np.sqrt((1.0 / num_channels) * sum_ratio)
    return ergas

def compute_ssim_multiband(pred, target):
    """Calcul du SSIM moyen sur l'ensemble des 120 bandes."""
    # skimage attend (H, W, C) ou demande explicitement channel_axis
    if pred.shape[0] < pred.shape[2]:  # format (C, H, W) -> on passe en (H, W, C)
        pred = np.moveaxis(pred, 0, -1)
        target = np.moveaxis(target, 0, -1)

    # data_range dépend de ta normalisation (ex: 1.0 si tes réflectances sont entre 0 et 1)
    data_range = target.max() - target.min()

    # On calcule le SSIM bande par bande en spécifiant channel_axis
    score = ssim_sk(
        pred, target, channel_axis=-1, data_range=data_range, gaussian_weights=True
    )
    return score

def compute_mse_numpy(gt, pred):
    return np.mean((gt - pred) ** 2)
 
def compute_sam_numpy(gt, pred, eps=1e-8):
    c = gt.shape[-1]
    gt_flat = gt.reshape(-1, c)
    pred_flat = pred.reshape(-1, c)
    dot_product = np.sum(gt_flat * pred_flat, axis=1)
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)
    cos_theta = dot_product / (norm_gt * norm_pred + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.mean(np.arccos(cos_theta))
 
def compute_sam_map(gt, pred, eps=1e-8):
    h, w, c = gt.shape
    gt_flat = gt.reshape(-1, c)
    pred_flat = pred.reshape(-1, c)
    dot_product = np.sum(gt_flat * pred_flat, axis=1)
    norm_gt = np.linalg.norm(gt_flat, axis=1)
    norm_pred = np.linalg.norm(pred_flat, axis=1)
    cos_theta = dot_product / (norm_gt * norm_pred + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta.reshape(h, w))
 
def compute_mae_numpy(gt, pred):
    return np.mean(np.abs(gt - pred))

def get_cmap(wvl):
    return "inferno" if ((1350 <= wvl <= 1500) or (1800 <= wvl <= 2000)) else "turbo"
# ---4. INFERENCE UNIQUE ---
def compute_scene_data(gt_path, input_path, model, model_name, is_simulated=False, is_residual=False,
                        srf_path=DEFAULT_SRF_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model.to(device)
    model.eval()

    with xr.open_dataset(gt_path) as ds_gt:
        cube_gt = np.nan_to_num(ds_gt["sr"].values, nan=0.0)
 
    h, w, c = cube_gt.shape #h,w,c
 
    if is_simulated:
        srf_matrix = np.load(srf_path) #c_hsi,c_msi
        gt_flat = cube_gt.reshape(-1, c) #h*w,c

        msi_flat = gt_flat @ srf_matrix #h*w,c_msi
        cube_msi = msi_flat.reshape(h, w, srf_matrix.shape[1])
        input_numpy = cube_msi.transpose(2, 0, 1) #c_msi,h,w
        msi_label = "MSI Simulée"
    else:
        if input_path is None:
            raise ValueError("Le paramètre 'input_path' est obligatoire pour les données réelles.")
        with xr.open_dataset(input_path) as ds_input:
            input_raw = np.nan_to_num(ds_input["sr"].values, nan=0.0) #h,w,c
        cube_msi = input_raw#h,w,c
        input_numpy = input_raw.transpose(2, 0, 1) #c,h,w
        msi_label = "MSI Réelle"
 
    cube_interp = None
    interp_tensor = None

    if is_residual:
        # Input_numpy a 12 bandes (Sentinel-2)
        interp_numpy = INTERP_MATRIX @ input_numpy.reshape(12, -1)
        interp_numpy = interp_numpy.reshape(c, h, w) #c,h,w
   
        interp_tensor = torch.from_numpy(interp_numpy).float().unsqueeze(0).to(device)
        cube_interp = interp_numpy.transpose(1, 2, 0) #h,w,c

    input_tensor = torch.from_numpy(input_numpy).float().unsqueeze(0).to(device)
    cube_res = None
    
    with torch.no_grad():
        if is_residual:
            pred = model(input_tensor, interp_tensor)
            res = pred - interp_tensor
            cube_res = res.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() #h,w,c
        else:
            pred = model(input_tensor)
            
    cube_predict = pred.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() #h,w,c
    
    # Calcul des métriques de reconstruction
    img_mse = compute_mse_numpy(cube_gt, cube_predict)
    img_sam = compute_sam_numpy(cube_gt, cube_predict)
    img_mae = compute_mae_numpy(cube_gt, cube_predict)
    img_ergas = compute_ergas(cube_predict, cube_gt)
    img_ssim = compute_ssim_multiband(cube_gt, cube_predict)
 
    return {
        "cube_gt": cube_gt,
        "cube_msi": cube_msi,
        "cube_predict": cube_predict,
        "cube_interp": cube_interp,
        "cube_res": cube_res,
        "msi_label": msi_label,
        "is_residual": is_residual,
        "img_mse": img_mse,
        "img_sam": img_sam,
        "img_mae": img_mae,
        "img_ergas": img_ergas,
        "img_ssim": img_ssim,
        "model name": model_name
    }

def add_colorbar(fig, im, ax):
    """Ajoute une colorbar ancrée à l'axe sans déformer la planche."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.xaxis.get_offset_text().set_fontsize(7)
    cbar.update_ticks()
    return cbar

def add_metrics_box(fig, data):
    """Ajoute une boîte de métriques en haut à droite de la figure."""
    metrics_text = (f"MSE: {data['img_mse']:.4e}\nMAE: {data['img_mae']:.4f}\n"
                    f"SAM: {data['img_sam']:.4f} rad\nSSIM: {data['img_ssim']:.4f}\n"
                    f"ERGAS: {data['img_ergas']:.2e}")
    fig.text(0.98, 0.95, metrics_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.9), 
             verticalalignment='top', horizontalalignment='right')
# --- 5. FONCTIONS DE VISUALISATION  ---
 
def visualise_curve(mse_path, var_path, name_curve):
    mse_per_channel = np.load(mse_path)
    var_per_channel = np.load(var_path)
 
    plt.figure(figsize=(10, 5))
    plt.axvspan(1350, 1500, color='red', alpha=0.3, label="Atmospheric Absorption")
    plt.axvspan(1800, 2000, color='red', alpha=0.3)
 
    plt.plot(WVL_PRS, mse_per_channel, color="royalblue", linewidth=2, label="Mean of MSE SAM : rad")
    plt.plot(WVL_PRS, mse_per_channel + var_per_channel, color="royalblue", linestyle="--", alpha=0.5)
 
    plt.title("Reconstruction error by each channels")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("MSE")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(name_curve, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graphique d'erreur sauvegardé : {name_curve}")
 
def visualise_hsi_from_data(data, save_name, channels, plot_dir):
    """
    First row: ground truth HSI
    Second row: Prediction HSI
    Third row: if residual, interpolation
    """
    cube_gt = data["cube_gt"]
    cube_predict = data["cube_predict"]
    cube_interp = data.get("cube_interp", None) # Utilisation de .get() pour éviter un KeyError
    is_residual = data["is_residual"]
    img_mse, img_sam, img_mae,img_ssim,img_ergas = data["img_mse"], data["img_sam"], data["img_mae"],data["img_ssim"],data["img_ergas"]

    num_rows = 3 if is_residual else 2
    num_cols = len(channels)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), constrained_layout=True)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    title_str = f"Planche Comparative — MSE globale : {img_mse:.4f} | MAE globale : {img_mae:.4f} | SAM global : {img_sam:.4f} rad | SSIM : {img_ssim:.4f} |ERGAS : {img_ergas:.2e}"
    fig.suptitle(title_str, fontsize=14, y=1.02)  

    for i, idx in enumerate(channels):
        canal_gt = cube_gt[:, :, idx]
        canal_pred = cube_predict[:, :, idx]
        chan_mae = compute_mae_numpy(canal_gt, canal_pred)

        current_wvl = WVL_PRS[idx]
        is_atmospheric = (1350 <= current_wvl <= 1500) or (1800 <= current_wvl <= 2000)
        map=get_cmap(current_wvl)

        # Calcul de la dynamique commune
        pct_min_list = [np.percentile(canal_gt, 2), np.percentile(canal_pred, 2)]
        pct_max_list = [np.percentile(canal_gt, 98), np.percentile(canal_pred, 98)]
        
        # Si on est en mode résiduel, on prend aussi en compte l'interp dans la dynamique
        if is_residual and cube_interp is not None:
            canal_interp = cube_interp[:, :, idx] # Déplacé ici pour éviter le crash si None
            pct_min_list.append(np.percentile(canal_interp, 2))
            pct_max_list.append(np.percentile(canal_interp, 98))

        vmin_val = min(pct_min_list)
        vmax_val = max(pct_max_list)

        if is_atmospheric:
            vmax_val = min(vmax_val, 0.2)
        
        

        suffixe_titre = " [Atm]" if is_atmospheric else ""

        # Ligne 0 : GT
        im0 = axes[0, i].imshow(canal_gt, cmap=map, vmin=vmin_val, vmax=vmax_val)
        axes[0, i].set_title(f"GT HSI - ({WVL_PRS[idx]:.1f} nm){suffixe_titre}", fontsize=9)
        axes[0, i].axis("off")
        add_colorbar(fig, im0, axes[0, i])

        # Ligne 1 : PRED
        im1 = axes[1, i].imshow(canal_pred, cmap=map, vmin=vmin_val, vmax=vmax_val)
        axes[1, i].set_title(f"PRED HSI - ({WVL_PRS[idx]:.1f} nm)\nMAE: {chan_mae:.4f} \n| Min/Max: {canal_pred.min():.2f}/{canal_pred.max():.2f}", fontsize=9)
        axes[1, i].axis("off")
        add_colorbar(fig, im1, axes[1, i])

        # Ligne 2 : INTERP 
        if is_residual and cube_interp is not None:
            im2 = axes[2, i].imshow(canal_interp, cmap=map, vmin=vmin_val, vmax=vmax_val)
            axes[2, i].set_title(f"HSI Interp - ({WVL_PRS[idx]:.1f} nm)", fontsize=9)
            axes[2, i].axis("off")
            add_colorbar(fig, im2, axes[2, i])

    output_plot_path = plot_dir / f"{save_name}.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")  
    plt.close()
    print(f"[Succès] Planche sauvegardée sous : {output_plot_path}")

def visualise_hsi_msi_from_data(data, save_name,  plot_dir):
    """
    First row: ground truth HSI
    Second row: Prediction HSI
    Third row: MSI
    """
    channels=CHANNELS_STD
    cube_gt = data["cube_gt"]
    cube_predict = data["cube_predict"]
    cube_msi=data["cube_msi"]
 
    img_mse, img_sam, img_mae,img_ssim,img_ergas = data["img_mse"], data["img_sam"], data["img_mae"],data["img_ssim"],data["img_ergas"]


    num_rows = 3 
    num_cols = len(channels)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), constrained_layout=True)

    title_str = f"Planche Comparative — MSE globale : {img_mse:.4f} | MAE globale : {img_mae:.4f} | SAM global : {img_sam:.4f} rad | SSIM : {img_ssim:.4f} |ERGAS : {img_ergas:.2e}"
    fig.suptitle(title_str, fontsize=14, y=1.02)  

    for i, idx in enumerate(channels):
        canal_gt = cube_gt[:, :, idx]
        canal_pred = cube_predict[:, :, idx]
        canal_msi=cube_msi[:,:,i]
        chan_mae = compute_mae_numpy(canal_gt, canal_pred)

        current_wvl = WVL_PRS[idx]
        is_atmospheric = (1350 <= current_wvl <= 1500) or (1800 <= current_wvl <= 2000)
        map=get_cmap(current_wvl)

        # Calcul de la dynamique commune
        vmin_val =min (np.percentile(canal_gt, 2), np.percentile(canal_pred, 2))
        vmax_val =max(np.percentile(canal_gt, 98), np.percentile(canal_pred, 98))
        
        if is_atmospheric:
            vmax_val = min(vmax_val, 0.2)

        suffixe_titre = " [Atm]" if is_atmospheric else ""

        # Ligne 0 : GT
        im0 = axes[0, i].imshow(canal_gt, cmap=map, vmin=vmin_val, vmax=vmax_val)
        axes[0, i].set_title(f"GT HSI - ({WVL_PRS[idx]:.1f} nm){suffixe_titre}", fontsize=9)
        axes[0, i].axis("off")
        add_colorbar(fig, im0, axes[0, i])

        # Ligne 1 : PRED
        im1 = axes[1, i].imshow(canal_pred, cmap=map, vmin=vmin_val, vmax=vmax_val)
        axes[1, i].set_title(f"PRED HSI - ({WVL_PRS[idx]:.1f} nm)\nMAE: {chan_mae:.4f} \n| Min/Max: {canal_pred.min():.2f}/{canal_pred.max():.2f}", fontsize=9)
        axes[1, i].axis("off")
        add_colorbar(fig, im1, axes[1, i])

        # Ligne 2 : MSI
        vmin_msi=np.percentile(canal_msi,2)
        vmax_msi=np.percentile(canal_msi,98)
        im2 = axes[2, i].imshow(canal_msi, cmap=map, vmin=vmin_msi, vmax=vmax_msi)
        axes[2, i].set_title(f"MSI - ({wvl_s2[i]:.1f} nm)", fontsize=9)
        axes[2, i].axis("off")
        add_colorbar(fig, im2, axes[2, i])

    output_plot_path = plot_dir / f"{save_name}.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")  
    plt.close()
    print(f"[Succès] Planche sauvegardée sous : {output_plot_path}")

def visualise_synthesis(data, save_name, plot_dir):
    """
    Génère la planche de synthèse ultime en format 2x3.
    Ligne 1 : RGB MSI (local) | RGB PRED (commun) | RGB GT (commun)
    Ligne 2 : Carte MAE | Carte SAM | Courbe de MAE spectrale par longueur d'onde
    """
    cube_gt = data["cube_gt"]
    cube_predict = data["cube_predict"]
    cube_msi = data["cube_msi"]
    model_name = data.get("model name", "Modèle ML")
    img_mse, img_sam, img_mae,img_ssim,img_ergas = data["img_mse"], data["img_sam"], data["img_mae"],data["img_ssim"],data["img_ergas"]
    
    
    # 1. Alignement des indices RGB pour PRISMA (HSI)
    idx_r_hsi = np.argmin(np.abs(WVL_PRS - 665.0))
    idx_v_hsi = np.argmin(np.abs(WVL_PRS - 560.0))
    idx_b_hsi = np.argmin(np.abs(WVL_PRS - 490.0))
    
    # Indices RGB pour Sentinel-2 (MSI : B4=Rouge, B3=Vert, B2=Bleu -> idx 3, 2, 1)
    idx_r_msi, idx_v_msi, idx_b_msi = 3, 2, 1
    
    # Fonction interne de conversion RGB robuste (percentiles 2-98)
    def to_rgb(cube, r, g, b, custom_bounds=None):
        rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
        rgb = np.nan_to_num(rgb)  # Sécurité pour les masques/NaN
        for c in range(3):
            if custom_bounds is None:
                v_min = np.percentile(rgb[:, :, c], 2)
                v_max = np.percentile(rgb[:, :, c], 98)
            else:
                v_min, v_max = custom_bounds[c]
            rgb[:, :, c] = np.clip((rgb[:, :, c] - v_min) / (v_max - v_min + 1e-8), 0, 1)
        return rgb

    # --- SÉPARATION DES DYNAMIQUES VISUELLES ---
    # RGB MSI autonome (évite le cramage)
    rgb_msi = to_rgb(cube_msi, idx_r_msi, idx_v_msi, idx_b_msi)
    
    # RGB HSI partagé (pour une comparaison Pred vs GT rigoureuse)
    bounds_hsi = []
    for chan in [idx_r_hsi, idx_v_hsi, idx_b_hsi]:
        v_min = min(np.percentile(cube_gt[:, :, chan], 2), np.percentile(cube_predict[:, :, chan], 2))
        v_max = max(np.percentile(cube_gt[:, :, chan], 98), np.percentile(cube_predict[:, :, chan], 98))
        bounds_hsi.append((v_min, v_max))
        
    rgb_pred = to_rgb(cube_predict, idx_r_hsi, idx_v_hsi, idx_b_hsi, custom_bounds=bounds_hsi)
    rgb_gt = to_rgb(cube_gt, idx_r_hsi, idx_v_hsi, idx_b_hsi, custom_bounds=bounds_hsi)
    
    # --- CALCUL DES DIAGNOSTICS ---
    mae_map = np.mean(np.abs(cube_gt - cube_predict), axis=-1)
    sam_map = compute_sam_map(cube_predict, cube_gt)
    mae_per_band = np.mean(np.abs(cube_gt - cube_predict), axis=(0, 1)) # Erreur spectrale
    
    # --- CONSTRUCTION DE LA FIGURE 2x3 ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    clean_title = model_name.replace('\n', ' ')
    fig.suptitle(f"Planche de Synthèse Globale — {clean_title} \n MSE globale : {img_mse:.4f} | MAE globale : {img_mae:.4f} | SAM global : {img_sam:.4f} rad | SSIM : {img_ssim:.4f} |ERGAS : {img_ergas:.2e}", fontsize=14, y=1.02, weight='bold')
    
    # --- LIGNE 1 : LES IMAGES RGB ---
    # [0, 0] MSI
    axes[0, 0].imshow(rgb_msi)
    axes[0, 0].set_title("RGB — Entrée MSI (Sentinel-2)\n[Dynamique Locale Locale]", fontsize=11, pad=6)
    axes[0, 0].axis("off")
    
    # [0, 1] PRED
    axes[0, 1].imshow(rgb_pred)
    axes[0, 1].set_title("RGB — HSI Prédite (Modèle)\n[Dynamique Commune HSI]", fontsize=11, pad=6)
    axes[0, 1].axis("off")
    
    # [0, 2] GT
    axes[0, 2].imshow(rgb_gt)
    axes[0, 2].set_title("RGB — HSI Vérité Terrain (PRISMA)\n[Dynamique Commune HSI]", fontsize=11, pad=6)
    axes[0, 2].axis("off")
    
    # --- LIGNE 2 : LES DIAGNOSTICS QUANTITATIFS ---
    # [1, 0] Carte MAE spatialisée
    vmax_mae = max(np.percentile(mae_map, 98), 0.02)
    im_mae = axes[1, 0].imshow(mae_map, cmap="inferno", vmin=0, vmax=vmax_mae)
    axes[1, 0].set_title(f"Carte d'Erreur MAE Spatiale\n(Moyenne globale : {data['img_mae']:.4f})", fontsize=11, pad=6)
    axes[1, 0].axis("off")
    add_colorbar(fig, im_mae, axes[1, 0])
    
    # [1, 1] Carte SAM spatialisée
    vmax_sam = max(np.percentile(sam_map, 98), 0.05)
    im_sam = axes[1, 1].imshow(sam_map, cmap="inferno", vmin=0, vmax=vmax_sam)
    axes[1, 1].set_title(f"Carte  SAM\n(Moyenne globale : {data['img_sam']:.4f} rad)", fontsize=11, pad=6)
    axes[1, 1].axis("off")
    add_colorbar(fig, im_sam, axes[1, 1])
    
    # [1, 2] Courbe physique d'erreur spectrale (Comble le trou !)
    ax_spec = axes[1, 2]
    ax_spec.plot(WVL_PRS, mae_per_band, color='crimson', lw=2, label='MAE par bande')
    ax_spec.set_title("Profil de l'Erreur Spectrale (MAE)", fontsize=11, pad=6)
    ax_spec.set_xlabel("Longueur d'onde (nm)", fontsize=9)
    ax_spec.set_ylabel("MAE", fontsize=9)
    ax_spec.grid(True, linestyle=':', alpha=0.6)
    
    # Zones d'absorption de la vapeur d'eau atmosphérique
    ax_spec.axvspan(1350, 1500, color='gray', alpha=0.15, label='Abs. H2O')
    ax_spec.axvspan(1800, 2000, color='gray', alpha=0.15)
    ax_spec.legend(fontsize=9, loc='upper right')

    # Enregistrement
    output_plot_path = plot_dir / f"{save_name}.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Succès] Planche de synthèse 2x3 enregistrée sous : {output_plot_path}")


def visualise_residual(data, save_name,channels,plot_dir):
    cube_gt = data["cube_gt"]
    cube_interp = data["cube_interp"]
    cube_predict = data["cube_predict"]
    cube_true_res = cube_gt - cube_interp
    cube_pred_res = cube_predict - cube_interp

    img_mse, img_sam, img_mae,img_ssim,img_ergas = data["img_mse"], data["img_sam"], data["img_mae"],data["img_ssim"],data["img_ergas"]
   
    num_rows = 4 
    num_cols = len(channels)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), constrained_layout=True)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    title_str = f"Planche Comparative Residu : MSE globale : {img_mse:.4f} | MAE globale : {img_mae:.4f} | SAM global : {img_sam:.4f} rad | SSIM : {img_ssim:.4f} |ERGAS : {img_ergas:.2e}"
    fig.suptitle(title_str, fontsize=14, y=1.02, weight='bold')

    for i, idx in enumerate(channels):
        canal_gt = cube_gt[:, :, idx]
        canal_interp = cube_interp[:, :, idx]
        canal_true_res = cube_true_res[:, :, idx]
        canal_res = cube_pred_res[:, :, idx]
        
        res_mae = compute_mae_numpy(canal_true_res, canal_res)


        # Ajout de la détection des bandes atmosphériques
        current_wvl = WVL_PRS[idx]
        is_atmospheric = (1350 <= current_wvl <= 1500) or (1800 <= current_wvl <= 2000)

        # Dynamique commune GT + Interp pour éviter les mauvaises surprises
        vmin_val = min(np.percentile(canal_gt, 2), np.percentile(canal_interp, 2))
        vmax_val = max(np.percentile(canal_gt, 98), np.percentile(canal_interp, 98))

        # Application du bridage si atmosphérique
        if is_atmospheric:
            vmax_val = min(vmax_val, 0.2)
            
        suffixe_titre = " [Atm]" if is_atmospheric else ""

        # --- Symétrisation adaptative et robuste (Parfait !) ---
        v_res = max(np.percentile(np.abs(canal_res), 98), np.percentile(np.abs(canal_true_res), 98))
        v_res = max(v_res, 0.02)

        # Ligne 0 : GT
        im0 = axes[0, i].imshow(canal_gt, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
        axes[0, i].set_title(f"GT HSI - ({current_wvl:.1f} nm){suffixe_titre}", fontsize=9)
        axes[0, i].axis("off")
        add_colorbar(fig, im0, axes[0, i])

        # Ligne 1 : Interp
        im1 = axes[1, i].imshow(canal_interp, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
        axes[1, i].set_title(f"HSI interpolée - ({current_wvl:.1f} nm)", fontsize=9)
        axes[1, i].axis("off")
        add_colorbar(fig, im1, axes[1, i])

        # Ligne 2 : Résidu Prédit
        im2 = axes[2, i].imshow(canal_res, cmap="seismic", vmin=-v_res, vmax=v_res)
        axes[2, i].set_title(f"Residu prédit\nMAE vs Vrai Res: {res_mae:.4f}", fontsize=9)
        axes[2, i].axis("off")
        add_colorbar(fig, im2, axes[2, i])

        # Ligne 3 : Vrai Résidu
        im3 = axes[3, i].imshow(canal_true_res, cmap="seismic", vmin=-v_res, vmax=v_res)
        axes[3, i].set_title(f"Vrai résidu - ({current_wvl:.1f} nm)", fontsize=9)
        axes[3, i].axis("off")
        add_colorbar(fig, im3, axes[3, i])
        
    output_plot_path = plot_dir / f"{save_name}.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")  
    plt.close()
    print(f"[Succès] Planche Résidus sauvegardée sous : {output_plot_path}")

def heat_map_err_from_data(data, save_name, channels,plot_dir):
    cube_gt = data["cube_gt"]
    cube_predict = data["cube_predict"]
    modele_name = data["model name"]
    img_mse, img_sam, img_mae,img_ssim,img_ergas = data["img_mse"], data["img_sam"], data["img_mae"],data["img_ssim"],data["img_ergas"]

    fig, axes = plt.subplots(1, len(channels) + 1, figsize=(18, 6), constrained_layout=True)
    title_str = f"— Heat map {modele_name}  \n MSE globale : {img_mse:.4f} | MAE globale : {img_mae:.4f} | SAM global : {img_sam:.4f} rad | SSIM : {img_ssim:.4f} |ERGAS : {img_ergas:.2e}"
    fig.suptitle(title_str, fontsize=14, y=1.02, weight='bold')

    # --- CORRECTIF : vmax calculé de manière robuste ---
    vmax_err = min(max([np.percentile(np.abs(cube_gt[:, :, c] - cube_predict[:, :, c]), 98) for c in channels]), 0.3)
    
    for i, idx in enumerate(channels):
        canal_gt = cube_gt[:, :, idx]
        canal_pred = cube_predict[:, :, idx]
        err = np.abs(canal_gt - canal_pred)

        im0 = axes[i].imshow(err, cmap="inferno", vmax=vmax_err)
        axes[i].set_title(f"Error Map {WVL_PRS[idx]:.1f} nm", fontsize=9)
        axes[i].axis("off")
        add_colorbar(fig, im0, axes[i])

    sam = compute_sam_map(cube_predict, cube_gt)
    im = axes[-1].imshow(sam, cmap='inferno')
    axes[-1].set_title(f"SAM map  ", fontsize=9)
    axes[-1].axis("off")
    add_colorbar(fig, im, axes[-1])

    output_plot_path = plot_dir / f"{save_name}.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Succès] Map sauvegardée sous : {output_plot_path}")

def compare_heat_map(data1, data2,  save_name,channels, plot_dir):
    cube_gt1 = data1["cube_gt"]
    cube_predict1 = data1["cube_predict"]
    name1 = data1["model name"]
    cube_predict2 = data2["cube_predict"]
    name2 = data2["model name"]
    img_mse, img_sam, img_mae,img_ssim,img_ergas = data1["img_mse"], data1["img_sam"], data1["img_mae"],data1["img_ssim"],data1["img_ergas"]
    im_mse, im_sam, im_mae,im_ssim,im_ergas = data2["img_mse"], data2["img_sam"], data2["img_mae"],data2["img_ssim"],data2["img_ergas"]
    

    fig, axes = plt.subplots(3, len(channels) + 1, figsize=(24, 12), constrained_layout=True)
    title_str = f"— Heat map : {name1} \n {name2} \n Difference \n MSE globale : {img_mse:.4f} | MAE globale : {img_mae:.4f} | SAM global : {img_sam:.4f} rad | SSIM : {img_ssim} |ERGAS : {img_ergas} \n MSE globale : {im_mse:.4f} | MAE globale : {im_mae:.4f} | SAM global : {im_sam:.4f} rad | SSIM : {im_ssim} |ERGAS : {im_ergas} "
    fig.suptitle(title_str, fontsize=14, y=1.02)

    for i, idx in enumerate(channels):
        canal_gt1 = cube_gt1[:, :, idx]
        canal_pred1 = cube_predict1[:, :, idx]
        err1 = np.abs(canal_gt1 - canal_pred1)
        
        canal_pred2 = cube_predict2[:, :, idx]
        err2 = np.abs(canal_gt1 - canal_pred2)
        
        vmax_err = min(max(np.percentile(err1, 98), np.percentile(err2, 98)), 0.3) 
        diff = (canal_pred1 - canal_pred2)
        vmax_diff = min(np.percentile(np.abs(diff), 98), 0.5)

        im0 = axes[0, i].imshow(err1, cmap="inferno", vmax=vmax_err)
        axes[0, i].set_title(f"Error Map {WVL_PRS[idx]:.1f} nm", fontsize=9)
        axes[0, i].axis("off")
        add_colorbar(fig, im0, axes[0, i])

        im1 = axes[1, i].imshow(err2, cmap="inferno", vmax=vmax_err)
        axes[1, i].set_title(f"Error Map {WVL_PRS[idx]:.1f} nm", fontsize=9)
        axes[1, i].axis("off")
        add_colorbar(fig, im1, axes[1, i])

        im2 = axes[2, i].imshow(diff, cmap="seismic", vmax=vmax_diff, vmin=-vmax_diff)
        axes[2, i].set_title(f"{WVL_PRS[idx]:.1f} nm", fontsize=9)
        axes[2, i].axis("off")
        add_colorbar(fig, im2, axes[2, i])

    sam1 = compute_sam_map(cube_predict1, cube_gt1)
    img_sam1 = compute_sam_numpy(cube_predict1, cube_gt1)
    im_sam1 = axes[0, -1].imshow(sam1, cmap='inferno')
    axes[0, -1].set_title(f"SAM map \n SAM global : {img_sam1:.4f}", fontsize=9)
    axes[0, -1].axis("off")
    add_colorbar(fig, im_sam1, axes[0, -1])

    sam2 = compute_sam_map(cube_predict2, cube_gt1) 
    img_sam2 = compute_sam_numpy(cube_predict2, cube_gt1)
    im_sam2 = axes[1, -1].imshow(sam2, cmap='inferno')
    axes[1, -1].set_title(f"SAM map \n SAM global : {img_sam2:.4f}", fontsize=9)
    axes[1, -1].axis("off")
    add_colorbar(fig, im_sam2, axes[1, -1])

    axes[2, -1].axis("off")  

    output_plot_path = plot_dir / f"{save_name}.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Succès] Comparaison sauvegardée sous : {output_plot_path}")

def compare_predictions_line(data1, data2,  save_name,channels, plot_dir):
    cube_gt = data1["cube_gt"]
    cube_predict1 = data1["cube_predict"]
    name1 = data1["model name"]
    cube_predict2 = data2["cube_predict"]
    name2 = data2["model name"]



    fig, axes = plt.subplots(3, len(channels) + 1, figsize=(22, 10), constrained_layout=True)
    if len(channels) == 1:
        axes = np.expand_dims(axes, axis=1)

    title_str = f"— Comparaison : Ground Truth vs {name1} vs {name2} \n "
    fig.suptitle(title_str, fontsize=14, y=1.02)

    for i, idx in enumerate(channels):
        canal_gt = cube_gt[:, :, idx]
        canal_pred1 = cube_predict1[:, :, idx]
        canal_pred2 = cube_predict2[:, :, idx]
        
        wavelength = WVL_PRS[idx]
        is_atmospheric = (1350 <= wavelength <= 1500) or (1800 <= wavelength <= 2000)

        vmin_col = min(np.percentile(canal_gt, 2), np.percentile(canal_pred1, 2), np.percentile(canal_pred2, 2))
        vmax_col = max(np.percentile(canal_gt, 98), np.percentile(canal_pred1, 98), np.percentile(canal_pred2, 98))
        
        if is_atmospheric:
            vmax_col = min(vmax_col, 0.3)

        im0 = axes[0, i].imshow(canal_gt, cmap="turbo", vmin=vmin_col, vmax=vmax_col)
        axes[0, i].set_title(f"GT | {wavelength:.1f} nm", fontsize=9)
        axes[0, i].axis("off")
        add_colorbar(fig, im0, axes[0, i])

        im1 = axes[1, i].imshow(canal_pred1, cmap="turbo", vmin=vmin_col, vmax=vmax_col)
        axes[1, i].set_title(f"{name1}", fontsize=9)
        axes[1, i].axis("off")
        add_colorbar(fig, im1, axes[1, i])

        im2 = axes[2, i].imshow(canal_pred2, cmap="turbo", vmin=vmin_col, vmax=vmax_col)
        axes[2, i].set_title(f"{name2}", fontsize=9)
        axes[2, i].axis("off")
        add_colorbar(fig, im2, axes[2, i])

    sam1 = compute_sam_map(cube_predict1, cube_gt)
    img_sam1 = compute_sam_numpy(cube_predict1, cube_gt)
    im_sam1 = axes[1, -1].imshow(sam1, cmap='inferno')
    axes[1, -1].set_title(f"SAM map \n SAM global : {img_sam1:.4f}", fontsize=9)
    axes[1, -1].axis("off")
    add_colorbar(fig, im_sam1, axes[1, -1])

    sam2 = compute_sam_map(cube_predict2, cube_gt) 
    img_sam2 = compute_sam_numpy(cube_predict2, cube_gt)
    im_sam2 = axes[2, -1].imshow(sam2, cmap='inferno')
    axes[2, -1].set_title(f"SAM map \n SAM global : {img_sam2:.4f}", fontsize=9)
    axes[2, -1].axis("off")
    add_colorbar(fig, im_sam2, axes[2, -1])

    axes[0, -1].axis("off") 

    output_plot_path = plot_dir / f"{save_name}.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Succès] Planche comparative sauvegardée sous : {output_plot_path}")

def load_fixed_weights(model, path, device):
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Chargement réussi !")
    except RuntimeError as e:
        print(f"Échec du chargement : {e}")
    return model
 
# --- 6. EXECUTION PRINCIPALE ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Modele residuel MSE")
    poids_res = MODEL_DIR / 'gradualexpansionunet_res_real_aug_SAM-0.5_MSE1.0_MAE0.0_lr-0.0005_no_mlp_residual'
    model_res = GradualExpansionUNet_residual(in_msi=12, in_hsi=230, interpolation_mode="Bilinear", activation="silu", with_batch_norm=True, with_mlp_spectral=False)
    model_res = load_fixed_weights(model_res, poids_res, device)
    
    print("Modele residuel MAE")
    poids_res_mae = MODEL_DIR / 'gradualexpansionunet_res_real_aug_SAM-0.5_MSE0.0_MAE1.0_lr-0.001_no_mlp_residual'
    model_res_mae = GradualExpansionUNet_residual(in_msi=12, in_hsi=230, interpolation_mode="Bilinear", activation="silu", with_batch_norm=True, with_mlp_spectral=False)
    model_res_mae = load_fixed_weights(model_res_mae, poids_res_mae, device)

    print("Modele residuel MSE avec MLP")
    poids_res_mlp = MODEL_DIR / 'gradualexpansionunet_res_real_aug_SAM-0.5_MSE1.0_MAE0.0_lr-0.0005_mlp_residual'
    model_res_mlp = GradualExpansionUNet_residual(in_msi=12, in_hsi=230, interpolation_mode="Bilinear", activation="silu", with_batch_norm=True, with_mlp_spectral=True)
    model_res_mlp = load_fixed_weights(model_res_mlp, poids_res_mlp, device)

    L_ens_scene = [L_test, L_train, L_val]
    L_plot_dir = [
        RESULT_DIR / "Result_plot" / "MSI_to_HSI_result_mae2" / "test",
        RESULT_DIR / "Result_plot" / "MSI_to_HSI_result_mae2" / "train",
        RESULT_DIR / "Result_plot" / "MSI_to_HSI_result_mae2" / "val"
    ]
            
    for i in range(len(L_plot_dir)):
        plot_dir = L_plot_dir[i]
        plot_dir.mkdir(parents=True, exist_ok=True)
        scenes = L_ens_scene[i]
 
        for scene in scenes:
            print(f"\n--- Traitement de la scène : {scene} ---")
            gt = DATA_DIR / f"{scene}" / f"{scene}-after-prs.nc"
            input_path = DATA_DIR / f"{scene}" / f"{scene}-after-s2.nc"

            # Inférence Modèle 1 (Residual MSE)
            data_res_mlp = compute_scene_data(
                gt_path=gt, input_path=input_path, model=model_res_mlp,
                is_simulated=False, is_residual=True, device=device,
                model_name="modèle residuel \n Loss MSE+0.5 SAM with mlp"
            )

            # Inférence Modèle 2 (Residual MSE avec MLP)
            data_res_mae = compute_scene_data(
                gt_path=gt, input_path=input_path, model=model_res_mae,
                is_simulated=False, is_residual=True, device=device,
                model_name="modèle residuel \n Loss MAE+0.5 SAM "
            )
       
            # --- Génération des plots ---
            visualise_hsi_from_data(data_res_mae, f"Image mae2 {scene}", CHANNELS_STD, plot_dir/"hsi")
            visualise_hsi_from_data(data_res_mae, f"Image mae2 with atm{scene}", CHANNELS_ATM, plot_dir/"hsi")
            visualise_hsi_from_data(data_res_mae, f"Image mae2 only atm {scene}", CHANNELS_ONLY_ATM, plot_dir/"hsi")
            visualise_residual(data_res_mae, f"Image  residu {scene}",CHANNELS_STD, plot_dir/"res")
            visualise_residual(data_res_mae,  f"Image  residu with atm{scene}", CHANNELS_ATM,plot_dir/"res")
            visualise_residual(data_res_mae,  f"Image residu only atm{scene}",CHANNELS_ONLY_ATM ,plot_dir/"res")
            heat_map_err_from_data(data_res_mae, f"heatmap avec mae2 {scene}",CHANNELS_STD, plot_dir/"heatmap")
            visualise_hsi_msi_from_data(data_res_mae,f"Image HSI MSI {scene}",plot_dir/"hsi_msi")
            visualise_synthesis(data_res_mae,f"Synthesis {scene}",plot_dir/"hsi_msi")
           
            # --- SÉCURITÉ MÉMOIRE : Nettoyage agressif ---
            del data_res_mae, data_res_mlp
            gc.collect()
            torch.cuda.empty_cache()
    