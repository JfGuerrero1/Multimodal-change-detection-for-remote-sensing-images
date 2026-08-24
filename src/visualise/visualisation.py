import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path de Python
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
  sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import wandb
# Sécurité pour les serveurs sans GUI
matplotlib.use('Agg')
from src.constants import WVL_PRS, DW_INFO,WVL_S2
from src.old.metrics_and_loss.metrics import compute_sam_map,compute_mae,compute_ergas,compute_mrae,compute_ssim_multiband,compute_mse,compute_sam,compute_psnr,compute_rmse    
CHANNELS_ATM = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 200, 220]
CHANNELS_STD = [5, 11, 20, 32, 36, 40, 44, 50, 52, 59, 122, 187]
CHANNELS_ONLY_ATM = [98, 101, 104, 107, 110, 113, 116, 119, 122, 125]



def add_colorbar(fig, im, ax):
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.05, extend='both')
    
    # Récupère les bornes actuelles
    vmin, vmax = im.get_clim()
    
    # Force les ticks aux extrémités + quelques valeurs intermédiaires
    ticks = [vmin, (vmin + vmax) / 2, vmax]
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal')

    # Force la notation scientifique (le multiplicateur 1e-X se placera sur le côté)
    cbar.formatter.set_powerlimits((0, 0))

    # OPTIONNEL : Réduis un peu la police des chiffres et du multiplicateur pour pas que ça bave
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.get_offset_text().set_fontsize(8)

    # Valide les changements
    cbar.update_ticks()
    return cbar
def get_cmap(wvl):
    return "inferno" if ((1350 <= wvl <= 1500) or (1800 <= wvl <= 2000)) else "turbo"


def visualise_image_from_data(data, save_name, channels, plot_dir):
    """
    1ere ligne GT
    2eme ligne Prediction
    3eme ligne MSI interpolée si en mode residuel
    Derniere ligne MSI réelle ou MSI simulée
    """


    cube_gt = data["cube_gt"]
    cube_msi = data["cube_msi"]
    cube_predict = data["cube_predict"]
    cube_interp = data["cube_interp"]
    msi_label = data["msi_label"]
    is_residual = data["is_residual"]
    img_mse, img_sam, img_mae = data["img_mse"], data["img_sam"], data["img_mae"]

    num_rows = 4 if is_residual else 3
    num_cols = len(channels)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows),constrained_layout=True)

    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    title_str = f"Planche Comparative — MSE globale : {img_mse:.4f} | MAE globale : {img_mae:.4f} | SAM global : {img_sam:.4f} rad"
    fig.suptitle(title_str, fontsize=14, y=1.02)  # 



    for i, idx in enumerate(channels):
        canal_gt = cube_gt[:, :, idx]
        canal_pred = cube_predict[:, :, idx]

        msi_idx = min(i, cube_msi.shape[2] - 1)
        canal_msi = cube_msi[:, :, msi_idx]

        chan_mse = compute_mse(canal_gt, canal_pred)
        chan_mae = compute_mae(canal_gt, canal_pred)

        if is_residual:
            canal_interp = cube_interp[:, :, idx]
            vmin_val = min(canal_gt.min(), canal_pred.min(), canal_interp.min(), canal_msi.min())
            vmax_val = max(canal_gt.max(), canal_pred.max(), canal_interp.max(), canal_msi.max())
        else:
            vmin_val = min(canal_gt.min(), canal_pred.min(), canal_msi.min())
            vmax_val = max(canal_gt.max(), canal_pred.max(), canal_msi.max())

        im0 = axes[0, i].imshow(canal_gt, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
        axes[0, i].set_title(f"GT HSI - ({WVL_PRS[idx]:.1f} nm)", fontsize=9)
        axes[0, i].axis("off")
        add_colorbar(fig,im0, axes[0, i])

        im1 = axes[1, i].imshow(canal_pred, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
        axes[1, i].set_title(f"PRED HSI - ({WVL_PRS[idx]:.1f} nm)\nMSE: {chan_mse:.4f} | MAE: {chan_mae:.4f}", fontsize=9)
        axes[1, i].axis("off")
        add_colorbar(fig,im1, axes[1, i])

        if is_residual:
            im2 = axes[2, i].imshow(canal_interp, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
            axes[2, i].set_title(f"HSI Interp - ({WVL_PRS[idx]:.1f} nm)", fontsize=9)
            axes[2, i].axis("off")
            add_colorbar(fig,im2, axes[2, i])

            im3 = axes[3, i].imshow(canal_msi, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
            axes[3, i].set_title(f"{msi_label} -  ({WVL_S2[msi_idx]}nm)", fontsize=9)
            axes[3, i].axis("off")
            add_colorbar(fig,im3, axes[3, i])
        else:
            im2 = axes[2, i].imshow(canal_msi, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
            axes[2, i].set_title(f"{msi_label} -  ({WVL_S2*[msi_idx]}nm)", fontsize=9)
            axes[2, i].axis("off")
            add_colorbar(fig,im2, axes[2, i])

    output_plot_path = plot_dir / save_name
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")  
    plt.close()
    print(f"[Succès] Planche {msi_label} sauvegardée sous : {output_plot_path}")

def visualise_residual(data, channels, save_name, plot_dir):
    """
    Génère une planche comparative des résidus :
    Ligne 0 : Vérité Terrain (GT HSI)
    Ligne 1 : HSI interpolée
    Ligne 2 : Résidu prédit (HSI_pred - HSI_interp)
    Ligne 3 : Vrai résidu (GT - HSI_interp)
    """
    cube_gt = data["cube_gt"]
    cube_interp = data["cube_interp"]
    cube_res = data["cube_res"]
    msi_label = data["msi_label"]
    cube_true_res = cube_gt - cube_interp

    num_rows = 4 
    num_cols = len(channels)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), constrained_layout=True)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    title_str = f"Planche Comparative Résidu — MSE globale : {data['img_mse']:.4f} | MAE globale : {data['img_mae']:.4f} | SAM global : {data['img_sam']:.4f} rad"
    fig.suptitle(title_str, fontsize=14, y=1.02)

    for i, idx in enumerate(channels):
        canal_gt = cube_gt[:, :, idx]
        canal_interp = cube_interp[:, :, idx]
        canal_res = cube_res[:, :, idx]
        canal_true_res = cube_true_res[:, :, idx]
        
        res_mae = compute_mae(canal_true_res, canal_res)

        # Dynamique pour l'image brute
        vmin_val = min(canal_gt.min(), canal_interp.min())
        vmax_val = max(canal_gt.max(), canal_interp.max())

        # Dynamique symétrique centrée sur 0 pour les résidus (colormap seismic)
        v_res = max(np.max(np.abs(canal_res)), np.max(np.abs(canal_true_res)))
        v_res = max(v_res, 1e-5) # Évite une division par zéro si les résidus sont nuls

        # Ligne 0 : GT
        im0 = axes[0, i].imshow(canal_gt, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
        axes[0, i].set_title(f"GT HSI - ({WVL_PRS[idx]:.1f} nm)", fontsize=9)
        add_colorbar(fig, im0, axes[0, i])

        # Ligne 1 : Interpolée
        im1 = axes[1, i].imshow(canal_interp, cmap="turbo", vmin=vmin_val, vmax=vmax_val)
        axes[1, i].set_title(f"HSI interpolée - ({WVL_PRS[idx]:.1f} nm)", fontsize=9)
        add_colorbar(fig, im1, axes[1, i])

        # Ligne 2 : Résidu Prédit
        im2 = axes[2, i].imshow(canal_res, cmap="seismic", vmin=-v_res, vmax=v_res)
        axes[2, i].set_title(f"Résidu prédit\nMAE: {res_mae:.4f}", fontsize=9)
        add_colorbar(fig, im2, axes[2, i])

        # Ligne 3 : Vrai Résidu
        im3 = axes[3, i].imshow(canal_true_res, cmap="seismic", vmin=-v_res, vmax=v_res)
        axes[3, i].set_title(f"Vrai résidu", fontsize=9)
        add_colorbar(fig, im3, axes[3, i])
        
        for r in range(num_rows):
            axes[r, i].axis("off")

    output_plot_path = plot_dir / save_name
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")  
    plt.close()
    print(f"[Succès] Planche résidus sauvegardée sous : {output_plot_path}")


def heat_map_err_from_data(data, channel, save_name, plot_dir):
    """
    Affiche la carte d'erreur absolue pour une sélection de canaux + la carte SAM globale.
    """
    cube_gt = data["cube_gt"]
    cube_predict = data["cube_predict"]
    modele_name = data["model name"]

    fig, axes = plt.subplots(1, len(channel) + 1, figsize=(4 * (len(channel) + 1), 5), constrained_layout=True)
    if len(channel) == 0:
        return

    title_str = f"Heat Map {modele_name} — MSE: {data['img_mse']:.4f} | MAE: {data['img_mae']:.4f} | SAM: {data['img_sam']:.4f} rad"
    fig.suptitle(title_str, fontsize=12, y=1.05)

    # Calcul vectoriel pour borner l'erreur max de manière homogène sur la planche
    vmax_err = min(np.max(np.abs(cube_gt[:, :, channel] - cube_predict[:, :, channel])), 0.3)
    vmax_err = max(vmax_err, 1e-5)

    for i, idx in enumerate(channel):
        err = np.abs(cube_gt[:, :, idx] - cube_predict[:, :, idx])
        im0 = axes[i].imshow(err, cmap="inferno", vmax=vmax_err, vmin=0.0)
        axes[i].set_title(f"Error Map\n{WVL_PRS[idx]:.1f} nm", fontsize=9)
        axes[i].axis("off")
        add_colorbar(fig, im0, axes[i])

    # Dernière colonne : Carte SAM
    sam_map = compute_sam_map(cube_predict, cube_gt)
    im_sam = axes[-1].imshow(sam_map, cmap='turbo')
    axes[-1].set_title(f"SAM Map\nGlobal: {data['img_sam']:.4f}", fontsize=9)
    axes[-1].axis("off")
    add_colorbar(fig, im_sam, axes[-1])

    output_plot_path = plot_dir / save_name
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Succès] Map d'erreur sauvegardée sous : {output_plot_path}")


def compare_heat_map(data1, data2, channel, save_name, plot_dir):
    """
    Compare côte à côte les cartes d'erreur de deux modèles ainsi que leur différence brute.
    """
    fig, axes = plt.subplots(3, len(channel) + 1, figsize=(4 * (len(channel) + 1), 12), constrained_layout=True)
    
    title_str = f"Comparaison d'Erreur : {data1['model name']} vs {data2['model name']}"
    fig.suptitle(title_str, fontsize=14, y=1.02)

    for i, idx in enumerate(channel):
        err1 = np.abs(data1["cube_gt"][:, :, idx] - data1["cube_predict"][:, :, idx])
        err2 = np.abs(data2["cube_gt"][:, :, idx] - data2["cube_predict"][:, :, idx])
        diff = data1["cube_predict"][:, :, idx] - data2["cube_predict"][:, :, idx]
        
        vmax_err = min(max(np.max(err1), np.max(err2)), 0.3) 
        vmax_diff = min(np.max(np.abs(diff)), 0.5)
        vmax_diff = max(vmax_diff, 1e-5)

        im0 = axes[0, i].imshow(err1, cmap="inferno", vmax=vmax_err, vmin=0.0)
        axes[0, i].set_title(f"Err {data1['model name']}\n{WVL_PRS[idx]:.1f} nm", fontsize=8)

        im1 = axes[1, i].imshow(err2, cmap="inferno", vmax=vmax_err, vmin=0.0)
        axes[1, i].set_title(f"Err {data2['model name']}\n{WVL_PRS[idx]:.1f} nm", fontsize=8)

        im2 = axes[2, i].imshow(diff, cmap="seismic", vmax=vmax_diff, vmin=-vmax_diff)
        axes[2, i].set_title(f"Diff (M1 - M2)\n{WVL_PRS[idx]:.1f} nm", fontsize=8)

        for r in range(3):
            axes[r, i].axis("off")
            add_colorbar(fig, [im0, im1, im2][r], axes[r, i])

    # Colonne SAM de fin pour les deux modèles
    for r, data in enumerate([data1, data2]):
        sam_map = compute_sam_map(data["cube_predict"], data["cube_gt"])
        im_sam = axes[r, -1].imshow(sam_map, cmap='inferno')
        axes[r, -1].set_title(f"SAM {data['model name']}\nGlobal: {data['img_sam']:.4f}", fontsize=9)
        axes[r, -1].axis("off")
        add_colorbar(fig, im_sam, axes[r, -1])

    axes[2, -1].axis("off")  

    output_plot_path = plot_dir / save_name
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Succès] Comparaison Heatmap sauvegardée sous : {output_plot_path}")


def compare_predictions_line(data1, data2, channel, save_name, plot_dir):
    """
    Planche comparative directe des profils reconstruits : GT vs Modèle 1 vs Modèle 2.
    """
    cube_gt = data1["cube_gt"] 
    fig, axes = plt.subplots(3, len(channel) + 1, figsize=(4 * (len(channel) + 1), 10), constrained_layout=True)

    title_str = f"Planche Profils : Ground Truth vs {data1['model name']} vs {data2['model name']}"
    fig.suptitle(title_str, fontsize=14, y=1.02)

    for i, idx in enumerate(channel):
        canal_gt = cube_gt[:, :, idx]
        canal_pred1 = data1["cube_predict"][:, :, idx]
        canal_pred2 = data2["cube_predict"][:, :, idx]
        
        vmin_col = min(np.percentile(canal_gt, 0.1), np.percentile(canal_pred1, 0.1), np.percentile(canal_pred2, 0.1))
        vmax_col = max(np.percentile(canal_gt, 99.9), np.percentile(canal_pred1, 99.9), np.percentile(canal_pred2, 99.9))
        
        wavelength = WVL_PRS[idx]
        # Filtrage des bandes d'absorption atmosphérique pour l'affichage dynamique
        is_outside_absorption = wavelength < 1350 or (1500 < wavelength < 1850) or (wavelength > 2000)
        v_min, v_max = (vmin_col, vmax_col) if is_outside_absorption else (0.0, 0.3)

        im0 = axes[0, i].imshow(canal_gt, cmap="turbo", vmin=v_min, vmax=v_max)
        axes[0, i].set_title(f"GT | {wavelength:.1f} nm", fontsize=9)

        im1 = axes[1, i].imshow(canal_pred1, cmap="turbo", vmin=v_min, vmax=v_max)
        axes[1, i].set_title(data1["model name"], fontsize=9)

        im2 = axes[2, i].imshow(canal_pred2, cmap="turbo", vmin=v_min, vmax=v_max)
        axes[2, i].set_title(data2["model name"], fontsize=9)

        for r in range(3):
            axes[r, i].axis("off")
            add_colorbar(fig, [im0, im1, im2][r], axes[r, i])

    # Ajout des cartes SAM comparatives sur la dernière colonne
    for r, data in enumerate([data1, data2]):
        sam_map = compute_sam_map(data["cube_predict"], data["cube_gt"])
        im_sam = axes[r + 1, -1].imshow(sam_map, cmap='inferno')
        axes[r + 1, -1].set_title(f"SAM {data['model name']}\nGlobal: {data['img_sam']:.4f}", fontsize=9)
        axes[r + 1, -1].axis("off")
        add_colorbar(fig, im_sam, axes[r + 1, -1])

    axes[0, -1].axis("off") 

    output_plot_path = plot_dir / save_name
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Succès] Planche comparative linéaire sauvegardée sous : {output_plot_path}")

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
        chan_mae = compute_mae(canal_gt, canal_pred)

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

    title_str = f"Planche Comparative —\n MSE globale : {img_mse:.4f} | MAE globale : {img_mae:.4f} | SAM global : {img_sam:.4f} rad | SSIM : {img_ssim:.4f} |ERGAS : {img_ergas:.2e}"
    fig.suptitle(title_str, fontsize=14, y=1.02)  

    for i, idx in enumerate(channels):
        canal_gt = cube_gt[:, :, idx]
        canal_pred = cube_predict[:, :, idx]
        canal_msi=cube_msi[:,:,i]
        chan_mae = compute_mae(canal_gt, canal_pred)

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
        axes[2, i].set_title(f"MSI - ({WVL_S2*[i]:.1f} nm)", fontsize=9)
        axes[2, i].axis("off")
        add_colorbar(fig, im2, axes[2, i])

    output_plot_path = plot_dir / f"{save_name}.png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")  
    plt.close()
    print(f"[Succès] Planche sauvegardée sous : {output_plot_path}")





def visualise_synthesis(
    data, save_name, plot_dir, kept_indices=None, log_to_wandb=False
):
    """Génère une planche de synthèse 2x3 avec gestion des bandes supprimées (NaN)."""
    
    cube_gt = data["cube_gt"]  # Shape (H, W, n_kept)
    cube_predict = data["cube_predict"]  # Shape (H, W, n_kept)
    cube_msi = data["cube_msi"]  # Shape (H, W, n_msi)
    model_name = data.get("model name", "Modèle ML")

    # Récupération ou calcul automatique des métriques
    img_rmse = data.get("img_rmse", compute_rmse(cube_predict, cube_gt))
    img_mae = data.get("img_mae", compute_mae(cube_predict, cube_gt))
    img_sam = data.get("img_sam", compute_sam(cube_gt, cube_predict))
    img_ssim = data.get("img_ssim", compute_ssim_multiband(cube_predict, cube_gt))
    img_ergas = data.get("img_ergas", compute_ergas(cube_predict, cube_gt))
    img_psnr = data.get("img_psnr", compute_psnr(cube_gt, cube_predict))

    n_kept_bands = cube_gt.shape[-1]

    # 1. Reconstitution du spectre d'erreur complet (avec NaN sur les bandes retirées)
    if kept_indices is None:
        wvl_kept = np.ones(len(WVL_PRS), dtype=bool) if WVL_PRS is not None else np.arange(n_kept_bands)
        mae_full_spectrum = np.mean(np.abs(cube_gt - cube_predict), axis=(0, 1))
    else:
        mae_per_kept_band = np.mean(np.abs(cube_gt - cube_predict), axis=(0, 1))
        mae_full_spectrum = np.full(len(WVL_PRS), np.nan)
        mae_full_spectrum[kept_indices] = mae_per_kept_band
        wvl_kept = WVL_PRS[kept_indices]

    # 2. Indices RGB HSI basés sur les longueurs d'onde conservées

    idx_r_hsi = np.argmin(np.abs(wvl_kept - 665.0))
    idx_v_hsi = np.argmin(np.abs(wvl_kept - 560.0))
    idx_b_hsi = np.argmin(np.abs(wvl_kept - 490.0))

    idx_r_msi, idx_v_msi, idx_b_msi = 3, 2, 1

    # Fonction d'affichage RGB
    def to_rgb(cube, r, g, b, bounds=None):
        rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
        rgb = np.nan_to_num(rgb)
        for c in range(3):
            if bounds is None:
                v_min, v_max = np.percentile(rgb[:, :, c], 2), np.percentile(rgb[:, :, c], 98)
            else:
                v_min, v_max = bounds[c]
            rgb[:, :, c] = np.clip((rgb[:, :, c] - v_min) / (v_max - v_min + 1e-8), 0, 1)
        return rgb

    rgb_msi = to_rgb(cube_msi, idx_r_msi, idx_v_msi, idx_b_msi)

    # Calcul des bornes communes HSI
    bounds_hsi = []
    for chan in [idx_r_hsi, idx_v_hsi, idx_b_hsi]:
        v_min = min(np.percentile(cube_gt[:, :, chan], 2), np.percentile(cube_predict[:, :, chan], 2))
        v_max = max(np.percentile(cube_gt[:, :, chan], 98), np.percentile(cube_predict[:, :, chan], 98))
        bounds_hsi.append((v_min, v_max))

    rgb_pred = to_rgb(cube_predict, idx_r_hsi, idx_v_hsi, idx_b_hsi, bounds=bounds_hsi)
    rgb_gt = to_rgb(cube_gt, idx_r_hsi, idx_v_hsi, idx_b_hsi, bounds=bounds_hsi)

    # Diagnostics spatiaux (MAE et SAM)
    mae_map = np.mean(np.abs(cube_gt - cube_predict), axis=-1)

    dot = np.sum(cube_predict * cube_gt, axis=-1)
    norm_p = np.linalg.norm(cube_predict, axis=-1)
    norm_g = np.linalg.norm(cube_gt, axis=-1)
    sam_map = np.arccos(np.clip(dot / (norm_p * norm_g + 1e-8), -1.0, 1.0))

    # FIGURE 2x3
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    clean_title = model_name.replace("\n", " ")
    
    # TITRE GLOBAL AVEC RMSE À LA PLACE DE MSE
    fig.suptitle(
        f"Planche de Synthèse Globale — {clean_title}\n"
        f"RMSE : {img_rmse:.4f} | MAE : {img_mae:.4f} | SAM : {img_sam:.4f} rad "
        f"({np.degrees(img_sam):.2f}°) | PSNR : {img_psnr:.2f} dB | SSIM : {img_ssim:.4f} | ERGAS : {img_ergas:.2f}",
        fontsize=14,
        weight="bold",
    )

    # LIGNE 1 : Images RGB
    axes[0, 0].imshow(rgb_msi)
    axes[0, 0].set_title(f"RGB — Entrée MSI (Sentinel-2)\n Min: {np.min(rgb_msi):.2f} | Max: {np.max(rgb_msi):.2f} | Mean: {np.mean(rgb_msi):.2f}", fontsize=11, pad=6)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_pred)
    axes[0, 1].set_title(f"RGB — HSI Prédite ({n_kept_bands} bandes)\n Min: {np.min(rgb_pred):.2f} | Max: {np.max(rgb_pred):.2f} | Mean: {np.mean(rgb_pred):.2f}", fontsize=11, pad=6)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb_gt)
    axes[0, 2].set_title(f"RGB — HSI Vérité Terrain ({n_kept_bands} bandes)\n Min: {np.min(rgb_gt):.2f} | Max: {np.max(rgb_gt):.2f} | Mean: {np.mean(rgb_gt):.2f}", fontsize=11, pad=6)
    axes[0, 2].axis("off")

    # LIGNE 2 : Cartes d'Erreurs et Spectre
    vmax_mae = max(np.percentile(mae_map, 98), 0.02)
    im_mae = axes[1, 0].imshow(mae_map, cmap="inferno", vmin=0, vmax=vmax_mae)
    axes[1, 0].set_title(f"Carte d'Erreur MAE Spatiale\n(Moyenne : {img_mae:.4f})", fontsize=11, pad=6)
    axes[1, 0].axis("off")
    fig.colorbar(im_mae, ax=axes[1, 0], fraction=0.046, pad=0.04)

    vmax_sam = max(np.percentile(sam_map, 98), 0.05)
    im_sam = axes[1, 1].imshow(sam_map, cmap="inferno", vmin=0, vmax=vmax_sam)
    axes[1, 1].set_title(f"Carte de Distorsion SAM\n(Moyenne : {img_sam:.4f} rad / {np.degrees(img_sam):.2f}°)", fontsize=11, pad=6)
    axes[1, 1].axis("off")
    fig.colorbar(im_sam, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Graphique spectral
    ax_spec = axes[1, 2]
    x_axis_vals = WVL_PRS if WVL_PRS is not None else np.arange(len(mae_full_spectrum))
    ax_spec.plot(x_axis_vals, mae_full_spectrum, color="crimson", lw=2, label="MAE par bande")
    ax_spec.set_title("Profil de l'Erreur Spectrale (MAE par Longueur d'Onde)", fontsize=11, pad=6)
    ax_spec.set_xlabel("Longueur d'onde (nm)", fontsize=9)
    ax_spec.set_ylabel("MAE", fontsize=9)
    ax_spec.grid(True, linestyle=":", alpha=0.6)

    # Masquage grisé des fenêtres d'absorption H2O (si WVL_PRS existe)
    if WVL_PRS is not None:
        ax_spec.axvspan(1350, 1500, color="gray", alpha=0.2, label="Abs. H₂O")
        ax_spec.axvspan(1800, 2000, color="gray", alpha=0.2)
    ax_spec.legend(fontsize=9, loc="upper right")

    # Sauvegarde locale
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_plot_path = plot_dir / f"{save_name}.png"
        plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")

    # Envoi sur WandB
    if log_to_wandb and wandb is not None and wandb.run is not None:
        wandb.log({f"synthesis_plots/{save_name}": wandb.Image(fig)})

    plt.close(fig)
    
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
    img_sam1 = compute_sam(cube_predict1, cube_gt)
    im_sam1 = axes[1, -1].imshow(sam1, cmap='inferno')
    axes[1, -1].set_title(f"SAM map \n SAM global : {img_sam1:.4f}", fontsize=9)
    axes[1, -1].axis("off")
    add_colorbar(fig, im_sam1, axes[1, -1])

    sam2 = compute_sam_map(cube_predict2, cube_gt) 
    img_sam2 = compute_sam(cube_predict2, cube_gt)
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

def visualise_zoom(data, save_name, plot_dir, bbox=None):
    """
    Affiche une vue comparative en 5 panneaux :
    MSI | HSI (GT) | HSI PRED | Différence (Pred vs GT) | Différence (Baseline / Autre)
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    cube_gt = data["cube_gt"]              # [H, W, C] (HR HSI)
    cube_pred = data["cube_predict"]       # [H, W, C] (HR HSI Prédit)
    cube_msi = data.get("cube_msi", None)  # [H, W, C_msi] (Guide MSI)
    cube_base = data.get("cube_baseline", data.get("cube_bicubic", cube_gt * 0.95)) # Baseline de secours
    model_name = data.get("model_name", "Modèle")
    
    # Découpage spatial optionnel (Zoom)
    if bbox is not None:
        ymin, ymax, xmin, xmax = bbox
        gt_z = cube_gt[ymin:ymax, xmin:xmax, :]
        pred_z = cube_pred[ymin:ymax, xmin:xmax, :]
        msi_z = cube_msi[ymin:ymax, xmin:xmax, :] if cube_msi is not None else None
        base_z = cube_base[ymin:ymax, xmin:xmax, :]
    else:
        gt_z, pred_z, msi_z, base_z = cube_gt, cube_pred, cube_msi, cube_base

    # Figure à 5 sous-graphiques
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)
    fig.suptitle(f"Comparaison Multi-modale & Résiduelles — {model_name}", fontsize=14, fontweight='bold')
    
    # Indices RGB pour HSI (PRISMA via longueurs d'onde)
    idx_r_hsi = np.argmin(np.abs(WVL_PRS - 665.0))
    idx_v_hsi = np.argmin(np.abs(WVL_PRS - 560.0))
    idx_b_hsi = np.argmin(np.abs(WVL_PRS - 490.0))
        
    # Indices RGB pour Sentinel-2 (MSI : B4=Rouge, B3=Vert, B2=Bleu -> idx 3, 2, 1)
    idx_r_msi, idx_v_msi, idx_b_msi = 3, 2, 1
        
    # Fonction interne de conversion RGB robuste (percentiles 2-98)
    def to_rgb(cube, r, g, b):
        if cube is None:
            return np.zeros((*gt_z.shape[:2], 3))
        rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
        rgb = np.nan_to_num(rgb)  # Sécurité pour les masques/NaN
        for c in range(3):
            v_min = np.percentile(rgb[:, :, c], 2)
            v_max = np.percentile(rgb[:, :, c], 98)
            rgb[:, :, c] = np.clip((rgb[:, :, c] - v_min) / (v_max - v_min + 1e-8), 0, 1)
        return rgb
        
    # Génération des images RVB
    img_msi = to_rgb(msi_z, idx_r_msi, idx_v_msi, idx_b_msi)
    img_gt = to_rgb(gt_z, idx_r_hsi, idx_v_hsi, idx_b_hsi)
    img_pred = to_rgb(pred_z, idx_r_hsi, idx_v_hsi, idx_b_hsi)
    
    # Cartes d'erreur (MAE moyenne sur les canaux)
    diff_pred = np.mean(np.abs(pred_z - gt_z), axis=-1)
    diff_base = np.mean(np.abs(base_z - gt_z), axis=-1)

    # Affichage des 5 panneaux
    axes[0].imshow(img_msi)
    axes[0].set_title("1. MSI", fontsize=10)
        
    axes[1].imshow(img_gt)
    axes[1].set_title("2. HSI (GT)", fontsize=10)
        
    axes[2].imshow(img_pred)
    axes[2].set_title("3. HSI PRED", fontsize=10)
        
    im3 = axes[3].imshow(diff_pred, cmap="inferno")
    axes[3].set_title("4. Différence (PRED - GT)", fontsize=10)
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
    im4 = axes[4].imshow(diff_base, cmap="inferno")
    axes[4].set_title("5. Différence (Baseline - GT)", fontsize=10)
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
        
    for ax in axes:
        ax.axis('off')
        
    output_path = plot_dir / f"Comparison_5_Panels_{save_name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Succès] Planche 5 panneaux sauvegardée : {output_path}")
