
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

# Sécurité pour les serveurs sans GUI
matplotlib.use('Agg')
from constants import WVL_PRS, DW_INFO,WVL_S2
from metrics import compute_sam_map,compute_mae,compute_ergas,compute_mrae,compute_ssim_multiband,compute_mse,compute_sam
CHANNELS_ATM = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 200, 220]
CHANNELS_STD = [5, 11, 20, 32, 36, 40, 44, 50, 52, 59, 122, 187]
CHANNELS_ONLY_ATM = [98, 101, 104, 107, 110, 113, 116, 119, 122, 125]



# Exemple de ce que tu peux faire dans ta fonction add_colorbar :
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

def visualise_curve(mse_path, var_path, name_curve):
    mse_per_channel = np.load(mse_path)
    var_per_channel = np.load(var_path)
 
    plt.figure(figsize=(10, 5))
    plt.axvspan(1350, 1500, color='red', alpha=0.3, label="Atmospheric Absorption")
    plt.axvspan(1800, 2000, color='red', alpha=0.3)
 
    plt.plot(WVL_PRS, mse_per_channel, color="royalblue", linewidth=2, label="Mean of MSE SAM : rad   ")
    plt.plot(WVL_PRS, mse_per_channel + var_per_channel, color="royalblue", linestyle="--", alpha=0.5)
 
    plt.title("Reconstruction error by each channel")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("MSE")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(name_curve, dpi=300, bbox_inches="ededededed")
    plt.close()
    print(f"Graphique d'erreur sauvegardé : {name_curve}")
 
 
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
    axes[1, 1].set_title(f"Carte de Distorsion SAM\n(Moyenne globale : {data['img_sam']:.4f} rad)", fontsize=11, pad=6)
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
        
        res_mae = compute_mae(canal_true_res, canal_res)


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
    img_sam1 = compute_sam(cube_predict1, cube_gt1)
    im_sam1 = axes[0, -1].imshow(sam1, cmap='inferno')
    axes[0, -1].set_title(f"SAM map \n SAM global : {img_sam1:.4f}", fontsize=9)
    axes[0, -1].axis("off")
    add_colorbar(fig, im_sam1, axes[0, -1])

    sam2 = compute_sam_map(cube_predict2, cube_gt1) 
    img_sam2 = compute_sam(cube_predict2, cube_gt1)
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

