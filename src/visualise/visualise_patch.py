

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
import xarray as xr
# Sécurité pour les serveurs sans GUI
matplotlib.use('Agg')
from src.constants import WVL_PRS, DW_INFO,WVL_S2,DATA_DIR,ROOT_DIR

def to_rgb(cube, r, g, b, bounds=None):
    """Génère une image RGB normalisée canal par canal avec gestion des NaN."""
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)
    rgb = np.nan_to_num(rgb)
    rgb_norm = np.zeros_like(rgb, dtype=np.float32)

    for c in range(3):
        if bounds is None:
            v_min, v_max = np.percentile(rgb[:, :, c], 2), np.percentile(
                rgb[:, :, c], 98
            )
        else:
            v_min, v_max = bounds[c]

        denom = v_max - v_min + 1e-8
        rgb_norm[:, :, c] = np.clip((rgb[:, :, c] - v_min) / denom, 0, 1)

    return rgb_norm


def decouper_et_sauvegarder_hsi(
    image_input,
    patch_size=256,
    var_name="sr",
    wvl=None,
    target_wvls=(665.0, 560.0, 490.0),
    rgb_bands=None,
    bounds=None,
    output_path="hsi_scene_patches.png",
    show_labels=True,
    dpi=300,
):
    """Découpe un cube HSI/MSI en patches, génère un rendu RGB fidèle

    et sauvegarde l'image globale avec la grille des limites de patches.

    Parameters:
    -----------
    image_input : str, xr.Dataset, ou np.ndarray
        Cube de données hyperspectral/multispectral.
    patch_size : int
        Taille de découpe (hauteur et largeur en pixels).
    var_name : str
        Nom de la variable dans le Dataset xarray (ex: 'sr').
    wvl : np.ndarray ou list, optionnel
        Tableau des longueurs d'onde (en nm). Si présent dans le dataset,
        il est automatiquement extrait.
    target_wvls : tuple de 3 float
        Longueurs d'onde cibles (R, G, B) en nm (ex: 665, 560, 490 nm).
    rgb_bands : tuple de 3 int, optionnel
        Indices directs des bandes (R, G, B) si les longueurs d'onde ne sont pas utilisées.
    bounds : list of tuple [(min, max), ...], optionnel
        Limites de normalisation prédéfinies pour les 3 canaux R, G, B.
    output_path : str
        Chemin du fichier PNG/JPG de sortie.
    """
    # 1. Chargement des données et des longueurs d'onde
    if isinstance(image_input, str):
        with xr.open_dataset(image_input) as ds:
            data = ds[var_name].values
            if wvl is None and "wavelength" in ds:
                wvl = ds["wavelength"].values
            elif wvl is None and "wvl" in ds:
                wvl = ds["wvl"].values
    elif isinstance(image_input, xr.Dataset):
        data = image_input[var_name].values
        if wvl is None and "wavelength" in image_input:
            wvl = image_input["wavelength"].values
        elif wvl is None and "wvl" in image_input:
            wvl = image_input["wvl"].values
    else:
        data = np.array(image_input)

    # Réorganisation en (H, W, C) si l'entrée est en (C, H, W)
    if (
        data.ndim == 3
        and data.shape[0] < data.shape[1]
        and data.shape[0] < data.shape[2]
    ):
        data = np.moveaxis(data, 0, -1)

    h, w, c = data.shape

    # 2. Détermination des indices des canaux R, G, B
    if rgb_bands is not None:
        idx_r, idx_g, idx_b = rgb_bands
    elif wvl is not None:
        wvl_arr = np.array(wvl)
        idx_r = int(np.argmin(np.abs(wvl_arr - target_wvls[0])))
        idx_g = int(np.argmin(np.abs(wvl_arr - target_wvls[1])))
        idx_b = int(np.argmin(np.abs(wvl_arr - target_wvls[2])))
        print(
            f"Bandes sélectionnées par wavelength: R={idx_r} ({wvl_arr[idx_r]:.1f}nm), "
            f"G={idx_g} ({wvl_arr[idx_g]:.1f}nm), B={idx_b} ({wvl_arr[idx_b]:.1f}nm)"
        )
    else:
        # Fallback par défaut (indices 3, 2, 1)
        idx_r, idx_g, idx_b = 3, 2, 1

    # 3. Rognage aux dimensions multiples de patch_size
    h_crop = h - (h % patch_size)
    w_crop = w - (w % patch_size)
    data_cropped = data[:h_crop, :w_crop, :]

    n_rows = h_crop // patch_size
    n_cols = w_crop // patch_size

    # 4. Extraire la liste des patches
    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            patch = data_cropped[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
                :,
            ]
            patches.append(patch)

    # 5. Rendu RGB via la fonction to_rgb
    rgb_scene = to_rgb(data_cropped, idx_r, idx_g, idx_b, bounds=bounds)

    # 6. Tracé du résultat et superposition des limites de patches
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb_scene)

    # Lignes de séparation de la grille
    for j in range(n_cols + 1):
        ax.axvline(
            j * patch_size, color="red", linestyle="--", linewidth=1.2, alpha=0.8
        )

    for i in range(n_rows + 1):
        ax.axhline(
            i * patch_size, color="red", linestyle="--", linewidth=1.2, alpha=0.8
        )

    # Numérotation au centre
    if show_labels:
        for i in range(n_rows):
            for j in range(n_cols):
                patch_idx = i * n_cols + j
                cx = j * patch_size + (patch_size / 2)
                cy = i * patch_size + (patch_size / 2)

                ax.text(
                    cx,
                    cy,
                    f"{patch_idx:04d}",
                    color="yellow",
                    fontsize=7,
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="black",
                        alpha=0.6,
                        edgecolor="none",
                    ),
                )

    ax.set_title(
        f"Scène HSI RGB ({h_crop}x{w_crop} px) — {len(patches)} patches\n"
        f"Canaux R:{idx_r}, G:{idx_g}, B:{idx_b}"
    )
    ax.axis("off")
    plt.tight_layout()

    # 7. Sauvegarde du fichier
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Vue de la scène sauvegardée sous : {output_path}")

    plt.show()

    return patches

import os
from pathlib import Path

# --- Configuration des listes de scènes ---
TEST_SCENES = [
    "baltijsk",
    "camerino",
    "codigoro",
    "copenhagen",
    "cullivel",
    "jagersfontein",
    "kirtland",
    "lorca",
]
VAL_SCENES = [
    "arborea",
    "athens",
    "beer_sheva",
    "istanbul",
    "los_cabos",
    "taiwan",
    "yuen_long",
]
TRAIN_SCENES = [
    "aranjuez",
    "bari",
    "beheira",
    "beirut",
    "belgrade",
    "binh_dai",
    "brasilia",
    "cape_town",
    "copperton",
    "cukotka",
    "dellys",
    "dubai",
    "dublin",
    "elsalto",
    "eyjafjoll",
    "fontainebleau",
    "fukushima",
    "guantanamo",
    "hanging_rock",
    "java",
    "jordan",
    "kitami",
    "lagos",
    "london",
    "los_angeles",
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
    "tampa_bay",
    "tientsin",
    "tijuana",
    "tirana",
    "valencia",
]

ALL_SCENES = TRAIN_SCENES + TEST_SCENES + VAL_SCENES

# --- Paramètres de dossiers ---

OUTPUT_DIR = ROOT_DIR /"data"/ "images_rgb_patch"

if __name__ == "__main__":
    # Création du dossier de sortie s'il n'existe pas
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timepoints = ["before", "after"]

    for scene in ALL_SCENES:
        for tp in timepoints:
            file_name = f"{scene}-{tp}-s2.nc"
            file_path = DATA_DIR /f"{scene}"/file_name

            # Vérification de l'existence du fichier NetCDF
            if not file_path.exists():
                print(f"[SMILE WARNING] Fichier manquant ignoré : {file_path}")
                continue

            # Nom du fichier image de sortie (ex: DATA_DIR/images_rgb_patch/paris_before_grid.png)
            output_image_path = OUTPUT_DIR / f"{scene}_{tp}_msi_grid.png"

            print(f"Traitement de : {file_name} ...")

            try:
                patches = decouper_et_sauvegarder_hsi(
                    image_input=str(file_path),
                    patch_size=256,
                    var_name="sr",
                    wvl=WVL_S2,  # Longueurs d'onde du capteur Sentinel-2cc
                    target_wvls=(665.0, 560.0, 490.0),  # Longueurs d'onde R, G, B
                    output_path=str(output_image_path),
                    show_labels=True,
                    dpi=200,  # Bonne résolution pour consulter la grille
                )
                print(f" -> Image sauvegardée dans {output_image_path}\n")

            except Exception as e:
                print(f"[ERREUR] Échec du traitement de {file_name}: {e}\n")