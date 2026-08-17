import sys
import time
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import CACHE_DIR, DATA_DIR, INTERP_MATRIX, SRF_MATRIX
from src.prepare_data.dataset import SpectralDataset

import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import xarray as xr

# Vos imports de constantes
from src.constants import DATA_DIR, INTERP_MATRIX, SRF_MATRIX

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import DATA_DIR, INTERP_MATRIX, SRF_MATRIX


def _process_single_scene(args):
    """Fonction exécutée en parallèle pour une scène donnée."""
    (
        x_file,
        y_file,
        base_name,
        patch_list,
        output_path,
        shape_msi,
        shape_hsi,
        c_hsi,
        c_multi,
        patch_size,
        srf_matrix,
        interp_matrix,
    ) = args

    # 1. Chargement complet de la scène
    with xr.open_dataset(x_file) as ds_x, xr.open_dataset(y_file) as ds_y:
        patch_multi_real_full = ds_x["sr"].to_numpy().astype(np.float32)
        patch_hyper_real_full = ds_y["sr"].to_numpy().astype(np.float32)

    h, w, _ = patch_hyper_real_full.shape

    # 2. Calculs spectraux sur toute la scène
    # MSI SIMULÉ GLOBAL
    hyper_2d = patch_hyper_real_full.reshape(-1, c_hsi)
    scene_multi_sim = (
        np.dot(hyper_2d, srf_matrix).reshape(h, w, c_multi).astype(np.float32)
    )

    # MSI NORMALISÉ GLOBAL
    mean_scene = np.mean(patch_multi_real_full, axis=(0, 1), keepdims=True)
    std_scene = np.std(patch_multi_real_full, axis=(0, 1), keepdims=True)
    scene_multi_norm = (patch_multi_real_full - mean_scene) / (
        std_scene + 1e-8
    )

    std_sim = np.std(scene_multi_sim, axis=(0, 1), keepdims=True)
    mean_sim = np.mean(scene_multi_sim, axis=(0, 1), keepdims=True)
    scene_multi_norm = scene_multi_norm * std_sim + mean_sim

    # HSI INTERPOLÉ GLOBAL
    m_real_chw_full = np.transpose(patch_multi_real_full, (2, 0, 1))
    interp_numpy = interp_matrix @ m_real_chw_full.reshape(c_multi, -1)
    scene_interp_chw = interp_numpy.reshape(c_hsi, h, w).astype(np.float32)

    # Transposition en format (C, H, W)
    scene_m_real_chw = m_real_chw_full
    scene_m_sim_chw = np.transpose(scene_multi_sim, (2, 0, 1))
    scene_m_norm_chw = np.transpose(scene_multi_norm, (2, 0, 1))
    scene_hsi_chw = np.transpose(patch_hyper_real_full, (2, 0, 1))

    # 3. Ouverture des memmaps en mode modification 'r+' pour écriture concurrente
    fp_msi_real = np.memmap(
        output_path / "msi_real.bin", dtype="float32", mode="r+", shape=shape_msi
    )
    fp_msi_sim = np.memmap(
        output_path / "msi_simulated.bin",
        dtype="float32",
        mode="r+",
        shape=shape_msi,
    )
    fp_msi_norm = np.memmap(
        output_path / "msi_normalised.bin",
        dtype="float32",
        mode="r+",
        shape=shape_msi,
    )
    fp_hsi = np.memmap(
        output_path / "hsi.bin", dtype="float32", mode="r+", shape=shape_hsi
    )
    fp_interp = np.memmap(
        output_path / "hsi_interp.bin",
        dtype="float32",
        mode="r+",
        shape=shape_hsi,
    )

    scene_results = []

    # Écriture des patchs aux indices globaux pré-attribués
    for global_idx, r, c, patch_id_str in patch_list:
        fp_msi_real[global_idx] = scene_m_real_chw[
            :, r : r + patch_size, c : c + patch_size
        ]
        fp_msi_sim[global_idx] = scene_m_sim_chw[
            :, r : r + patch_size, c : c + patch_size
        ]
        fp_msi_norm[global_idx] = scene_m_norm_chw[
            :, r : r + patch_size, c : c + patch_size
        ]
        fp_hsi[global_idx] = scene_hsi_chw[
            :, r : r + patch_size, c : c + patch_size
        ]
        fp_interp[global_idx] = scene_interp_chw[
            :, r : r + patch_size, c : c + patch_size
        ]

        scene_results.append((global_idx, patch_id_str))

    # Validation de l'écriture sur disque pour ce worker
    fp_msi_real.flush()
    fp_msi_sim.flush()
    fp_msi_norm.flush()
    fp_hsi.flush()
    fp_interp.flush()

    return scene_results


def prepare_mumucd_single_file(
    scene_ids,
    split_name,
    blacklist_patches=None,
    patch_size=256,
    output_dir="cache/all_data_memmap",
    num_workers=16,
):
    if blacklist_patches is None:
        blacklist_patches = set()
    else:
        blacklist_patches = set(blacklist_patches)

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)

    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)

    scene_pairs = []
    for item in scene_ids:
        scene_name = item.split("-")[0]
        hsi_file = DATA_DIR / scene_name / f"{item}-prs.nc"
        msi_file = DATA_DIR / scene_name / f"{item}-s2.nc"

        if hsi_file.exists() and msi_file.exists():
            scene_pairs.append((msi_file, hsi_file, item))
        else:
            print(f"⚠️ Fichiers introuvables pour {item}, ignoré.")

    print(f"Analyse préliminaire sur {len(scene_pairs)} scènes/items valides...")

    c_hsi = INTERP_MATRIX.shape[0]
    c_multi = SRF_MATRIX.shape[1]

    # Construction des tâches par scène et attribution des indices globaux
    scene_tasks = []
    global_patch_idx = 0

    for x_file, y_file, base_name in scene_pairs:
        with xr.open_dataset(y_file) as ds_y:
            h, w, _ = ds_y["sr"].shape
            h_crop = h - (h % patch_size)
            w_crop = w - (w % patch_size)

        current_scene_patches = []
        for i in range(h_crop // patch_size):
            for j in range(w_crop // patch_size):
                patch_idx = i * (w_crop // patch_size) + j
                patch_id_str = f"{base_name}_patch_{patch_idx:04d}"

                if patch_id_str in blacklist_patches:
                    print(f"{patch_id_str} est dans la liste noire, ignoré.")
                    continue

                current_scene_patches.append((
                    global_patch_idx,
                    i * patch_size,
                    j * patch_size,
                    patch_id_str,
                ))
                global_patch_idx += 1

        if current_scene_patches:
            scene_tasks.append((x_file, y_file, base_name, current_scene_patches))

    total_patches_global = global_patch_idx
    if total_patches_global == 0:
        print("❌ Aucun patch valide à traiter.")
        return

    shape_msi = (total_patches_global, c_multi, patch_size, patch_size)
    shape_hsi = (total_patches_global, c_hsi, patch_size, patch_size)

    print(
        f"🚀 Pre-allocation des 5 fichiers memmap pour {total_patches_global} patches..."
    )

    # Pre-création / allocation des fichiers binaires vides sur disque (mode w+)
    for name, shape in [
        ("msi_real.bin", shape_msi),
        ("msi_simulated.bin", shape_msi),
        ("msi_normalised.bin", shape_msi),
        ("hsi.bin", shape_hsi),
        ("hsi_interp.bin", shape_hsi),
    ]:
        fp = np.memmap(
            output_path / name, dtype="float32", mode="w+", shape=shape
        )
        fp.flush()
        del fp  # Libère le descripteur de fichier pour les workers

    # Préparation des arguments pour multiprocessing
    worker_args = [
        (
            x_file,
            y_file,
            base_name,
            patch_list,
            output_path,
            shape_msi,
            shape_hsi,
            c_hsi,
            c_multi,
            patch_size,
            SRF_MATRIX,
            INTERP_MATRIX,
        )
        for x_file, y_file, base_name, patch_list in scene_tasks
    ]

    print(f"🔥 Traitement parallèle sur {num_workers} cœurs CPU...")
    all_indexed_patch_ids = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_process_single_scene, arg) for arg in worker_args
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Progression scènes",
        ):
            scene_results = future.result()
            all_indexed_patch_ids.extend(scene_results)

    # Tri pour garantir l'ordre exact de patch_ids dans le JSON
    all_indexed_patch_ids.sort(key=lambda x: x[0])
    ordered_patch_ids = [p_id for _, p_id in all_indexed_patch_ids]

    # Sauvegarde du manifest JSON
    metadata = {
        "total_patches": total_patches_global,
        "msi_shape": list(shape_msi),
        "hsi_shape": list(shape_hsi),
        "dtype": "float32",
        "patch_ids": ordered_patch_ids,
    }

    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Génération parallèle des 5 fichiers memmap terminée avec succès !")
from torch.utils.data import DataLoader


from torch.utils.data import DataLoader
from pathlib import Path
 
# Assume CACHE_DIR et SpectralDataset sont définis ailleurs
# CACHE_DIR = Path("cache/all_data")
# from spectral_dataset_fixed import SpectralDataset
 
from torch.utils.data import DataLoader


def create_data_loaders_spectral(
    train_dir,
    val_dir,
    test_dir,
    simulated=False,
    augment=False,
    augment_illumination=False,
    batch_size=8,
    num_workers=4,
    is_residual=False,
    is_normalised=False,
    kept_indices=None,
):
  """Crée et renvoie les DataLoaders (train, val, test) pour les données spectrales.

  Compatible memmap : num_workers peut être augmenté librement pour maximiser le
  débit vers le GPU.
  """

  train_dataset = SpectralDataset(
      dataset_dir=train_dir,
      simulated=simulated,
      is_normalised=is_normalised,
      augment=augment,
      augment_illumination=augment_illumination,
      is_residual=is_residual,
      kept_indices=kept_indices,
  )

  val_dataset = SpectralDataset(
      dataset_dir=val_dir,
      simulated=simulated,
      is_normalised=is_normalised,
      augment=False,
      augment_illumination=False,
      is_residual=is_residual,
      kept_indices=kept_indices,
  )

  test_dataset = SpectralDataset(
      dataset_dir=test_dir,
      simulated=simulated,
      is_normalised=is_normalised,
      augment=False,
      augment_illumination=False,
      is_residual=is_residual,
      kept_indices=kept_indices,
  )

  # Configuration optimisée pour memmap (multi-worker débloqué)
  loader_kwargs = {
      "batch_size": batch_size,
      "num_workers": num_workers,
      "pin_memory": True,
      "persistent_workers": num_workers > 0,
  }

  return (
      DataLoader(train_dataset, shuffle=True, **loader_kwargs),
      DataLoader(val_dataset, shuffle=False, **loader_kwargs),
      DataLoader(test_dataset, shuffle=False, **loader_kwargs),
  )

TEST_SCENES= ["baltijsk", "camerino", "codigoro", "copenhagen", "cullivel", "jagersfontein", "kirtland", "lorca"]
VAL_SCENES = ["arborea", "athens", "beer_sheva", "istanbul", "los_cabos", "taiwan", "yuen_long"]
TRAIN_SCENES= [
    "aranjuez","bari", "beheira", "beirut", "belgrade", "binh_dai", "brasilia", "cape_town", "copperton",
    "cukotka", "dellys", "dubai", "dublin", "elsalto", "eyjafjoll", "fontainebleau", "fukushima",
    "guantanamo", "hanging_rock", "java", "jordan", "lagos", "london", "los_angeles",
    "malindi", "mantua", "mexico_city", "montevideo", "mosul", "mrirt", "muscat", "nagaoka",
    "new_york", "nicosia", "nouakchott", "novara", "palermo", "paris", "poinciana", "port_au_prince",
    "prague", "quito", "rome", "salinas", "sanaa", "shanghai", "spinazzola", "suez", "sydney",
    "tampa_bay", "tientsin", "tijuana", "tirana", "valencia"
]
ALL_SCENE=TRAIN_SCENES+TEST_SCENES+VAL_SCENES

scene_ids=[]

TRAIN_SCENES_1 = [
    "aranjuez-after",
    "bari-after",
    "beheira-after",
    "beirut-after",
    "belgrade-before",
    "binh_dai-after",
    "brasilia-before",
    "cape_town-after",
    "copperton-before",
    "cukotka-before",
    "dellys-after",
    "dubai-after",
    "dublin-before",
    "elsalto-before",
    "eyjafjoll-after",
    "fontainebleau-before",
    "fukushima-after",
    "guantanamo-before",
    "hanging_rock-before",
    "java-before",
    "jordan-before",
    "lagos-after",
    "london-after",
    "los_angeles-before",
    "malindi-before",
    "mantua-before",
    "mexico_city-after",
    "montevideo-before",
    "mosul-after",
    "mrirt-after",
    "muscat-after",
    "nagaoka-after",
    "new_york-after",
    "nicosia-before",
    "nouakchott-before",
    "novara-before",
    "palermo-after",
    "paris-after",
    "poinciana-after",
    "port_au_prince-after",
    "prague-after",
    "quito-before",
    "rome-before",
    "salinas-before",
    "sanaa-after",
    "shanghai-after",
    "spinazzola-before",
    "suez-after",
    "sydney-after",
    "tampa_bay-after",
    "tientsin-after",
    "tijuana-before",
    "tirana-before",
    "valencia-after",
]

VAL_SCENES_1 = [
    "arborea-before",
    "athens-after",
    "beer_sheva-after",
    "istanbul-before",
    "los_cabos-before",
    "taiwan-before",
    "yuen_long-after",
]

TEST_SCENES_1 = [
    "baltijsk-before",
    "camerino-after",
    "codigoro-after",
    "copenhagen-after",
    "cullivel-after",
    "jagersfontein-after",
    "kirtland-before",
    "lorca-after",
]

BLACKLIST_PATCHES = [
    "aranjuez-after_patch_0000",
    "beheira-after_patch_0019",
    "beheira-after_patch_0022",
    "beheira-after_patch_0027",
    "beheira-after_patch_0028",
    "codigoro-after_patch_0004",
    "codigoro-after_patch_0005",
    "codigoro-after_patch_0006",
    "codigoro-after_patch_0011",
    "codigoro-after_patch_0014",
    "codigoro-after_patch_0023",
    "codigoro-after_patch_0030",
    "codigoro-after_patch_0031",
    "codigoro-after_patch_0032",
    "codigoro-after_patch_0033",
    "copenhagen-after_patch_0000",
    "copenhagen-after_patch_0001",
    "copenhagen-after_patch_0025",
    "copenhagen-after_patch_0032",
    "dublin-before_patch_0034",
    "elsalto-after_patch_0000",
    "elsalto-before_patch_0001",
    "elsalto-before_patch_0002",
    "elsalto-before_patch_0003",
    "elsalto-before_patch_0006",
    "elsalto-before_patch_0007",
    "elsalto-before_patch_0008",
    "elsalto-before_patch_0014",
    "elsalto-before_patch_0019",
    "elsalto-before_patch_0020",
    "elsalto-before_patch_0021",
    "elsalto-before_patch_0026",
    "elsalto-before_patch_0027",
    "elsalto-before_patch_0029",
    "elsalto-before_patch_0033",
    "elsalto-before_patch_0035",
    "eyjafjoll-after_patch_0005",
    "eyjafjoll-after_patch_0009",
    "eyjafjoll-after_patch_0012",
    "eyjafjoll-after_patch_0014",
    "eyjafjoll-after_patch_0015",
    "eyjafjoll-after_patch_0016",
    "eyjafjoll-after_patch_0019",
    "eyjafjoll-after_patch_0020",
    "eyjafjoll-after_patch_0021",
    "eyjafjoll-after_patch_0022",
    "eyjafjoll-after_patch_0025",
    "eyjafjoll-after_patch_0026",
    "eyjafjoll-after_patch_0027",
    "eyjafjoll-after_patch_0028",
    "eyjafjoll-after_patch_0029",
    "eyjafjoll-after_patch_0031",
    "eyjafjoll-after_patch_0032",
    "eyjafjoll-after_patch_0033",
    "eyjafjoll-after_patch_0034",
    "eyjafjoll-after_patch_0035",
    "istanbul-before_patch_0002",
    "istanbul-before_patch_0003",
    "istanbul-before_patch_0004",
    "istanbul-before_patch_0005",
    "istanbul-before_patch_0008",
    "istanbul-before_patch_0009",
    "istanbul-before_patch_0014",
    "istanbul-before_patch_0020",
    "java-before_patch_0003",
    "java-before_patch_0008",
    "java-before_patch_0009",
    "java-before_patch_0019",
    "java-before_patch_0025",
    "java-before_patch_0026",
    "java-before_patch_0032",
    "java-before_patch_0033",
    "java-before_patch_0034",
    "kitami-before_patch_0000",
    "kitami-before_patch_0001",
    "kitami-before_patch_0002",
    "kitami-before_patch_0003",
    "kitami-before_patch_0004",
    "kitami-before_patch_0005",
    "kitami-before_patch_0006",
    "kitami-before_patch_0007",
    "kitami-before_patch_0008",
    "kitami-before_patch_0009",
    "kitami-before_patch_0010",
    "kitami-before_patch_0011",
    "kitami-before_patch_0012",
    "kitami-before_patch_0013",
    "kitami-before_patch_0014",
    "kitami-before_patch_0015",
    "kitami-before_patch_0016",
    "kitami-before_patch_0017",
    "kitami-before_patch_0018",
    "kitami-before_patch_0019",
    "kitami-before_patch_0020",
    "kitami-before_patch_0021",
    "kitami-before_patch_0022",
    "kitami-before_patch_0023",
    "kitami-before_patch_0024",
    "kitami-before_patch_0025",
    "kitami-before_patch_0026",
    "kitami-before_patch_0027",
    "kitami-before_patch_0028",
    "kitami-before_patch_0029",
    "kitami-before_patch_0030",
    "kitami-before_patch_0031",
    "kitami-before_patch_0032",
    "kitami-before_patch_0033",
    "kitami-before_patch_0034",
    "kitami-before_patch_0035",
    "lorca-after_patch_0009",
    "los_cabos-before_patch_0000",
    "los_cabos-before_patch_0008",
    "los_cabos-before_patch_0012",
    "los_cabos-before_patch_0013",
    "los_cabos-before_patch_0019",
    "los_cabos-before_patch_0024",
    "los_cabos-before_patch_0025",
    "los_cabos-before_patch_0030",
    "malindi-before_patch_0028",
    "malindi-before_patch_0029",
    "mexico_city-after_patch_0021",
    "nagaoka-after_patch_0000",
    "nagaoka-after_patch_0022",
    "nagaoka-after_patch_0024",
    "nagaoka-after_patch_0030",
    "nicosia-after_patch_0002",
    "nicosia-after_patch_0011",
    "palermo-after_patch_0001",
    "palermo-after_patch_0002",
    "palermo-after_patch_0010",
    "palermo-after_patch_0012",
    "palermo-after_patch_0016",
    "palermo-after_patch_0024",
    "palermo-after_patch_0030",
    "paris-after_patch_0002",
    "paris-after_patch_0007",
    "paris-after_patch_0015",
    "paris-after_patch_0017",
    "paris-after_patch_0022",
    "paris-after_patch_0029",
    "salinas-before_patch_0002",
    "spinazzola-after_patch_0005",
    "suez-after_patch_0030",
    "taiwan-after_patch_0010",
    "taiwan-after_patch_0011",
    "taiwan-after_patch_0016",
    "taiwan-after_patch_0017",
    "taiwan-after_patch_0021",
    "taiwan-after_patch_0022",
    "taiwan-after_patch_0023",
    "taiwan-before_patch_0028",
    "taiwan-before_patch_0029",
    "taiwan-before_patch_0030",
    "taiwan-before_patch_0032",
    "taiwan-before_patch_0034",
    "taiwan-before_patch_0035",
]
if __name__=="__main__":

    scene_ids1=["baltijsk-after","camerino-after","codigoro-after","copenhagen-after","cullivel-after","jagersfontein-after","lorca-after"]
    scene_ids2 = ["arborea-after", "athens-after", "beer_sheva-after", "istanbul-after", "los_cabos-after", "taiwan-after", "yuen_long-after"]
    scene_ids=[scene+"-after" for scene in TRAIN_SCENES]

    prepare_mumucd_single_file(VAL_SCENES_1, "val-clean", blacklist_patches=BLACKLIST_PATCHES, patch_size=256,num_workers=1, output_dir=CACHE_DIR)
    #prepare_mumucd_single_file(TEST_SCENES_1, "test-clean", blacklist_patches=BLACKLIST_PATCHES, patch_size=256,num_workers=3, output_dir=CACHE_DIR)
    #prepare_mumucd_single_file(TRAIN_SCENES_1, "train-clean", blacklist_patches=BLACKLIST_PATCHES, patch_size=256,num_workers=3, output_dir=CACHE_DIR)
 
"""
# Ouvre un fichier et vois la config
    f = h5py.File("/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/data/patches_caches/test-after/hsi_interp.h5", "r")
    print(f"Compression: {f['data'].compression}")
    print(f"Chunks: {f['data'].chunks}")
    print(f"Shape: {f['data'].shape}")

# Teste une lecture
    start = time.time()
    data = f['data'][0]  # Lis JUSTE le premier patch
    print(f"Temps lecture 1 patch: {time.time()-start:.2f}s")

    f.close()
"""