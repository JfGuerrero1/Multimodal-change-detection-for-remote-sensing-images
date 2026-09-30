"""
Pipeline de génération des données contrastives MSI vs HSI.

Étapes:
1. Charge les scènes (MSI Sentinel-2 + HSI Prisma)
2. Génère les memmaps de base: msi_real.bin, hsi.bin, hsi_interp.bin
3. Exécute l'inférence des modèles de reconstruction et d'incertitude
4. Sauvegarde reconstruction.bin et uncertainty.bin

Utilisation:
    python generate_contrastive_pipeline_clean.py --config config.yaml --split train
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import xarray as xr
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import CACHE_DIR, DATA_DIR, INTERP_MATRIX, SRF_MATRIX, WVL_PRS
from src.models.dual_branch import DualBranchNAFNet
from src.models.unet_residual import GradualExpansionUNet_residual
from src.models.unet_standard import GradualExpansionUNet
from src.training.utils_train import get_kept_wavelength_indices


# ============================================================================
# CONFIGURATION SCÈNES & BLACKLIST
# ============================================================================

TRAIN_SCENES = [
    "aranjuez-after", "bari-after", "beheira-after", "beirut-after", "belgrade-before",
    "binh_dai-after", "brasilia-before", "cape_town-after", "copperton-before", "cukotka-before",
    "dellys-after", "dubai-after", "dublin-before", "elsalto-before", "eyjafjoll-after",
    "fontainebleau-before", "fukushima-after", "guantanamo-before", "hanging_rock-before",
    "java-before", "jordan-before", "lagos-after", "london-after", "los_angeles-before",
    "malindi-before", "mantua-before", "mexico_city-after", "montevideo-before", "mosul-after",
    "mrirt-after", "muscat-after", "nagaoka-after", "new_york-after", "nicosia-before",
    "nouakchott-before", "novara-before", "palermo-after", "paris-after", "poinciana-after",
    "port_au_prince-after", "prague-after", "quito-before", "rome-before", "salinas-before",
    "sanaa-after", "shanghai-after", "spinazzola-before", "suez-after", "sydney-after",
    "tampa_bay-after", "tientsin-after", "tijuana-before", "tirana-before", "valencia-after",
]

VAL_SCENES = [
    "arborea-before", "athens-after", "beer_sheva-after", "istanbul-before",
    "los_cabos-before", "taiwan-before", "yuen_long-after",
]

TEST_SCENES = [
    "baltijsk-before", "camerino-after", "codigoro-after", "copenhagen-after",
    "cullivel-after", "jagersfontein-after", "kirtland-before", "lorca-after",
]

BLACKLIST_PATCHES = {
    "aranjuez-after_patch_0000", "beheira-after_patch_0019", "beheira-after_patch_0022",
    "beheira-after_patch_0027", "beheira-after_patch_0028", "codigoro-after_patch_0004",
    "codigoro-after_patch_0005", "codigoro-after_patch_0006", "codigoro-after_patch_0011",
    "codigoro-after_patch_0014", "codigoro-after_patch_0023", "codigoro-after_patch_0030",
    "codigoro-after_patch_0031", "codigoro-after_patch_0032", "codigoro-after_patch_0033",
    "copenhagen-after_patch_0000", "copenhagen-after_patch_0001", "copenhagen-after_patch_0025",
    "copenhagen-after_patch_0032", "dublin-before_patch_0034", "elsalto-after_patch_0000",
    "elsalto-before_patch_0001", "elsalto-before_patch_0002", "elsalto-before_patch_0003",
    "elsalto-before_patch_0006", "elsalto-before_patch_0007", "elsalto-before_patch_0008",
    "elsalto-before_patch_0014", "elsalto-before_patch_0019", "elsalto-before_patch_0020",
    "elsalto-before_patch_0021", "elsalto-before_patch_0026", "elsalto-before_patch_0027",
    "elsalto-before_patch_0029", "elsalto-before_patch_0033", "elsalto-before_patch_0035",
    "eyjafjoll-after_patch_0005", "eyjafjoll-after_patch_0009", "eyjafjoll-after_patch_0012",
    "eyjafjoll-after_patch_0014", "eyjafjoll-after_patch_0015", "eyjafjoll-after_patch_0016",
    "eyjafjoll-after_patch_0019", "eyjafjoll-after_patch_0020", "eyjafjoll-after_patch_0021",
    "eyjafjoll-after_patch_0022", "eyjafjoll-after_patch_0025", "eyjafjoll-after_patch_0026",
    "eyjafjoll-after_patch_0027", "eyjafjoll-after_patch_0028", "eyjafjoll-after_patch_0029",
    "eyjafjoll-after_patch_0031", "eyjafjoll-after_patch_0032", "eyjafjoll-after_patch_0033",
    "eyjafjoll-after_patch_0034", "eyjafjoll-after_patch_0035", "istanbul-before_patch_0002",
    "istanbul-before_patch_0003", "istanbul-before_patch_0004", "istanbul-before_patch_0005",
    "istanbul-before_patch_0008", "istanbul-before_patch_0009", "istanbul-before_patch_0014",
    "istanbul-before_patch_0020", "java-before_patch_0003", "java-before_patch_0008",
    "java-before_patch_0009", "java-before_patch_0019", "java-before_patch_0025",
    "java-before_patch_0026", "java-before_patch_0032", "java-before_patch_0033",
    "java-before_patch_0034", "kitami-before_patch_0000", "kitami-before_patch_0001",
    "kitami-before_patch_0002", "kitami-before_patch_0003", "kitami-before_patch_0004",
    "kitami-before_patch_0005", "kitami-before_patch_0006", "kitami-before_patch_0007",
    "kitami-before_patch_0008", "kitami-before_patch_0009", "kitami-before_patch_0010",
    "kitami-before_patch_0011", "kitami-before_patch_0012", "kitami-before_patch_0013",
    "kitami-before_patch_0014", "kitami-before_patch_0015", "kitami-before_patch_0016",
    "kitami-before_patch_0017", "kitami-before_patch_0018", "kitami-before_patch_0019",
    "kitami-before_patch_0020", "kitami-before_patch_0021", "kitami-before_patch_0022",
    "kitami-before_patch_0023", "kitami-before_patch_0024", "kitami-before_patch_0025",
    "kitami-before_patch_0026", "kitami-before_patch_0027", "kitami-before_patch_0028",
    "kitami-before_patch_0029", "kitami-before_patch_0030", "kitami-before_patch_0031",
    "kitami-before_patch_0032", "kitami-before_patch_0033", "kitami-before_patch_0034",
    "kitami-before_patch_0035", "lorca-after_patch_0009", "los_cabos-before_patch_0000",
    "los_cabos-before_patch_0008", "los_cabos-before_patch_0012", "los_cabos-before_patch_0013",
    "los_cabos-before_patch_0019", "los_cabos-before_patch_0024", "los_cabos-before_patch_0025",
    "los_cabos-before_patch_0030", "malindi-before_patch_0028", "malindi-before_patch_0029",
    "mexico_city-after_patch_0021", "nagaoka-after_patch_0000", "nagaoka-after_patch_0022",
    "nagaoka-after_patch_0024", "nagaoka-after_patch_0030", "nicosia-after_patch_0002",
    "nicosia-after_patch_0011", "palermo-after_patch_0001", "palermo-after_patch_0002",
    "palermo-after_patch_0010", "palermo-after_patch_0012", "palermo-after_patch_0016",
    "palermo-after_patch_0024", "palermo-after_patch_0030", "paris-after_patch_0002",
    "paris-after_patch_0007", "paris-after_patch_0015", "paris-after_patch_0017",
    "paris-after_patch_0022", "paris-after_patch_0029", "salinas-before_patch_0002",
    "spinazzola-after_patch_0005", "suez-after_patch_0030", "taiwan-after_patch_0010",
    "taiwan-after_patch_0011", "taiwan-after_patch_0016", "taiwan-after_patch_0017",
    "taiwan-after_patch_0021", "taiwan-after_patch_0022", "taiwan-after_patch_0023",
    "taiwan-before_patch_0028", "taiwan-before_patch_0029", "taiwan-before_patch_0030",
    "taiwan-before_patch_0032", "taiwan-before_patch_0034", "taiwan-before_patch_0035",
}


# ============================================================================
# ÉTAPE 1: PRÉPARATION DES MEMMAPS (MSI, HSI, Interp)
# ============================================================================

def _process_single_scene(args):
    """Traite une scène en parallèle: écrit dans les memmaps."""
    (
        x_file, y_file, base_name, patch_list, output_path,
        shape_msi, shape_hsi, c_hsi, c_multi, patch_size,
        srf_matrix, interp_matrix, kept_indices,
    ) = args

    with xr.open_dataset(x_file) as ds_x, xr.open_dataset(y_file) as ds_y:
        msi_full = ds_x["sr"].to_numpy().astype(np.float32)
        hsi_full = ds_y["sr"].to_numpy().astype(np.float32)

    if kept_indices is not None:
        hsi_full = hsi_full[..., kept_indices]

    h, w, _ = hsi_full.shape
    actual_c_hsi = hsi_full.shape[-1]

    # Simulations spectrales
    hsi_2d = hsi_full.reshape(-1, actual_c_hsi)
    msi_sim = np.dot(hsi_2d, srf_matrix).reshape(h, w, c_multi).astype(np.float32)

    mean_msi = np.mean(msi_full, axis=(0, 1), keepdims=True)
    std_msi = np.std(msi_full, axis=(0, 1), keepdims=True)
    msi_norm = (msi_full - mean_msi) / (std_msi + 1e-8)

    std_sim = np.std(msi_sim, axis=(0, 1), keepdims=True)
    mean_sim = np.mean(msi_sim, axis=(0, 1), keepdims=True)
    msi_norm = msi_norm * std_sim + mean_sim

    msi_chw = np.transpose(msi_full, (2, 0, 1))
    hsi_interp_chw = (interp_matrix @ msi_chw.reshape(c_multi, -1)).reshape(actual_c_hsi, h, w).astype(np.float32)
    hsi_chw = np.transpose(hsi_full, (2, 0, 1))

    # Accès aux memmaps
    fp_msi = np.memmap(output_path / "msi_real.bin", dtype="float32", mode="r+", shape=shape_msi)
    fp_hsi = np.memmap(output_path / "hsi.bin", dtype="float32", mode="r+", shape=shape_hsi)
    fp_interp = np.memmap(output_path / "hsi_interp.bin", dtype="float32", mode="r+", shape=shape_hsi)

    scene_results = []
    for global_idx, r, c, patch_id_str in patch_list:
        fp_msi[global_idx] = msi_chw[:, r:r + patch_size, c:c + patch_size]
        fp_hsi[global_idx] = hsi_chw[:, r:r + patch_size, c:c + patch_size]
        fp_interp[global_idx] = hsi_interp_chw[:, r:r + patch_size, c:c + patch_size]
        scene_results.append((global_idx, patch_id_str))

    fp_msi.flush()
    fp_hsi.flush()
    fp_interp.flush()
    return scene_results


def prepare_base_memmaps(scene_ids, split_name, config, patch_size=256, output_dir="cache/mumucd", num_workers=4):
    """Génère les memmaps de base: msi_real.bin, hsi.bin, hsi_interp.bin"""
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)

    kept_indices = get_kept_wavelength_indices(WVL_PRS, config)
    interp_matrix = INTERP_MATRIX[kept_indices, :] if kept_indices is not None else INTERP_MATRIX
    srf_matrix = SRF_MATRIX[kept_indices, :] if kept_indices is not None else SRF_MATRIX

    # Trouver les fichiers valides
    scene_pairs = []
    for item in scene_ids:
        scene_name = item.split("-")[0]
        hsi_file = DATA_DIR / scene_name / f"{item}-prs.nc"
        msi_file = DATA_DIR / scene_name / f"{item}-s2.nc"
        if hsi_file.exists() and msi_file.exists():
            scene_pairs.append((msi_file, hsi_file, item))

    print(f"📍 {len(scene_pairs)} scènes trouvées")

    c_multi = SRF_MATRIX.shape[1]
    with xr.open_dataset(scene_pairs[0][1]) as ds:
        hsi = ds["sr"].to_numpy().astype(np.float32)
        if kept_indices is not None:
            hsi = hsi[..., kept_indices]
        c_hsi = hsi.shape[-1]
    print(f"📊 HSI: {c_hsi} canaux | MSI: {c_multi} canaux")

    # Créer les patch lists et pré-allouer les memmaps
    scene_tasks = []
    global_idx = 0
    for msi_file, hsi_file, base_name in scene_pairs:
        with xr.open_dataset(hsi_file) as ds:
            h, w, _ = ds["sr"].shape
            h_crop = h - (h % patch_size)
            w_crop = w - (w % patch_size)

        patches = []
        for i in range(h_crop // patch_size):
            for j in range(w_crop // patch_size):
                patch_id_str = f"{base_name}_patch_{i * (w_crop // patch_size) + j:04d}"
                if patch_id_str not in BLACKLIST_PATCHES:
                    patches.append((global_idx, i * patch_size, j * patch_size, patch_id_str))
                    global_idx += 1

        if patches:
            scene_tasks.append((msi_file, hsi_file, base_name, patches))

    total_patches = global_idx
    if total_patches == 0:
        print("❌ Aucun patch valide")
        return output_path

    shape_msi = (total_patches, c_multi, patch_size, patch_size)
    shape_hsi = (total_patches, c_hsi, patch_size, patch_size)

    print(f"🚀 Pré-allocation {total_patches} patches...")
    for name, shape in [("msi_real.bin", shape_msi), ("hsi.bin", shape_hsi), ("hsi_interp.bin", shape_hsi)]:
        fp = np.memmap(output_path / name, dtype="float32", mode="w+", shape=shape)
        fp.flush()
        del fp

    # Traitement parallèle
    worker_args = [
        (msi_f, hsi_f, base, patches, output_path, shape_msi, shape_hsi,
         c_hsi, c_multi, patch_size, srf_matrix, interp_matrix, kept_indices)
        for msi_f, hsi_f, base, patches in scene_tasks
    ]

    print(f"🔥 Traitement parallèle ({num_workers} workers)...")
    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_single_scene, arg) for arg in worker_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scènes"):
            all_results.extend(future.result())

    all_results.sort(key=lambda x: x[0])
    patch_ids = [p_id for _, p_id in all_results]

    metadata = {
        "total_patches": total_patches,
        "msi_shape": list(shape_msi),
        "hsi_shape": list(shape_hsi),
        "dtype": "float32",
        "patch_ids": patch_ids,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Memmaps générés dans {output_path}")
    return output_path


# ============================================================================
# ÉTAPE 2: INFÉRENCE RECONSTRUCTION & INCERTITUDE
# ============================================================================

class MemmapDataset(Dataset):
    """Lit les memmaps de base."""
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "metadata.json") as f:
            meta = json.load(f)
        self.total_patches = meta["total_patches"]
        self.msi_shape = tuple(meta["msi_shape"])
        self.hsi_shape = tuple(meta["hsi_shape"])
        self.patch_ids = meta["patch_ids"]

        self.fp_msi = np.memmap(self.data_dir / "msi_real.bin", dtype="float32", mode="r", shape=self.msi_shape)
        self.fp_interp = np.memmap(self.data_dir / "hsi_interp.bin", dtype="float32", mode="r", shape=self.hsi_shape)

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        msi = torch.from_numpy(np.array(self.fp_msi[idx]))
        interp = torch.from_numpy(np.array(self.fp_interp[idx]))
        return msi, interp, self.patch_ids[idx]


def build_model_reconstruction(config, n_msi, n_hsi):
    """Instancie le modèle de reconstruction."""
    cfg = config["model"]
    name = cfg.get("name", "").lower()
    if name == "gradualexpansionunet":
        return GradualExpansionUNet(
            in_msi=n_msi, in_hsi=n_hsi,
            interpolation_mode=cfg.get("interpolation_mode", "Bilinear"),
            base_channel=cfg.get("base_channel", 64),
            activation=cfg.get("activation", "silu"),
            with_batch_norm=cfg.get("with_batch_norm", True),
            with_mlp_spectral=cfg.get("with_mlp_spectral", False),
            final_activation=cfg.get("final_activation", None),
        )
    elif name == "gradualexpansionunet_res":
        return GradualExpansionUNet_residual(
            in_msi=n_msi, in_hsi=n_hsi,
            interpolation_mode=cfg.get("interpolation_mode", "Bilinear"),
            base_channel=cfg.get("base_channel", 64),
            activation=cfg.get("activation", "silu"),
            with_batch_norm=cfg.get("with_batch_norm", True),
            with_mlp_spectral=cfg.get("with_mlp_spectral", False),
            final_activation=cfg.get("final_activation", None),
        )
    raise ValueError(f"Modèle inconnu: {name}")


def build_model_uncertainty(config, n_msi, n_hsi):
    """Instancie le modèle d'incertitude."""
    cfg = config.get("model_uncertainty", {})
    name = cfg.get("name", "dualbranchnafnet").lower()
    if name == "dualbranchnafnet":
        return DualBranchNAFNet(
            n_msi=n_msi, n_hsi=n_hsi, out_channels=n_hsi,
            width=cfg.get("base_channel", 64),
            middle_blk_num=cfg.get("middle_blk_num", 1),
            enc_blk_nums=cfg.get("enc_blk_nums", []),
            dec_blk_nums=cfg.get("dec_blk_nums", []),
            drop_out_rate=cfg.get("drop_out_rate", 0.0),
            final_op=cfg.get("final_activation", "softplus"),
        )
    raise ValueError(f"Modèle incertitude inconnu: {name}")


def run_inference(data_path, config, output_dir="cache/mumucd"):
    """Lance l'inférence reconstruction & incertitude."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("evaluation", {}).get("use_amp", True) and device.type == "cuda"

    dataset = MemmapDataset(data_path)
    loader = DataLoader(dataset, batch_size=config.get("data", {}).get("batch_size", 8), shuffle=False, num_workers=0)

    n_msi, n_hsi = dataset.msi_shape[1], dataset.hsi_shape[1]
    total_patches = dataset.total_patches
    patch_h, patch_w = dataset.hsi_shape[2], dataset.hsi_shape[3]

    eval_cfg = config.get("evaluation", {})

    # Charger modèles
    print("🏗️ Chargement modèles...")
    model_rec = build_model_reconstruction(config, n_msi, n_hsi).to(device)
    ckpt_rec = torch.load(eval_cfg.get("weights_path_reconstruction"), map_location=device, weights_only=False)
    sd_rec = ckpt_rec.get("model_state_dict", ckpt_rec.get("state_dict", ckpt_rec))
    model_rec.load_state_dict(sd_rec)
    model_rec.eval()

    ckpt_unc = torch.load(eval_cfg.get("weights_path_uncertainty"), map_location=device, weights_only=False)
    sd_unc = ckpt_unc.get("model_state_dict", ckpt_unc.get("state_dict", ckpt_unc))
    n_unc = sd_unc["ending.weight"].shape[0] if "ending.weight" in sd_unc else n_hsi

    model_unc = build_model_uncertainty(config, n_msi, n_hsi).to(device)
    model_unc.load_state_dict(sd_unc)
    model_unc.eval()

    # Pré-allouer memmaps
    shape_rec = (total_patches, n_hsi, patch_h, patch_w)
    shape_unc = (total_patches, n_unc, patch_h, patch_w)

    print(f"🚀 Création memmaps de sortie:\n   Rec: {shape_rec}\n   Unc: {shape_unc}")

    fp_rec = np.memmap(data_path / "reconstruction.bin", dtype="float32", mode="w+", shape=shape_rec)
    fp_unc = np.memmap(data_path / "uncertainty.bin", dtype="float32", mode="w+", shape=shape_unc)

    # Inférence
    global_idx = 0
    with torch.no_grad():
        for msi, interp, _ in tqdm(loader, desc="Inférence"):
            msi = msi.to(device, non_blocking=True)
            interp = interp.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                rec = model_rec(msi, interp)
                unc = model_unc(torch.cat([msi, rec], dim=1))

            rec_np = rec.detach().cpu().numpy()
            unc_np = unc.detach().cpu().numpy()

            for b in range(rec.shape[0]):
                fp_rec[global_idx] = rec_np[b]
                fp_unc[global_idx] = unc_np[b]
                global_idx += 1

    fp_rec.flush()
    fp_unc.flush()

    # Métadonnées
    with open(data_path / "metadata_reconstruction.json", "w") as f:
        json.dump({"total_patches": total_patches, "shape": list(shape_rec), "dtype": "float32", "patch_ids": dataset.patch_ids}, f, indent=2)
    with open(data_path / "metadata_uncertainty.json", "w") as f:
        json.dump({"total_patches": total_patches, "shape": list(shape_unc), "dtype": "float32", "patch_ids": dataset.patch_ids}, f, indent=2)

    print(f"✅ Inférence terminée dans {data_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline contrastif MSI vs HSI")
    parser.add_argument("--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="cache/mumucd")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"🚀 Pipeline {args.split} | Config: {args.config}")

    # Sélectionner les scènes
    if args.split == "train":
        scene_ids, split_name = TRAIN_SCENES, "train-contrastive"
    elif args.split == "val":
        scene_ids, split_name = VAL_SCENES, "val-contrastive"
    else:
        scene_ids, split_name = TEST_SCENES, "test-contrastive"

    output_path = Path(args.output_dir) / split_name

    # Étape 1: Memmaps de base
    print(f"\n{'='*60}")
    print("ÉTAPE 1: Génération des memmaps de base")
    print(f"{'='*60}\n")
    data_path = prepare_base_memmaps(
        scene_ids=scene_ids,
        split_name=split_name,
        config=config,
        patch_size=config.get("data", {}).get("patch_size", 256),
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )

    # Étape 2: Inférence
    print(f"\n{'='*60}")
    print("ÉTAPE 2: Inférence reconstruction & incertitude")
    print(f"{'='*60}\n")
    run_inference(data_path, config, output_dir=args.output_dir)

    print(f"\n{'='*60}")
    print(f"🎉 Pipeline terminé | Sortie: {data_path}")
    print(f"{'='*60}\n")