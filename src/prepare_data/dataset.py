import os
import json
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from src.constants import INTERP_MATRIX

def create_data_loaders_spectral(
    train_dir,
    val_dir,
    test_dir,
    proportion_simulated=0.0,
    augment=False,
    augment_illumination=False,
    batch_size=8,
    num_workers=4,
    is_residual=False,
    is_normalised=False,
    kept_indices=None,
    training=True
):
    """Crée et renvoie les DataLoaders (train, val, test) pour les données spectrales."""
    train_dataset = SpectralDataset(
        dataset_dir=train_dir,
        proportion_simulated=proportion_simulated,
        is_normalised=is_normalised,
        augment=augment,
        augment_illumination=augment_illumination,
        is_residual=is_residual,
        kept_indices=kept_indices,
    )

    val_dataset = SpectralDataset(
        dataset_dir=val_dir,
        proportion_simulated=0.0,
        is_normalised=is_normalised,
        augment=False,
        augment_illumination=False,
        is_residual=is_residual,
        kept_indices=kept_indices,
    )

    test_dataset = SpectralDataset(
        dataset_dir=test_dir,
        proportion_simulated=0.0,
        is_normalised=is_normalised,
        augment=False,
        augment_illumination=False,
        is_residual=is_residual,
        kept_indices=kept_indices,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    
    if training:
        return (
            DataLoader(train_dataset, shuffle=True, **loader_kwargs),
            DataLoader(val_dataset, shuffle=False, **loader_kwargs),
            DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        )
    else:
        return (
            DataLoader(train_dataset, shuffle=False, **loader_kwargs),
            DataLoader(val_dataset, shuffle=False, **loader_kwargs),
            DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        )


class SpectralDataset(Dataset):

    def __init__(
        self,
        dataset_dir,
        proportion_simulated=0.0,
        is_normalised=False,
        augment=False,
        augment_illumination=False,
        is_residual=False,
        kept_indices=None,
        patch_size=None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment
        self.is_residual = is_residual
        self.proportion_simulated = proportion_simulated
        self.is_normalised = is_normalised

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

        if kept_indices is None:
            self.indices_to_use = None
        else:
            self.indices_to_use = kept_indices

        self.is_color_augmented = augment_illumination

        self.paths = {
            "meta": self.dataset_dir / "metadata.json",
            "msi_real": self.dataset_dir / "msi_real.bin",
            "msi_sim": self.dataset_dir / "msi_sim.bin",
            "msi_norm": self.dataset_dir / "msi_norm.bin",
            "hsi": self.dataset_dir / "hsi.bin",
            "interp": self.dataset_dir / "interp.bin",
        }

        with open(self.paths["meta"], "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.length = self.meta["total_patches"]
        self.msi_shape = tuple(self.meta["msi_shape"])
        self.hsi_shape = tuple(self.meta["hsi_shape"])
        self.dtype = self.meta["dtype"]
        self.patch_ids = self.meta["patch_ids"]

        self.fp_msi_real = None
        self.fp_msi_sim = None
        self.fp_msi_norm = None
        self.fp_hsi = None
        self.fp_interp = None

    def _init_memmap(self):
        if self.fp_hsi is not None:
            return

        try:
            if self.paths["msi_real"].exists():
                self.fp_msi_real = np.memmap(
                    self.paths["msi_real"], dtype=self.dtype, mode="r", shape=self.msi_shape
                )
            if self.paths["msi_sim"].exists():
                self.fp_msi_sim = np.memmap(
                    self.paths["msi_sim"], dtype=self.dtype, mode="r", shape=self.msi_shape
                )
            if self.paths["msi_norm"].exists():
                self.fp_msi_norm = np.memmap(
                    self.paths["msi_norm"], dtype=self.dtype, mode="r", shape=self.msi_shape
                )

            self.fp_hsi = np.memmap(
                self.paths["hsi"], dtype=self.dtype, mode="r", shape=self.hsi_shape
            )

            if self.is_residual:
                if not self.paths["interp"].exists():
                    raise FileNotFoundError(f"Le fichier requis {self.paths['interp']} est introuvable !")
                self.fp_interp = np.memmap(
                    self.paths["interp"], dtype=self.dtype, mode="r", shape=self.hsi_shape
                )

        except Exception as e:
            print(f"❌ Erreur lors de l'ouverture des fichiers memmap dans {self.dataset_dir} : {e}")
            raise e

    def __len__(self):
        return self.length

    def augment_color_jittering(self, *tensors, p=0.3):
        if random.random() < p:
            factor = random.uniform(0.97, 1.03)
            offset = random.uniform(-0.003, 0.003)
            tensors = [t * factor + offset for t in tensors]
        return tensors

    def augment_geometric(self, *tensors, p_flip=0.5):
        if random.random() < p_flip:
            tensors = [torch.flip(t, dims=[2]) for t in tensors]
        if random.random() < p_flip:
            tensors = [torch.flip(t, dims=[1]) for t in tensors]
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            tensors = [torch.rot90(t, k, dims=[1, 2]) for t in tensors]
        return tensors
        
    def _apply_crop(self, *tensors):
        if self.patch_size is None:
            return tensors
        
        _, h, w = tensors[0].shape
        target_h, target_w = self.patch_size

        if h < target_h or w < target_w:
            raise ValueError(f"La taille du patch stocké ({h}x{w}) est inférieure au patch_size demandé ({target_h}x{target_w})")

        if h == target_h and w == target_w:
            return tensors

        top = random.randint(0, h - target_h) if self.augment else (h - target_h) // 2
        left = random.randint(0, w - target_w) if self.augment else (w - target_w) // 2

        return [t[:, top:top + target_h, left:left + target_w] for t in tensors]

    def __getitem__(self, idx):
        if self.fp_hsi is None:
            self._init_memmap()

        if self.is_normalised:
            x_patch = self.fp_msi_norm[idx].copy()
        elif random.random() < self.proportion_simulated:
            x_patch = self.fp_msi_sim[idx].copy()
        else:
            x_patch = self.fp_msi_real[idx].copy()

        y_patch = self.fp_hsi[idx].copy()
        patch_id = self.patch_ids[idx]

        if self.indices_to_use is not None:
            y_patch = y_patch[self.indices_to_use, :, :]

        x = torch.from_numpy(x_patch).float()
        y = torch.from_numpy(y_patch).float()

        if self.is_residual:
            if self.is_normalised:
                c_multi = x_patch.shape[0]
                c_hsi_full = self.hsi_shape[1]
                h, w = self.hsi_shape[2], self.hsi_shape[3]
                interp_numpy = INTERP_MATRIX @ x_patch.reshape(c_multi, -1)
                x_interp_numpy = interp_numpy.reshape(c_hsi_full, h, w).astype(np.float32)
            else:
                x_interp_patch = self.fp_interp[idx].copy()
                x_interp_numpy = x_interp_patch.astype(np.float32)

            if self.indices_to_use is not None:
                x_interp_numpy = x_interp_numpy[self.indices_to_use, :, :]

            x_interp = torch.from_numpy(x_interp_numpy).float()

            x, x_interp, y = self._apply_crop(x, x_interp, y)

            if self.is_color_augmented:
                x, x_interp = self.augment_color_jittering(x, x_interp)

            if self.augment:
                x, x_interp, y = self.augment_geometric(x, x_interp, y)

            return x, x_interp, y, patch_id

        else:
            x, y = self._apply_crop(x, y)

            if self.is_color_augmented:
                x, = self.augment_color_jittering(x)

            if self.augment:
                x, y = self.augment_geometric(x, y)

            return x, y, patch_id


import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UniversalRawMSIHSIDataset(Dataset):
    """
    Dataset universel brut.
    Renvoie un tuple complet : (msi, hsi, rec, unc, mask) 
    avec gestion optionnelle du mode contrastif (two_zone).
    """
    def __init__(
        self,
        data_dir: Path,
        kept_indices=None,
        is_train=False,
        patch_size=None,
        return_mask: bool = True,
        threshold_mode: str = "empirical",
        tau_low_pct: float = 50.0,
        tau_high_pct: float = 98.0,
        gamma_low_pct: float = 50.0,
        sample_size_for_percentiles: int = 100,
        two_zone: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.kept_indices = kept_indices
        self.is_train = is_train
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.return_mask = return_mask
        self.threshold_mode = threshold_mode
        self.two_zone = two_zone

        with open(self.data_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.hsi_shape = tuple(self.metadata["hsi_shape"])
        self.msi_shape = tuple(self.metadata["msi_shape"])
        self.total_patches = int(self.metadata["total_patches"])
        self.patch_ids = self.metadata["patch_ids"]

        # Indexation par scène pour le two_zone
        self.scene_to_indices = {}
        for idx, p_id in enumerate(self.patch_ids):
            scene_name = p_id.rsplit("_patch_", 1)[0]
            self.scene_to_indices.setdefault(scene_name, []).append(idx)

        # Calcul des seuils si le masque est demandé
        if self.return_mask:
            self.tau_low, self.tau_high, self.gamma_low = self._compute_thresholds(
                threshold_mode, tau_low_pct, tau_high_pct, gamma_low_pct, sample_size_for_percentiles
            )
            print(f"📂 Dataset brut (mask=True, two_zone={self.two_zone}): {self.total_patches} patches")
        else:
            print(f"📂 Dataset brut (mode standard): {self.total_patches} patches")

        self.fp_msi = None
        self.fp_hsi = None
        self.fp_rec = None
        self.fp_unc = None

    def __len__(self):
        return self.total_patches

    def _init_memmap(self):
        if self.fp_hsi is not None:
            return
        self.fp_msi = np.memmap(self.data_dir / "msi_real.bin", dtype=np.float32, mode="r", shape=self.msi_shape)
        self.fp_hsi = np.memmap(self.data_dir / "hsi.bin", dtype=np.float32, mode="r", shape=self.hsi_shape)
        self.fp_rec = np.memmap(self.data_dir / "reconstruction.bin", dtype=np.float32, mode="r", shape=self.hsi_shape)
        self.fp_unc = np.memmap(self.data_dir / "uncertainty.bin", dtype=np.float32, mode="r", shape=self.hsi_shape)

    def _compute_thresholds(self, mode, tau_low_pct, tau_high_pct, gamma_low_pct, sample_size):
        if mode == "theoretical":
            return (
                float(-np.log(1.0 - tau_low_pct / 100.0)),
                float(-np.log(1.0 - tau_high_pct / 100.0)),
                float(-np.log(1.0 - gamma_low_pct / 100.0))
            )
        elif mode == "empirical":
            fp_hsi = np.memmap(self.data_dir / "hsi.bin", dtype=np.float32, mode="r", shape=self.hsi_shape)
            fp_rec = np.memmap(self.data_dir / "reconstruction.bin", dtype=np.float32, mode="r", shape=self.hsi_shape)
            fp_unc = np.memmap(self.data_dir / "uncertainty.bin", dtype=np.float32, mode="r", shape=self.hsi_shape)

            sample_size = min(sample_size, self.total_patches)
            indices = np.random.choice(self.total_patches, size=sample_size, replace=False)
            all_errors, all_uncs = [], []

            for idx in indices:
                abs_err = np.abs(fp_hsi[idx] - fp_rec[idx])
                b_map = np.clip(fp_unc[idx], a_min=1e-6, a_max=None)
                norm_err_map = np.mean(abs_err / b_map, axis=0)
                unc_mean = np.mean(fp_unc[idx], axis=0)

                all_errors.extend(norm_err_map.ravel())
                all_uncs.extend(unc_mean.ravel())

            return (
                float(np.percentile(all_errors, tau_low_pct)),
                float(np.percentile(all_errors, tau_high_pct)),
                float(np.percentile(all_uncs, gamma_low_pct))
            )
        else:
            raise ValueError(f"Mode de seuillage inconnu : {mode}")

    def _apply_synchronized_crop_and_aug(self, msi, hsi, rec, unc, mask=None):
        _, h, w = hsi.shape
        target_h, target_w = self.patch_size

        if h < target_h or w < target_w:
            raise ValueError(f"Patch trop petit: ({h}x{w}) < ({target_h}x{target_w})")

        # 1. Crop synchrone
        if h > target_h or w > target_w:
            top = random.randint(0, h - target_h) if self.is_train else (h - target_h) // 2
            left = random.randint(0, w - target_w) if self.is_train else (w - target_w) // 2

            msi = msi[:, top:top + target_h, left:left + target_w]
            hsi = hsi[:, top:top + target_h, left:left + target_w]
            rec = rec[:, top:top + target_h, left:left + target_w]
            unc = unc[:, top:top + target_h, left:left + target_w]
            if mask is not None:
                mask = mask[top:top + target_h, left:left + target_w]

        # 2. Augmentations synchrones (uniquement en train)
        if self.is_train:
            if random.random() > 0.5:
                msi, hsi, rec, unc = torch.flip(msi, [-1]), torch.flip(hsi, [-1]), torch.flip(rec, [-1]), torch.flip(unc, [-1])
                if mask is not None: mask = torch.flip(mask, [-1])
            if random.random() > 0.5:
                msi, hsi, rec, unc = torch.flip(msi, [-2]), torch.flip(hsi, [-2]), torch.flip(rec, [-2]), torch.flip(unc, [-2])
                if mask is not None: mask = torch.flip(mask, [-2])
            k = random.randint(0, 3)
            if k > 0:
                msi, hsi, rec, unc = torch.rot90(msi, k, [-2, -1]), torch.rot90(hsi, k, [-2, -1]), torch.rot90(rec, k, [-2, -1]), torch.rot90(unc, k, [-2, -1])
                if mask is not None: mask = torch.rot90(mask, k, [-2, -1])
            
            # Petites variations radiométriques optionnelles
            if random.random() > 0.3:
                f = random.uniform(0.9, 1.1)
                msi = msi * f
                rec = rec * f
            if random.random() > 0.3:
                hsi = hsi * random.uniform(0.9, 1.1)

        return msi, hsi, rec, unc, mask

    def _process_single_patch(self, idx):
        msi = torch.from_numpy(self.fp_msi[idx].copy()).float()
        hsi = torch.from_numpy(self.fp_hsi[idx].copy()).float()
        rec = torch.from_numpy(self.fp_rec[idx].copy()).float()
        unc = torch.from_numpy(np.clip(self.fp_unc[idx].copy(), 1e-6, None)).float()

        mask = None
        if self.return_mask:
            abs_err = torch.abs(hsi - rec)
            norm_err = torch.mean(abs_err / unc, dim=0)
            unc_mean = torch.mean(unc, dim=0)

            pos_cond = (norm_err < self.tau_low) & (unc_mean < self.gamma_low)
            neg_cond = (norm_err > self.tau_high)

            mask = torch.zeros(norm_err.shape, dtype=torch.int8)
            mask[pos_cond] = 1
            mask[neg_cond] = -1

        if self.patch_size is not None or self.is_train:
            msi, hsi, rec, unc, mask = self._apply_synchronized_crop_and_aug(msi, hsi, rec, unc, mask)

        # Filtrage optionnel des bandes HSI/Rec/Unc
        if self.kept_indices is not None and hsi.shape[0] >= len(self.kept_indices):
            hsi = hsi[self.kept_indices]
            rec = rec[self.kept_indices]
            unc = unc[self.kept_indices]

        return msi, hsi, rec, unc, mask

    def __getitem__(self, idx):
        self._init_memmap()

        msi1, hsi1, rec1, unc1, mask1 = self._process_single_patch(idx)

        if not self.two_zone:
            return (msi1, hsi1, rec1, unc1, mask1) if self.return_mask else (msi1, hsi1, rec1, unc1)
        else:
            scene_name = self.patch_ids[idx].rsplit("_patch_", 1)[0]
            same_scene = self.scene_to_indices[scene_name]
            neighbor_idx = random.choice([i for i in same_scene if i != idx]) if len(same_scene) > 1 else idx

            msi2, hsi2, rec2, unc2, mask2 = self._process_single_patch(neighbor_idx)
            return (msi1, hsi1, rec1, unc1, mask1, msi2, hsi2, rec2, unc2, mask2)
from torch.utils.data import DataLoader

def create_universal_dataloaders(
    train_dir,
    val_dir,
    test_dir,
    batch_size=8,
    num_workers=4,
    patch_size=64,
    return_mask=True,
    two_zone=False,
    kept_indices=None,
    threshold_mode: str = "empirical",
    tau_low_pct: float = 50.0,
    tau_high_pct: float = 98.0,
    gamma_low_pct: float = 50.0,
    sample_size_for_percentiles: int = 100
):
    """
    Crée et renvoie les DataLoaders (Train, Val, Test) pour le UniversalRawMSIHSIDataset.
    """
    
    dataset_kwargs = {
        "kept_indices": kept_indices,
        "patch_size": patch_size,
        "return_mask": return_mask,
        "two_zone": two_zone,
        "threshold_mode": threshold_mode,
        "tau_low_pct": tau_low_pct,
        "tau_high_pct": tau_high_pct,
        "gamma_low_pct": gamma_low_pct,
        "sample_size_for_percentiles": sample_size_for_percentiles
    }

    # Instanciation des datasets
    train_dataset = UniversalRawMSIHSIDataset(data_dir=train_dir, is_train=True, **dataset_kwargs)
    val_dataset = UniversalRawMSIHSIDataset(data_dir=val_dir, is_train=False, **dataset_kwargs)
    test_dataset = UniversalRawMSIHSIDataset(data_dir=test_dir, is_train=False, **dataset_kwargs)

    # Paramètres communs des DataLoaders
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, test_loader