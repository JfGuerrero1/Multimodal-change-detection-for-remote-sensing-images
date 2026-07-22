
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xarray as xr
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xarray as xr
from torch.utils.data import DataLoader
from .dataset import SpectralDataset
from src.constants import CACHE_DIR, INTERP_MATRIX, DATA_DIR, SRF_MATRIX

def prepare_mumucd_single_file(scene_ids, split_name, blacklist_patches=None, patch_size=256, output_dir="cache/all_data"):
    if blacklist_patches is None:
        blacklist_patches = set()
    else:
        blacklist_patches = set(blacklist_patches)

    output_path = Path(output_dir)/split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    scene_pairs = []
    for item in scene_ids:
        scene_name = item.split('-')[0]
        hsi_file = DATA_DIR / scene_name / f"{item}-prs.nc"
        msi_file = DATA_DIR / scene_name / f"{item}-s2.nc"
        
        if hsi_file.exists() and msi_file.exists():
            scene_pairs.append((msi_file, hsi_file, item))
        else:
            print(f" Fichiers introuvables pour {item}, ignoré.")

    print(f" Analyse préliminaire sur {len(scene_pairs)} scènes/items valides...")
    total_patches_global = 0
    valid_tasks = []

    for x_file, y_file, base_name in scene_pairs:
        with xr.open_dataset(y_file) as ds_y:
            h, w, _ = ds_y["sr"].shape
            h_crop = h - (h % patch_size)
            w_crop = w - (w % patch_size)
            
            for i in range(h_crop // patch_size):
                for j in range(w_crop // patch_size):
                    patch_idx = i * (w_crop // patch_size) + j
                    patch_id_str = f"{base_name}_patch_{patch_idx:04d}"
                    
                    if patch_id_str in blacklist_patches:
                        continue
                        
                    valid_tasks.append((x_file, y_file, base_name, i * patch_size, j * patch_size, patch_id_str))
                    total_patches_global += 1

    c_hsi = INTERP_MATRIX.shape[0]  
    c_multi = SRF_MATRIX.shape[1]   

    f_msi_real = h5py.File(output_path / "msi_real.h5", 'w')
    f_msi_sim = h5py.File(output_path / "msi_simulated.h5", 'w')
    f_msi_norm = h5py.File(output_path / "msi_normalized.h5", 'w')
    f_hsi = h5py.File(output_path / "hsi.h5", 'w')
    f_interp = h5py.File(output_path / "hsi_interp.h5", 'w')

    dset_msi_real = f_msi_real.create_dataset('data', shape=(total_patches_global, c_multi, patch_size, patch_size), dtype=np.float32, chunks=True, compression="gzip")
    dset_msi_sim = f_msi_sim.create_dataset('data', shape=(total_patches_global, c_multi, patch_size, patch_size), dtype=np.float32, chunks=True, compression="gzip")
    dset_msi_norm = f_msi_norm.create_dataset('data', shape=(total_patches_global, c_multi, patch_size, patch_size), dtype=np.float32, chunks=True, compression="gzip")
    dset_hsi = f_hsi.create_dataset('data', shape=(total_patches_global, c_hsi, patch_size, patch_size), dtype=np.float32, chunks=True, compression="gzip")
    dset_interp = f_interp.create_dataset('data', shape=(total_patches_global, c_hsi, patch_size, patch_size), dtype=np.float32, chunks=True, compression="gzip")

    print(f" Écriture des 5 gros fichiers HDF5 pour {total_patches_global} patches...")

    current_idx = 0
    current_base_name = None
    ds_x, ds_y = None, None
    scene_multi_norm_full = None
    scene_multi_sim = None
    scene_interp_full = None

    for x_file, y_file, base_name, r, c, patch_id_str in tqdm(valid_tasks):
        
        if current_base_name != base_name:
            if ds_x is not None:
                ds_x.close()
                ds_y.close()
            
            ds_x = xr.open_dataset(x_file)
            ds_y = xr.open_dataset(y_file)
            current_base_name = base_name
            print(f"Traitement de la scène {current_base_name}")

            patch_multi_real_full = ds_x["sr"].to_numpy().astype(np.float32)
            patch_hyper_real_full = ds_y["sr"].to_numpy().astype(np.float32)
            h, w, _ = patch_hyper_real_full.shape
        
            # MSI SIMULÉ GLOBAL (HWC)
            hyper_2d = patch_hyper_real_full.reshape(-1, c_hsi)
            scene_multi_sim = np.dot(hyper_2d, SRF_MATRIX).reshape(h, w, c_multi).astype(np.float32)

            # MSI_NORMALISE GLOBAL (HWC)
            mean_scene = np.mean(patch_multi_real_full, axis=(0, 1), keepdims=True)
            std_scene = np.std(patch_multi_real_full, axis=(0, 1), keepdims=True)
            scene_multi_norm_full = (patch_multi_real_full - mean_scene) / (std_scene + 1e-8)
            std_sim = np.std(scene_multi_sim, axis=(0, 1), keepdims=True)
            mean_sim = np.mean(scene_multi_sim, axis=(0, 1), keepdims=True)
            scene_multi_norm_full = scene_multi_norm_full * std_sim + mean_sim

            # HSI_INTERPOLE GLOBAL (CHW)
            m_real_chw_full = np.transpose(patch_multi_real_full, (2, 0, 1))
            interp_numpy = INTERP_MATRIX @ m_real_chw_full.reshape(c_multi, -1)
            scene_interp_full = interp_numpy.reshape(c_hsi, h, w)

        # Extraction des patches (en HWC pour les MSI/HSI de base)
        patch_hyper = ds_y["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32)
        patch_multi_real = ds_x["sr"][r:r+patch_size, c:c+patch_size, :].to_numpy().astype(np.float32)
        patch_multi_norm = scene_multi_norm_full[r:r+patch_size, c:c+patch_size, :]
        patch_multi_sim = scene_multi_sim[r:r+patch_size, c:c+patch_size, :]
        
        # Extraction du patch interpolé déjà calculé globalement (CHW)
        interp_chw_patch = scene_interp_full[:, r:r+patch_size, c:c+patch_size]

        # Transposition HWC -> CHW pour les autres
        m_real_chw = np.transpose(patch_multi_real, (2, 0, 1))
        m_sim_chw = np.transpose(patch_multi_sim, (2, 0, 1))
        m_norm_chw = np.transpose(patch_multi_norm, (2, 0, 1))
        hsi_chw = np.transpose(patch_hyper, (2, 0, 1))

        # Stockage direct
        dset_msi_real[current_idx] = m_real_chw
        dset_msi_sim[current_idx] = m_sim_chw
        dset_msi_norm[current_idx] = m_norm_chw
        dset_hsi[current_idx] = hsi_chw
        dset_interp[current_idx] = interp_chw_patch

        current_idx += 1

    if ds_x is not None:
        ds_x.close()
        ds_y.close()

    f_msi_real.close()
    f_msi_sim.close()
    f_msi_norm.close()
    f_hsi.close()
    f_interp.close()

    print(" Génération des 5 fichiers HDF5 globaux terminée avec succès !")



def create_data_loaders_spectral(simulated, augment,augment_illumination, batch_size=8, num_workers=4, is_residual=False, is_normalized=False, mask=None):
    """
    Crée les DataLoaders PyTorch en utilisant les fichiers HDF5 centralisés.
    """
    train_dir = CACHE_DIR /  'train'
    val_dir = CACHE_DIR /  'val'
    test_dir = CACHE_DIR /'test'
    
    train_dataset = SpectralDataset(
        dataset_dir=train_dir, 
        simulated=simulated, 
        is_normalized=is_normalized, 
        augment=augment, 
        augment_illumination=augment_illumination,
        is_residual=is_residual, 
        mask=mask
    )
    
    val_dataset = SpectralDataset(
        dataset_dir=val_dir, 
        simulated=simulated, 
        is_normalized=is_normalized, 
        augment=False, 
        augment_illumination=False,
        is_residual=is_residual, 
        mask=mask
    )
    
    test_dataset = SpectralDataset(
        dataset_dir=test_dir, 
        simulated=simulated, 
        is_normalized=is_normalized, 
        augment=False, 
        augment_illumination=False,
        is_residual=is_residual, 
        mask=mask
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    )