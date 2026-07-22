
import torch
import h5py
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class SpectralDataset(Dataset):
    def __init__(self, dataset_dir, simulated=False, is_normalized=False, augment=False,augment_illumination=False, is_residual=False, mask=None):
        """
        dataset_dir : Dossier contenant les fichiers HDF5 (ex: Path("cache/all_data"))
        use_simulated_msi : Booléen pour utiliser le MSI simulé au lieu du MSI réel
        is_normalized : Booléen pour utiliser le MSI normalisé
        """
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment
        self.is_residual = is_residual
        self.simulated = simulated
        self.is_normalized = is_normalized
        self.indices_to_use = np.where(mask)[0] if mask is not None else slice(None)
        self.is_color_augmented=augment_illumination
        
        self.h5_files = {}
        self.datasets = {}
        self.load_h5_handles()

    def load_h5_handles(self):
        # Sélection du fichier MSI source selon les options
        if self.is_normalized:
            msi_filename = "msi_normalized.h5"
        elif self.use_simulated_msi:
            msi_filename = "msi_simulated.h5"
        else:
            msi_filename = "msi_real.h5"

        paths = {
            'msi': self.dataset_dir / msi_filename,
            'hsi': self.dataset_dir / "hsi.h5",
            'interp': self.dataset_dir / "hsi_interp.h5"
        }

    def augment_color_jittering(self, x):
        if random.random() < 0.3:  
            factor = random.uniform(0.97, 1.03)
            offset = random.uniform(-0.003, 0.003)
            x = x * factor + offset    
        return x
     
    def augment_pair(self, x, y):
        if random.random() < 0.5:
            x ,y= torch.flip(x, dims=[2]), torch.flip(y, dims=[2])
        if random.random() < 0.5:
            x,y = torch.flip(x, dims=[1]), torch.flip(y, dims=[1])
        k = torch.randint(0, 4, (1,)).item()

        x,y = torch.rot90(x, k, dims=[1,2]),torch.rot90(y, k, dims=[1,2])
        return x, y
    
    def augment_triplet(self, x, x_interp, y):
        if random.random() < 0.5:
            x, x_interp, y = (torch.flip(x, dims=[2]),torch.flip(x_interp, dims=[2]),torch.flip(y, dims=[2]),)
        if random.random() < 0.5:
            x, x_interp, y = (torch.flip(x, dims=[1]),torch.flip(x_interp, dims=[1]),torch.flip(y, dims=[1]),)
        k = torch.randint(0, 4, (1,)).item()

        x,x_interp,y= torch.rot90(x, k, dims=[1, 2]),torch.rot90(x_interp, k, dims=[1, 2]), torch.rot90(y, k, dims=[1, 2])
        return x, x_interp, y

  

    def __getitem__(self, idx):
         
        x_patch = self.dset_msi[idx]  # shape (C_msi, H, W)
        y_patch = self.dset_hsi[idx]  # shape (C_hsi, H, W)
        
        y_patch = y_patch[self.indices_to_use, :, :]
        
        x = torch.from_numpy(x_patch).float()
        y = torch.from_numpy(y_patch).float()

        if self.is_color_augmented:
            x=self.augment_color_jittering(self,x)
        
        if self.is_residual:
            x_interp_patch = self.dset_interp[idx]
            x_interp_patch = x_interp_patch[self.indices_to_use, :, :]
            x_interp = torch.from_numpy(x_interp_patch).float()
            if self.augment:
                x, x_interp, y = self.augment_triplet(x, x_interp, y)
                return x, x_interp, y
            else:
                if self.augment:
                    x, y = self.augment_pair(x, y)
                return x, y


class UncertaintyDataset(Dataset):
    def __init__(self, dataset_dir, use_simulated_msi=False, is_normalized=False, augment=False, augment_illumination=False, mask=None):
        """
        dataset_dir : Dossier contenant les fichiers HDF5 (ex: Path("cache/all_data"))
        """
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment
        self.use_simulated_msi = use_simulated_msi
        self.is_normalized = is_normalized
        self.indices_to_use = np.where(mask)[0] if mask is not None else slice(None)
        self.indices_to_use = np.where(mask)[0] if mask is not None else slice(None)
        self.is_color_augmented=augment_illumination
        
        self.load_h5_handles()

    def load_h5_handles(self):
        # Sélection du fichier MSI source
        if self.is_normalized:
            msi_filename = "msi_normalized.h5"
        elif self.use_simulated_msi:
            msi_filename = "msi_simulated.h5"
        else:
            msi_filename = "msi_real.h5"

        paths = {
            'msi': self.dataset_dir / msi_filename,
            'hsi_true': self.dataset_dir / "hsi.h5",
            'hsi_sim': self.dataset_dir / "hsi_interp.h5"  # ou msi_simulated interpolé selon ton usage initial
        }

        for key, path in paths.items():
            assert path.exists(), f"❌ Fichier HDF5 introuvable : {path}"

        self.f_msi = h5py.File(paths['msi'], 'r')
        self.f_hsi_true = h5py.File(paths['hsi_true'], 'r')
        self.f_hsi_sim = h5py.File(paths['hsi_sim'], 'r')
        
        self.dset_msi = self.f_msi['data']
        self.dset_hsi_true = self.f_hsi_true['data']
        self.dset_hsi_sim = self.f_hsi_sim['data']
        
        self.total_length = self.dset_msi.shape[0]
        print(f"[{self.dataset_dir.name.upper()}] Chargé {self.total_length} patches pour l'incertitude (MSI: {msi_filename}).")

    def __len__(self):
        return self.total_length

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
        x_msi = torch.rot90(x_msi, k, dims=[1, 2])
        x_hsi = torch.rot90(x_hsi, k, dims=[1, 2])
        y_res = torch.rot90(y_res, k, dims=[1, 2])

        return x_msi, x_hsi, y_res

    def __getitem__(self, idx):
        # Lecture directe et rapide via h5py
        msi = torch.from_numpy(self.dset_msi[idx]).float()
        hsi_true = torch.from_numpy(self.dset_hsi_true[idx]).float()
        hsi_sim = torch.from_numpy(self.dset_hsi_sim[idx]).float()

        # Application du masque spectral si nécessaire
        hsi_true = hsi_true[self.indices_to_use, :, :]
        hsi_sim = hsi_sim[self.indices_to_use, :, :]

        X = torch.cat([msi, hsi_sim], dim=0)
        y = torch.abs(hsi_true - hsi_sim)

        if self.augment:
            c_msi = msi.shape[0]
            msi_aug, hsi_sim_aug, y = self.augment_triplet(X[:c_msi], X[c_msi:], y)
            X = torch.cat([msi_aug, hsi_sim_aug], dim=0)

        return X, y 
