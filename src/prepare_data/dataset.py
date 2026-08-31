from pathlib import Path
import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import json

import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.constants import INTERP_MATRIX

class SpectralDataset(Dataset):

  def __init__(
      self,
      dataset_dir,
      simulated=False,
      is_normalised=False,
      augment=False,
      augment_illumination=False,
      is_residual=False,
      kept_indices=None,
  ):
    """dataset_dir : Dossier contenant les fichiers binaires et metadata.json (ex: Path("cache/all_data_memmap/train"))

    simulated : Booléen pour utiliser le MSI simulé au lieu du MSI réel
    is_normalised : Booléen pour utiliser le MSI normalisé augment : Data
    augmentation spatiale (Flips / Rotations) augment_illumination : Jittering
    sur les valeurs du MSI et de l'interpolation is_residual : True si le modèle prend
    (x_msi, x_interp) en entrée kept_indices : Masque booléen/indices pour
    filtrer les bandes spectro
    """
    self.dataset_dir = Path(dataset_dir)
    self.augment = augment
    self.is_residual = is_residual
    self.simulated = simulated
    self.is_normalised = is_normalised

    if kept_indices is None:
      self.indices_to_use = None
    else:
      self.indices_to_use = kept_indices

    self.is_color_augmented = augment_illumination

    # Sélection du nom du fichier MSI source
    if self.is_normalised:
      self.msi_filename = "msi_normalised.bin"
    elif self.simulated:
      self.msi_filename = "msi_simulated.bin"
    else:
      self.msi_filename = "msi_real.bin"

    self.paths = {
        "msi": self.dataset_dir / self.msi_filename,
        "hsi": self.dataset_dir / "hsi.bin",
        "interp": self.dataset_dir / "hsi_interp.bin",
        "meta": self.dataset_dir / "metadata.json",
    }

    # Lecture des métadonnées JSON
    if not self.paths["meta"].exists():
      raise FileNotFoundError(
          f"Le fichier metadata.json est introuvable dans {self.dataset_dir}"
      )

    with open(self.paths["meta"], "r", encoding="utf-8") as f:
      self.meta = json.load(f)

    self.length = self.meta["total_patches"]
    self.msi_shape = tuple(self.meta["msi_shape"])
    self.hsi_shape = tuple(self.meta["hsi_shape"])
    self.dtype = self.meta["dtype"]
    self.patch_ids = self.meta["patch_ids"]

    # Pointers memmap (lazy loading pour compatibilité multi-worker DataLoader)
    self.fp_msi = None
    self.fp_hsi = None
    self.fp_interp = None

  def _init_memmap(self):
    """Ouvre les fichiers memmap en lecture seule au démarrage de chaque worker."""
    if self.fp_msi is not None:
      return  # Déjà ouvert dans ce worker

    try:
      self.fp_msi = np.memmap(
          self.paths["msi"],
          dtype=self.dtype,
          mode="r",
          shape=self.msi_shape,
      )
      self.fp_hsi = np.memmap(
          self.paths["hsi"],
          dtype=self.dtype,
          mode="r",
          shape=self.hsi_shape,
      )

      if self.is_residual:
        if not self.paths["interp"].exists():
          raise FileNotFoundError(
              f"Le fichier requis {self.paths['interp']} est introuvable !"
          )
        self.fp_interp = np.memmap(
            self.paths["interp"],
            dtype=self.dtype,
            mode="r",
            shape=self.hsi_shape,
        )

    except Exception as e:
      print(
          f"❌ Erreur lors de l'ouverture des fichiers memmap dans"
          f" {self.dataset_dir} : {e}"
      )
      raise e

  def __len__(self):
    return self.length

  def augment_color_jittering(self, *tensors, p=0.3):
    """
    Jittering d'illumination appliqué uniquement sur les entrées (*tensors),
    laissant la vérité terrain (y) strictement intacte.
    """
    if random.random() < p:
      factor = random.uniform(0.97, 1.03)
      offset = random.uniform(-0.003, 0.003)
      tensors = [t * factor + offset for t in tensors]
    return tensors

  def augment_geometric(self, *tensors, p_flip=0.5):
    """
    Applique des transformations géométriques synchrones (flips et rotations) 
    sur un nombre arbitraire de tenseurs passés en arguments (*tensors), 
    incluant cette fois la vérité terrain (y) car la géométrie les concerne tous.
    """
    # 1. Flip horizontal (axe des colonnes, dims=[2])
    if random.random() < p_flip:
      tensors = [torch.flip(t, dims=[2]) for t in tensors]
        
    # 2. Flip vertical (axe des lignes, dims=[1])
    if random.random() < p_flip:
      tensors = [torch.flip(t, dims=[1]) for t in tensors]
        
    # 3. Rotation aléatoire par multiples de 90°
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
      tensors = [torch.rot90(t, k, dims=[1, 2]) for t in tensors]
        
    return tensors
    
  def __getitem__(self, idx):
    # Garantit que les fichiers memmap sont ouverts dans le worker actuel
    if self.fp_msi is None:
      self._init_memmap()

    # Lecture memmap directe (.copy() pour instancier un array propre en RAM)
    x_patch = self.fp_msi[idx].copy()  # (C_msi, H, W)
    y_patch = self.fp_hsi[idx].copy()  # (C_hsi, H, W)

    # ID du patch depuis les métadonnées
    patch_id = self.patch_ids[idx]

    # Filtrage des bandes atmosphériques sur la GT (HSI)
    if self.indices_to_use is not None:
      y_patch = y_patch[self.indices_to_use, :, :]

    x = torch.from_numpy(x_patch).float()
    y = torch.from_numpy(y_patch).float()

    #  CAS 1 : Modèle Résiduel (MSI, HSI_interp, HSI_gt)
  #  CAS 1 : Modèle Résiduel (MSI, HSI_interp, HSI_gt)
    if self.is_residual:
      if self.is_normalised:
        c_multi = x_patch.shape[0]
        
        # 1. On récupère les dimensions HSI complètes (avant filtrage) depuis metadata
        c_hsi_full = self.hsi_shape[1]
        h, w = self.hsi_shape[2], self.hsi_shape[3]

        # 2. Calcul avec la matrice d'interpolation (taille complète)
        interp_numpy = INTERP_MATRIX @ x_patch.reshape(c_multi, -1)
        x_interp_numpy = interp_numpy.reshape(c_hsi_full, h, w).astype(np.float32)
      else:
        # 3. Lecture directe du fichier interp (taille complète aussi)
        x_interp_patch = self.fp_interp[idx].copy()
        x_interp_numpy = x_interp_patch.astype(np.float32)

      # 4. FILTRAGE APRÈS : On applique les indices sur le tenseur complet
      if self.indices_to_use is not None:
        x_interp_numpy = x_interp_numpy[self.indices_to_use, :, :]

      x_interp = torch.from_numpy(x_interp_numpy).float()

      # Color Jittering appliqué SEULEMENT sur x et x_interp (y reste intact)
      if self.is_color_augmented:
        x, x_interp = self.augment_color_jittering(x, x_interp)

      # Augmentation géométrique appliquée sur tout le triplet (x, x_interp, y)
      if self.augment:
        x, x_interp, y = self.augment_geometric(x, x_interp, y)

      return x, x_interp, y, patch_id


    #  CAS 2 : Modèle Standard (MSI, HSI_gt)
    else:
      # Color Jittering appliqué SEULEMENT sur x (y reste intact)
      if self.is_color_augmented:
        x, = self.augment_color_jittering(x)

      # Augmentation géométrique appliquée sur la paire (x, y)
      if self.augment:
        x, y = self.augment_geometric(x, y)

      return x, y, patch_id