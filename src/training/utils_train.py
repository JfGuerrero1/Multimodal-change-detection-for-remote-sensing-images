# src/train/commons.py

import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
import yaml
import wandb
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import torch


class EarlyStopping:

  def __init__(
      self, patience=15, min_delta=1e-4, verbose=True, start_epoch=10
  ):
    """patience : Nombre d'époques à attendre sans amélioration min_delta :

    Amélioration minimale considérée comme significative start_epoch : Époque à
    partir de laquelle l'Early Stopping s'active
    """
    self.patience = patience
    self.min_delta = min_delta
    self.verbose = verbose
    self.start_epoch = start_epoch

    self.counter = 0
    self.best_loss = np.inf
    self.early_stop = False

  def __call__(self, val_loss, epoch):
    # Ne pas déclencher l'early stopping durant les toutes premières époques (ex: warmup)
    if epoch < self.start_epoch:
      return

    if val_loss < self.best_loss - self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
    else:
      self.counter += 1
      if self.verbose:
        print(
          f"  ⏳ EarlyStopping : {self.counter}/{self.patience} époques sans"
          f" amélioration (Meilleure Val Loss: {self.best_loss:.6f})"
        )

      if self.counter >= self.patience:
        self.early_stop = True
def build_run_name(config: dict) -> str:
    """Construit un nom de run exhaustif, structuré et aligné sur la config YAML."""
    m = config.get("model", {})
    d = config.get("data", {})
    t = config.get("training", {})
    wl = config.get("wavelength_filtering", {})

    # 1. Architecture & Modèle
    model_name = m.get("name", "unet")
    mlp_mode = "mlp" if m.get("with_mlp_spectral", False) else "nomlp"
    
    # 2. Données & Filtrage Spectral (Crucial pour éviter les size mismatches)
    sim_mode = "sim" if d.get("simulated", False) else "real"
    aug_mode = "aug" if d.get("augment", False) else "noaug"
    res_mode = "res" if d.get("is_residual", True) else "nores"
    wl_mode = "filtered" if wl.get("enabled", False) else "fullspec"

    # 3. Entraînement & Optimisation
    optimizer = t.get("optimizer_name", "adamw")
    lr_val = t.get("lr", 1e-4)
    lr_str = f"lr{lr_val:.0e}" if lr_val < 0.01 else f"lr{lr_val}"
    batch_size = f"b{d.get('batch_size', 8)}"

    # 4. Loss
    loss_components = (
        f"sam{t.get('lambda_sam', 0.5)}_"
        f"mse{t.get('lambda_mse', 0.0)}_"
        f"mae{t.get('lambda_mae', 1.0)}"
    )

    # Assemblage final
    run_parts = [
        model_name,
        wl_mode,        # <-- Garantit la distinction 195b vs 230b
        sim_mode,
        aug_mode,
        mlp_mode,
        res_mode,
        optimizer,
        lr_str,
        batch_size,
        loss_components,
    ]

    return "_".join(str(p) for p in run_parts)

def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Construit le scheduler LR dynamiquement selon la config YAML."""
    train_cfg = config.get("training", {})
    
    warm_up = train_cfg.get("warm_up", False)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    eta_min = train_cfg.get("eta_min", 1e-6)
    scheduler_type = train_cfg.get("scheduler_type", "cosine_restarts").lower()
    
    # 1. Sélection du scheduler principal
    if scheduler_type == "cosine_restarts":
        t_0 = train_cfg.get("T_0", 50)
        t_mult = train_cfg.get("T_mult", 1)
        main_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
        )
    elif scheduler_type == "cosine":
        epochs = train_cfg.get("epochs", 100)
        main_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )
    elif scheduler_type == "plateau":
        patience = train_cfg.get("patience_plateau", 5)
        # Note : ReduceLROnPlateau s'appelle avec .step(val_loss) dans la boucle d'entraînement
        main_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=0.5, min_lr=eta_min
        )
        print(f" Scheduler configuré : ReduceLROnPlateau (patience={patience})")
        return main_scheduler # Le plateau se gère souvent seul sans warm-up séquentiel complexe
    else:
        raise ValueError(f" Type de scheduler inconnu : '{scheduler_type}'. Choisis parmi ['cosine_restarts', 'cosine', 'plateau'].")

    # 2. Gestion du Warm-up (uniquement pour les schedulers basés sur les époques)
    if not warm_up:
        print(f" Scheduler configuré : {scheduler_type} (sans warm-up)")
        return main_scheduler
    
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )

    scheduler = lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    
    print(f" Scheduler configuré : Warm-up ({warmup_epochs} epochs) + {scheduler_type}")
    return scheduler


def get_kept_wavelength_indices(wvl_full: np.ndarray, config: dict) -> np.ndarray:
  wv_cfg = config.get("wavelength_filtering", {})

  if not wv_cfg.get("enabled", False):
    return np.ones(len(wvl_full), dtype=bool)

  # On initialise le masque à True (toutes les bandes sont gardées par défaut)
  kept_mask = np.ones(len(wvl_full), dtype=bool)

  # 1. Exclusion par fenêtres d'absorption (ex: [1350, 1500])
  excluded_windows = wv_cfg.get("excluded_windows", [])
  for w_min, w_max in excluded_windows:
    window_mask = (wvl_full >= w_min) & (wvl_full <= w_max)
    kept_mask &= ~window_mask  # On passe à False les bandes dans la fenêtre

  # 2. Exclusion par indices de bandes spécifiques (si spécifié)
  excluded_bands = wv_cfg.get("excluded_bands", [])
  if excluded_bands:
    kept_mask[excluded_bands] = False

  return kept_mask

def set_seed(seed: int = 42):
    """Garantit la reproductibilité des expériences sur CPU et GPU."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Détecte automatiquement si CUDA est disponible."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device détecté : {device}")
    return device

def get_project_root() -> Path:
    """Récupère proprement la racine du projet (Multimodal-change-detection/)."""
    return Path(__file__).resolve().parents[2]

def load_config(config_filename: str) -> dict:
    """Charge un fichier de configuration YAML situé à la racine du projet."""
    root = get_project_root()
    config_path = root / config_filename
    
    if not config_path.exists():
        raise FileNotFoundError(f" Le fichier de configuration {config_filename} n'existe pas à la racine ({root}).")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def init_wandb(
    config: dict, run_name: str = None, run_name_prefix: str = ""
) -> str:
  """Initialise le tracking Weights & Biases de manière unifiée."""
  # Si un nom de run complet est passé, on l'utilise
  if run_name is None:
    task = config["data"].get("task", "unknown")
    model_name = config["model"]["name"]
    loss_name = config["training"]["loss_type"]
    run_name = f"{run_name_prefix}{model_name}_{task}_{loss_name}"
  elif run_name_prefix:
    run_name = f"{run_name_prefix}_{run_name}"

  wandb.init(
      entity=config["experiment"]["entity"],
      project=config["experiment"]["project"],
      name=run_name,
      config=config,
  )
  return run_name
def save_checkpoint(model: torch.nn.Module, config: dict, run_name: str, is_best: bool = False):
    """
    Sauvegarde les poids du modèle et effectue une copie du fichier de config associé.
    Très utile pour retrouver les hyperparamètres d'un run spécifique plus tard.
    """
    root = get_project_root()
    checkpoint_dir = root / config["experiment"]["output_dir"] / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde des poids
    filename = "best_model.pth" if is_best else "latest_model.pth"
    torch.save(model.state_dict(), checkpoint_dir / filename)
    
    # Sauvegarde de la configuration associée (copie physique du YAML utilisé)
    config_name = f"config_{config['data']['task']}.yaml"
    if (root / config_name).exists():
        shutil.copy(root / config_name, checkpoint_dir / f"reproduced_{config_name}")
def build_optimizer(config: dict, model_params) -> torch.optim.Optimizer:
    """
    Instancie l'optimiseur spécifié dans la configuration.
    """
    # Si la section optimizer n'existe pas, on met AdamW par défaut
    opt_cfg = config.get("training", {})
    opt_name = opt_cfg.get("optimizer_name", "adamw").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)

    print(f" Instanciation de l'optimiseur : {opt_name.upper()} (lr={lr})")

    if opt_name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        # Pour SGD, on ajoute souvent le momentum (0.9 par défaut)
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f" Optimiseur inconnu : '{opt_name}'. Choisis parmi ['adamw', 'adam', 'sgd'].")

def build_lr_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Construit le scheduler LR selon la config."""
    if not config["training"].get("warm_up", False):
        # Pas de warmup, juste un cosine annealing simple
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=1e-6
        )
    
    # Avec warmup : LinearLR 3 epochs + CosineAnnealing
    warmup_epochs = 3
    total_epochs = config["training"]["epochs"]
    
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    
    cosine_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6
    )

    scheduler = lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    return scheduler


def build_run_name(config: dict) -> str:
    """Construit le nom du run depuis la config."""
    model_name = config["model"]["name"]
    data_mode = "sim" if config["data"]["simulated"] else "real"
    aug_mode = "aug" if config["data"]["augment"] else "noaug"
    mlp_mode = "mlp" if config["model"]["with_mlp_spectral"] else "no_mlp"
    
    loss_components = (
        f"SAM-{config['training']['lambda_sam']}_"
        f"MSE-{config['training']['lambda_mse']}_"
        f"MAE-{config['training']['lambda_mae']}"
    )
    
    run_name = (
        f"{model_name}_{data_mode}_{aug_mode}_"
        f"{loss_components}_lr-{config['training']['lr']}_{mlp_mode}_residual"
    )
    return run_name


