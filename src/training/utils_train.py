# src/train/commons.py

import os
import random
import shutil
from pathlib import Path
import numpy as np
import torch
import yaml
import wandb

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

def init_wandb(config: dict, run_name_prefix: str = "") -> str:
    """Initialise le tracking Weights & Biases de manière unifiée."""
    task = config["data"].get("task", "unknown")
    model_name = config["model"]["name"]
    loss_name = config["training"]["loss_type"]
    
    run_name = f"{run_name_prefix}{model_name}_{task}_{loss_name}"
    
    wandb.init(
        entity=config["experiment"]["entity"],
        project=config["experiment"]["project"],
        name=run_name,
        config=config
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

