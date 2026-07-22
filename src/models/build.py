# src/models/build.py

import torch
# On importe toutes tes architectures depuis ton dossier local 'models'
from models import (
    UNet, 
    GradualExpansionUNet, 
    GradualExpansionUNet_residual, 
    NAFNet, 
    DualBranchUNet, 
    DualBranchNAFNet
)

def build_model(config: dict, n_msi: int = 12, n_hsi: int = 230) -> torch.nn.Module:
    """
    Fabrique, configure et charge éventuellement les poids du modèle 
    spécifié dans la configuration.
    
    Args:
        config (dict): Le dictionnaire de configuration (chargé depuis le YAML).
        n_msi (int): Nombre de bandes de l'image Multispectrale (ex: 12).
        n_hsi (int): Nombre de bandes de l'image Hyperspectrale (ex: 230).
    """
    model_name = config["model"]["name"].lower()
    m_cfg = config["model"]
    task = config["data"].get("task", "spectral")

    print(f" Construction du modèle : {model_name.upper()} (Tâche : {task})")

    # --- 1. SELECTION ET INSTANCIATION DE L'ARCHITECTURE ---
    if model_name == "unet":
        in_channels = (n_msi + n_hsi) if task == "uncertainty" else n_msi
        model = UNet(
            in_channels=in_channels,
            activation=m_cfg["activation"],
            interpolation_mode=m_cfg["interpolation_mode"],
            learning_mode=m_cfg.get("learning_mode", "standard"),
            drop_out_rate=m_cfg.get("drop_out_rate", 0.0)
        )
        
    elif model_name == "gradualexpansionunet":
        model = GradualExpansionUNet(
            activation=m_cfg["activation"],
            interpolation_mode=m_cfg["interpolation_mode"],
            with_mlp_spectral=m_cfg.get("with_mlp_spectral", False)
        )
        
    elif model_name == "gradualexpansionunet_res":
        model = GradualExpansionUNet_residual(
            activation=m_cfg["activation"],
            interpolation_mode=m_cfg["interpolation_mode"],
            with_mlp_spectral=m_cfg.get("with_mlp_spectral", False)
        )

    elif model_name == "nafnet":
        model = NAFNet(
            in_channels=(n_msi + n_hsi),
            out_channels=n_hsi,
            width=m_cfg["width"],          
            enc_blk_nums=m_cfg["enc_blk_nums"],
            middle_blk_num=m_cfg["middle_blk_num"],   
            dec_blk_nums=m_cfg["dec_blk_nums"],
            drop_out_rate=m_cfg.get("drop_out_rate", 0.0)
        )
        
    elif model_name == "dualbranchunet":
        model = DualBranchUNet(
            n_msi=n_msi, 
            n_hsi=n_hsi, 
            base_features=64,
            interpolation_mode=m_cfg["interpolation_mode"],
            activation=m_cfg["activation"],
            final_op=m_cfg.get("final_op", "identity")
        )
        
    elif model_name == "dualbranchnafnet":
        model = DualBranchNAFNet(
            n_msi=n_msi, 
            n_hsi=n_hsi, 
            out_channels=n_hsi, 
            width=m_cfg["width"],          
            enc_blk_nums=m_cfg["enc_blk_nums"],
            middle_blk_num=m_cfg["middle_blk_num"],   
            dec_blk_nums=m_cfg["dec_blk_nums"],
            drop_out_rate=m_cfg.get("drop_out_rate", 0.0),
            final_op=m_cfg.get("final_op", "identity")
        )

    else:
        raise ValueError(f" Modèle inconnu : '{model_name}'. Vérifie ton fichier de config.")

    if m_cfg.get("load_model") is not None:
        checkpoint_path = m_cfg["load_model"]
        print(f" Chargement des poids pré-entraînés depuis : {checkpoint_path}")
        
        # Le weights_only=True évite les failles de sécurité au chargement (bonne pratique PyTorch)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    return model