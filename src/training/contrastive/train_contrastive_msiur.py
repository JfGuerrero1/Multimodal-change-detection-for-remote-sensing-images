import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt

from src.constants import WVL_PRS
from src.models.contrastive import SiameseNetwork_heterogene

# Import de notre dataset universel brut et de sa fonction de dataloader
from src.prepare_data.dataset import UniversalRawMSIHSIDataset

from src.training.utils_train import (
    set_seed, 
    get_device, 
    get_project_root, 
    load_config,
    init_wandb, 
    save_checkpoint, 
    build_optimizer,
    EarlyStopping,
    build_run_name,
    build_lr_scheduler,
    get_kept_wavelength_indices
)

# --- Données Spectrales ---
WVL_S2 = np.array([443., 490., 560., 665., 705., 740., 783., 842., 865., 940., 1610., 2190.])


# ============================================================================
# FONCTIONS UTILITAIRES & MONITORING
# ============================================================================

def build_model(config: dict, n_msi: int, n_hsi: int) -> torch.nn.Module:
    """Instancie le modèle selon la configuration YAML."""
    model_cfg = config["model"]
    name = model_cfg.get("name", "").lower()

    if name == "siamese_heterogene":
        model = SiameseNetwork_heterogene(
            in_channel_msi=n_msi,
            in_channel_hsi=n_hsi,
            base_channels=model_cfg.get("base_channels", 64),
            latent_dim=model_cfg.get("latent_dim", 128),
        )
    else:
        raise ValueError(f"❌ Modèle non reconnu : '{name}'")

    return model


def get_rgb_indices_for_modalities(wvl_s2, wvl_prs):
    """
    Trouve les indices des canaux RGB pour le MSI (Sentinel-2) 
    et les indices les plus proches dans le HSI (Prisma).
    """
    target_blue = 490.0
    target_green = 560.0
    target_red = 665.0
    
    # Pour le MSI (WVL_S2)
    s2_red_idx = np.argmin(np.abs(wvl_s2 - target_red))
    s2_green_idx = np.argmin(np.abs(wvl_s2 - target_green))
    s2_blue_idx = np.argmin(np.abs(wvl_s2 - target_blue))
    s2_rgb_indices = [s2_red_idx, s2_green_idx, s2_blue_idx]
    
    # Pour le HSI complet (WVL_PRS)
    prs_red_idx = np.argmin(np.abs(wvl_prs - target_red))
    prs_green_idx = np.argmin(np.abs(wvl_prs - target_green))
    prs_blue_idx = np.argmin(np.abs(wvl_prs - target_blue))
    prs_rgb_indices = [prs_red_idx, prs_green_idx, prs_blue_idx]
    
    return s2_rgb_indices, prs_rgb_indices


def log_change_detection_map_to_wandb_with_diff(
    model,
    batch_hsi,
    batch_msi,  # Shape: [B, 12+C_rec+C_unc, H, W]
    batch_mask=None,
    kept_indices=None,
    device="cuda",
    tag_suffix="epoch"
):
    """
    Logue les visualisations sur WandB en affichant :
    1. MSI RGB
    2. HSI RGB
    3. Reconstruction RGB
    4. Incertitude moyenne
    5. Score de distance entre embeddings
    """
    model.eval()
    
    with torch.no_grad():
        batch_msi = batch_msi.to(device)
        batch_hsi = batch_hsi.to(device)
        
        # Extraction dynamique basée sur les dimensions
        n_msi_bands = 12  # Nombre de bandes MSI de base (Sentinel-2)
        n_filtered = batch_hsi.shape[1] # Nombre de bandes HSI filtrées
        
        msi = batch_msi[:, :n_msi_bands, :, :]
        reconstruction = batch_msi[:, n_msi_bands : n_msi_bands + n_filtered, :, :]
        uncertainty = batch_msi[:, n_msi_bands + n_filtered :, :, :]
        hsi = batch_hsi
        
        # Forward du modèle
        z_msi, z_hsi = model(batch_msi, batch_hsi)
        
        # Calcul du score (distance L2 normalisée)
        if z_msi.dim() == 4 and z_hsi.dim() == 4:
            z_msi_norm = F.normalize(z_msi, p=2, dim=1)
            z_hsi_norm = F.normalize(z_hsi, p=2, dim=1)
            scores = torch.norm(z_msi_norm - z_hsi_norm, p=2, dim=1, keepdim=True)
        else:
            B, H, W = batch_hsi.shape[0], batch_hsi.shape[2], batch_hsi.shape[3]
            scores = torch.zeros(B, 1, H, W, device=device)
        
        # Conversion numpy
        scores_np = scores.cpu().numpy()
        msi_np = msi.cpu().numpy()
        rec_np = reconstruction.cpu().numpy()
        unc_np = uncertainty.cpu().numpy()
        hsi_np = hsi.cpu().numpy()
        mask_np = batch_mask.cpu().numpy() if batch_mask is not None else None
        
        # Indices RGB
        s2_rgb_idx, prs_rgb_idx_full = get_rgb_indices_for_modalities(WVL_S2, WVL_PRS)
        
        if kept_indices is not None:
            kept_indices_np = (
                kept_indices.cpu().numpy()
                if isinstance(kept_indices, torch.Tensor)
                else np.array(kept_indices)
            )
            valid_indices = np.where(kept_indices_np)[0]
            prs_rgb_idx = []
            for target_idx in prs_rgb_idx_full:
                closest_idx = valid_indices[np.argmin(np.abs(valid_indices - target_idx))]
                filtered_position = np.where(valid_indices == closest_idx)[0][0]
                prs_rgb_idx.append(filtered_position)
        else:
            prs_rgb_idx = prs_rgb_idx_full
        
        # Visualisation: 6 colonnes (MSI, HSI, Reconstruction, Incertitude, Score, Masque)
        batch_size = msi_np.shape[0]
        n_to_plot = min(batch_size, 4)
        
        fig, axes = plt.subplots(n_to_plot, 5, figsize=(30, 4 * n_to_plot))
        if n_to_plot == 1:
            axes = np.expand_dims(axes, 0)
        
        for i in range(n_to_plot):
            # 1. Col 1: MSI RGB
            try:
                rgb_msi = np.transpose(msi_np[i, s2_rgb_idx], (1, 2, 0))
                rgb_msi = np.clip((rgb_msi - rgb_msi.min()) / (rgb_msi.max() - rgb_msi.min() + 1e-8), 0, 1)
            except:
                rgb_msi = msi_np[i, 0]
            axes[i, 0].imshow(rgb_msi)
            axes[i, 0].set_title("MSI RGB (S2)")
            axes[i, 0].axis('off')
            
            # 2. Col 2: HSI RGB
            try:
                rgb_hsi = np.transpose(hsi_np[i, prs_rgb_idx], (1, 2, 0))
                rgb_hsi = np.clip((rgb_hsi - rgb_hsi.min()) / (rgb_hsi.max() - rgb_hsi.min() + 1e-8), 0, 1)
            except:
                rgb_hsi = hsi_np[i, 0]
            axes[i, 1].imshow(rgb_hsi)
            axes[i, 1].set_title("HSI RGB")
            axes[i, 1].axis('off')

            # 3. Col 3: Reconstruction RGB
            try:
                rgb_rec = np.transpose(rec_np[i, prs_rgb_idx], (1, 2, 0))
                rgb_rec = np.clip((rgb_rec - rgb_rec.min()) / (rgb_rec.max() - rgb_rec.min() + 1e-8), 0, 1)
            except:
                rgb_rec = rec_np[i, 0]
            axes[i, 2].imshow(rgb_rec)
            axes[i, 2].set_title("Reconstruction RGB")
            axes[i, 2].axis('off')

            # 4. Col 4: Incertitude moyenne sur les bandes
            mean_unc = np.mean(unc_np[i], axis=0)
            im_unc = axes[i, 3].imshow(mean_unc, cmap='inferno')
            axes[i, 3].set_title("Incertitude Moyenne")
            axes[i, 3].axis('off')
            fig.colorbar(im_unc, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            # 5. Col 5: Score de distance
            pred_map = scores_np[i, 0]
            im_score = axes[i, 4].imshow(pred_map, cmap='jet')
            axes[i, 4].set_title("Score Contrastif")
            axes[i, 4].axis('off')
            fig.colorbar(im_score, ax=axes[i, 4], fraction=0.046, pad=0.04)


        
        plt.tight_layout()
        wandb.log({f"change_detection_maps_diff/{tag_suffix}": wandb.Image(fig)})
        plt.close(fig)
        print(f"✅ Visualisation enrichie loggée: {tag_suffix}")


def train_one_epoch(model, dataloader, optimizer, criterion_infonce, device, scaler=None, max_grad_norm=1.0, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0
    n_batches = len(dataloader)
    
    desc = f"Epoch [Train] {epoch}/{total_epochs}" if epoch else "Train"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for batch in pbar:
        msi, hsi, rec, unc, mask = batch
        
        batch_hsi = hsi.to(device)
        batch_msi = torch.cat([msi, rec, unc], dim=1).to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                z_msi, z_hsi = model(batch_msi, batch_hsi)
                if z_msi.dim() == 4:
                    z_msi = F.adaptive_avg_pool2d(z_msi, (1, 1)).flatten(1)
                if z_hsi.dim() == 4:
                    z_hsi = F.adaptive_avg_pool2d(z_hsi, (1, 1)).flatten(1)
                
                loss = criterion_infonce(z_hsi, z_msi)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            z_msi, z_hsi = model(batch_msi, batch_hsi)
            if z_msi.dim() == 4:
                z_msi = F.adaptive_avg_pool2d(z_msi, (1, 1)).flatten(1)
            if z_hsi.dim() == 4:
                z_hsi = F.adaptive_avg_pool2d(z_hsi, (1, 1)).flatten(1)
                
            loss = criterion_infonce(z_hsi, z_msi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return {"train/loss": running_loss / n_batches}


def validate(model, dataloader, criterion_infonce, device, use_amp=False, epoch=None, total_epochs=None):
    model.eval()
    running_loss = 0.0
    n_batches = len(dataloader)
    
    desc = f"Epoch [Val]   {epoch}/{total_epochs}" if epoch else "Validation"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch in pbar:
            msi, hsi, rec, unc, mask = batch
            batch_hsi = hsi.to(device)
            batch_msi = torch.cat([msi, rec, unc], dim=1).to(device)
            
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    z_msi, z_hsi = model(batch_msi, batch_hsi)
                    if z_msi.dim() == 4: z_msi = F.adaptive_avg_pool2d(z_msi, (1, 1)).flatten(1)
                    if z_hsi.dim() == 4: z_hsi = F.adaptive_avg_pool2d(z_hsi, (1, 1)).flatten(1)
                    loss = criterion_infonce(z_hsi, z_msi)
            else:
                z_msi, z_hsi = model(batch_msi, batch_hsi)
                if z_msi.dim() == 4: z_msi = F.adaptive_avg_pool2d(z_msi, (1, 1)).flatten(1)
                if z_hsi.dim() == 4: z_hsi = F.adaptive_avg_pool2d(z_hsi, (1, 1)).flatten(1)
                loss = criterion_infonce(z_hsi, z_msi)
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return {"val/loss": running_loss / n_batches}


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entraînement Contrastif Siamois Hétérogène")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de config YAML")
    parser.add_argument("--output_tag", type=str, default="", help="Tag additionnel pour le run")
    parser.add_argument("--override_epochs", type=int, default=None, help="Nombre d'epochs à override")
    parser.add_argument("--override_lr", type=float, default=None, help="Learning rate à override")
    args = parser.parse_args()

    # === CHARGEMENT DE LA CONFIG ===
    config = load_config(args.config)
    
    if "data" not in config:
        config["data"] = {}
    config["data"]["task"] = config["data"].get("task", "msi_hsi")

    if args.override_epochs:
        config["training"]["epochs"] = args.override_epochs
    if args.override_lr:
        config["training"]["lr"] = args.override_lr
    
    print(f"\n{'='*60}\nCONFIG CHARGÉE : {args.config}\n{'='*60}\n")

    # === INITIALISATION ===
    seed_val = config.get("experiment", {}).get("seed", 42)
    set_seed(seed_val)  
    device = get_device()
    
    use_amp = config["training"].get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    print(f"⚡ Précision mixte (AMP) : {'ACTIVÉE' if use_amp else 'DÉSACTIVÉE'}")

    # === FILTRAGE DES BANDES HSI ===
    kept_indices = get_kept_wavelength_indices(WVL_PRS, config)
    print(f"Bandes HSI conservées : {kept_indices.sum()} / {len(WVL_PRS)}")
    
    # === DATA LOADERS (Via le Dataset Universel Brut) ===
    print("📦 Création des dataloaders via UniversalRawMSIHSIDataset...")
    patch_size = config["data"].get("patch_size", 64)
    threshold = config["data"].get("threshold", 0.5)
    
    dataset_kwargs = {
        "kept_indices": kept_indices,
        "patch_size": patch_size,
        "two_zone": False
    }

    train_dataset = UniversalRawMSIHSIDataset(data_dir=config["data"]["data_dir_train"], is_train=True, **dataset_kwargs)
    val_dataset = UniversalRawMSIHSIDataset(data_dir=config["data"]["data_dir_val"], is_train=False, **dataset_kwargs)
    test_dataset = UniversalRawMSIHSIDataset(data_dir=config["data"]["data_dir_test"], is_train=False, **dataset_kwargs)

    loader_kwargs = {
        "batch_size": config["data"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
        "pin_memory": True,
        "persistent_workers": config["data"]["num_workers"] > 0,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    # === CAPTURE D'UN BATCH FIXE DE VALIDATION ===
    print("📸 Capture du batch fixe de validation...")
    fixed_batch = next(iter(val_loader))
    fixed_msi, fixed_hsi, fixed_rec, fixed_unc, fixed_mask = fixed_batch
    
    # Reconstruction du batch msi combiné pour la visualisation et initialisation du modèle
    fixed_val_msi = torch.cat([fixed_msi, fixed_rec, fixed_unc], dim=1)
    fixed_val_hsi = fixed_hsi

    n_msi = fixed_val_msi.shape[1]
    n_hsi = fixed_val_hsi.shape[1]
    print(f"📐 Formats effectifs -> msi (avec rec+unc): {fixed_val_msi.shape} | HSI (filtré): {fixed_val_hsi.shape}")

    # === MODÈLE ===
    print("🧠 Instanciation du modèle...")
    model = build_model(config, n_msi, n_hsi).to(device)
    
    if config["model"].get("load_model"):
        checkpoint_path = Path(config["model"]["load_model"])
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ Modèle chargé depuis {checkpoint_path}")

    # === CRITÈRE DE PERTE & OPTIMISEUR ===
    from src.metrics_and_loss.loss import InfoNce
    temperature = config["training"].get("temperature", 0.07)
    criterion_infonce = InfoNce(temperature=temperature)

    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_lr_scheduler(optimizer, config)

    # === INITIALISATION W&B ===
    run_name = f"{config['model']['name']}_lr{config['training']['lr']}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.output_tag:
        run_name = f"{run_name}_{args.output_tag}"

    run_name_wb = init_wandb(config, run_name=run_name)

    # === BOUCLE D'ENTRAÎNEMENT ===
    best_val_loss = float("inf")
    total_epochs = config["training"]["epochs"]
    early_stopper = EarlyStopping(
        patience=config["training"].get("early_stopping_patience", 15),
        min_delta=config["training"].get("early_stopping_min_delta", 1e-4),
        start_epoch=config["training"].get("early_stopping_start_epoch", 10)
    )

    for epoch in range(total_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion_infonce, device, 
            scaler=scaler, max_grad_norm=config["training"].get("max_grad_norm", 1.0),
            epoch=epoch + 1, total_epochs=total_epochs
        )
        
        val_metrics = validate(
            model, val_loader, criterion_infonce, device, 
            use_amp=use_amp, epoch=epoch + 1, total_epochs=total_epochs
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["val/loss"])
        else:
            scheduler.step()

        print(f"\n[Epoch {epoch+1}/{total_epochs}] — LR: {current_lr:.2e} | Train Loss: {train_metrics['train/loss']:.4f} | Val Loss: {val_metrics['val/loss']:.4f}")

        wandb.log({"epoch": epoch + 1, "lr": current_lr, **train_metrics, **val_metrics})
        
        # Log visuel périodique
        log_interval = config["training"].get("similarity_log_interval", 5)
        if (epoch + 1) % log_interval == 0:
            log_change_detection_map_to_wandb_with_diff(
                model, fixed_val_hsi, fixed_val_msi, batch_mask=fixed_mask, kept_indices=kept_indices,
                device=device, tag_suffix=f"epoch_{epoch + 1}"
            )
        
        # Sauvegarde du meilleur modèle
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            save_checkpoint(model, config, run_name_wb, is_best=True)
            print(f"    ✨ Nouveau meilleur modèle sauvegardé !")

        early_stopper(val_metrics["val/loss"], epoch=epoch + 1)
        if early_stopper.early_stop:
            print(f"\n⏹ Arrêt précoce déclenché à l'époque {epoch+1} !")
            break

    wandb.finish()


if __name__ == "__main__":
    main()