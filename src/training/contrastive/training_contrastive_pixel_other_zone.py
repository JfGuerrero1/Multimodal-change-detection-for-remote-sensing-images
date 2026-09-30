import sys
from pathlib import Path
import inspect
import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt

# Ajoute la racine du projet au sys.path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.constants import WVL_PRS
from src.models.contrastive import SiameseNetwork_heterogene
from src.prepare_data.dataset import UniversalRawMSIHSIDataset
from src.metrics_and_loss.loss import PixelWiseOtherZoneInfoNCE
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
    target_blue = 490.0
    target_green = 560.0
    target_red = 665.0
    
    s2_red_idx = np.argmin(np.abs(wvl_s2 - target_red))
    s2_green_idx = np.argmin(np.abs(wvl_s2 - target_green))
    s2_blue_idx = np.argmin(np.abs(wvl_s2 - target_blue))
    s2_rgb_indices = [s2_red_idx, s2_green_idx, s2_blue_idx]
    
    prs_red_idx = np.argmin(np.abs(wvl_prs - target_red))
    prs_green_idx = np.argmin(np.abs(wvl_prs - target_green))
    prs_blue_idx = np.argmin(np.abs(wvl_prs - target_blue))
    prs_rgb_indices = [prs_red_idx, prs_green_idx, prs_blue_idx]
    
    return s2_rgb_indices, prs_rgb_indices


def extract_batch_data(batch):
    """
    Extrait précisément (msi1, hsi1, msi2, hsi2, mask1) à partir du tuple à 10 éléments
    retourné par UniversalRawMSIHSIDataset lorsque two_zone=True.
    """
    if len(batch) == 10:
        # Tuple: (msi1, hsi1, rec1, unc1, mask1, msi2, hsi2, rec2, unc2, mask2)
        return batch[0], batch[1], batch[5], batch[6], batch[4]
    elif len(batch) == 5:
        # Tuple: (msi1, hsi1, rec1, unc1, mask1)
        return batch[0], batch[1], None, None, batch[4]
    elif len(batch) == 4:
        return batch[0], batch[1], None, None, None
    else:
        raise ValueError(f"Taille de batch inattendue : {len(batch)}")


def compute_loss(criterion, z1, z2, z1_other=None, z2_other=None, mask=None):
    """
    Convertit le masque trinaire en masques booléens adaptés :
      -  1 : Positif (Arapparier)
      - -1 : Négatif (Repousser)
      -  0 : Neutre (Ignorer totalement via valid_mask)
    """
    if isinstance(criterion, PixelWiseOtherZoneInfoNCE):
        if mask is not None:
            positive_mask = (mask == 1)
            negative_mask = (mask == -1)
            valid_mask = (mask != 0)
        else:
            positive_mask = torch.ones(z1.shape[0], z1.shape[2], z1.shape[3], dtype=torch.bool, device=z1.device)
            negative_mask = torch.zeros_like(positive_mask)
            valid_mask = torch.ones_like(positive_mask)

        if z1_other is None:
            z1_other = z1
        if z2_other is None:
            z2_other = z2

        sig = inspect.signature(criterion.forward)
        kwargs = {
            "z1": z1,
            "z2": z2,
            "z1_other": z1_other,
            "z2_other": z2_other,
            "positive_mask": positive_mask
        }
        
        if "negative_mask" in sig.parameters:
            kwargs["negative_mask"] = negative_mask
        if "valid_mask" in sig.parameters:
            kwargs["valid_mask"] = valid_mask

        return criterion(**kwargs)
    
    # Fallback si loss globale
    if z1.dim() == 4:
        z1 = F.adaptive_avg_pool2d(z1, (1, 1)).flatten(1)
    if z2.dim() == 4:
        z2 = F.adaptive_avg_pool2d(z2, (1, 1)).flatten(1)
    return criterion(z1, z2)


def log_change_detection_map_to_wandb(
    model,
    batch_hsi,
    batch_msi,
    batch_mask=None,
    kept_indices=None,
    device="cuda",
    tag_suffix="epoch"
):
    """Génère et loggue sur WandB les cartes de similarité contrastive."""
    model.eval()
    
    with torch.no_grad():
        batch_msi = batch_msi.to(device)
        batch_hsi = batch_hsi.to(device)
        
        out = model(batch_msi, batch_hsi)
        z_msi, z_hsi = out[0], out[1]
        
        if z_msi.dim() == 4 and z_hsi.dim() == 4:
            z_msi_norm = F.normalize(z_msi, p=2, dim=1)
            z_hsi_norm = F.normalize(z_hsi, p=2, dim=1)
            scores = torch.norm(z_msi_norm - z_hsi_norm, p=2, dim=1, keepdim=True)
        else:
            B, H, W = batch_hsi.shape[0], batch_hsi.shape[2], batch_hsi.shape[3]
            scores = torch.zeros(B, 1, H, W, device=device)
        
        scores_np = scores.cpu().numpy()
        msi_np = batch_msi.cpu().numpy()
        hsi_np = batch_hsi.cpu().numpy()
        mask_np = batch_mask.cpu().numpy() if batch_mask is not None else None
        
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
        
        batch_size = msi_np.shape[0]
        n_to_plot = min(batch_size, 4)
        
        fig, axes = plt.subplots(n_to_plot, 4, figsize=(20, 4 * n_to_plot))
        if n_to_plot == 1:
            axes = np.expand_dims(axes, 0)
        
        for i in range(n_to_plot):
            # 1. MSI RGB
            try:
                rgb_msi = np.transpose(msi_np[i, s2_rgb_idx], (1, 2, 0))
                rgb_msi = np.clip((rgb_msi - rgb_msi.min()) / (rgb_msi.max() - rgb_msi.min() + 1e-8), 0, 1)
            except:
                rgb_msi = msi_np[i, 0]
            axes[i, 0].imshow(rgb_msi)
            axes[i, 0].set_title("MSI RGB (S2)")
            axes[i, 0].axis('off')
            
            # 2. HSI RGB
            try:
                rgb_hsi = np.transpose(hsi_np[i, prs_rgb_idx], (1, 2, 0))
                rgb_hsi = np.clip((rgb_hsi - rgb_hsi.min()) / (rgb_hsi.max() - rgb_hsi.min() + 1e-8), 0, 1)
            except:
                rgb_hsi = hsi_np[i, 0]
            axes[i, 1].imshow(rgb_hsi)
            axes[i, 1].set_title("HSI RGB")
            axes[i, 1].axis('off')

            # 3. Score de distance
            pred_map = scores_np[i, 0]
            im_score = axes[i, 2].imshow(pred_map, cmap='jet')
            axes[i, 2].set_title("Score Contrastif")
            axes[i, 2].axis('off')
            fig.colorbar(im_score, ax=axes[i, 2], fraction=0.046, pad=0.04)

            # 4. Masque (GT: -1, 0, 1)
            if mask_np is not None:
                im_mask = axes[i, 3].imshow(mask_np[i], cmap='coolwarm', vmin=-1, vmax=1)
                cbar_mask = fig.colorbar(im_mask, ax=axes[i, 3], fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
                cbar_mask.set_ticklabels(['-1 (Neg)', '0 (Neutre)', '1 (Pos)'])
                axes[i, 3].set_title("Masque (GT)")
            else:
                axes[i, 3].set_title("Masque (N/A)")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        wandb.log({f"change_detection_maps/{tag_suffix}": wandb.Image(fig)})
        plt.close(fig)


# ============================================================================
# BOUCLES DE TRAIN & VAL
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion_infonce, device, scaler=None, max_grad_norm=1.0, epoch=None, total_epochs=None):
    model.train()
    running_loss = 0.0
    n_batches = len(dataloader)
    
    desc = f"Epoch [Train] {epoch}/{total_epochs}" if epoch else "Train"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for batch in pbar:
        msi1, hsi1, msi2, hsi2, mask = extract_batch_data(batch)
        
        batch_msi = msi1.to(device)
        batch_hsi = hsi1.to(device)
        batch_mask = mask.to(device) if mask is not None else None
        
        batch_msi2 = msi2.to(device) if msi2 is not None else None
        batch_hsi2 = hsi2.to(device) if hsi2 is not None else None

        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                z_msi, z_hsi = model(batch_msi, batch_hsi)
                if batch_msi2 is not None and batch_hsi2 is not None:
                    z_msi_other, z_hsi_other = model(batch_msi2, batch_hsi2)
                else:
                    z_msi_other, z_hsi_other = z_msi, z_hsi

                loss = compute_loss(
                    criterion_infonce, 
                    z1=z_hsi, z2=z_msi, 
                    z1_other=z_hsi_other, z2_other=z_msi_other, 
                    mask=batch_mask
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            z_msi, z_hsi = model(batch_msi, batch_hsi)
            if batch_msi2 is not None and batch_hsi2 is not None:
                z_msi_other, z_hsi_other = model(batch_msi2, batch_hsi2)
            else:
                z_msi_other, z_hsi_other = z_msi, z_hsi

            loss = compute_loss(
                criterion_infonce, 
                z1=z_hsi, z2=z_msi, 
                z1_other=z_hsi_other, z2_other=z_msi_other, 
                mask=batch_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    return {"train/loss": running_loss / n_batches}


def validate(model, dataloader, criterion_infonce, device, use_amp=False, epoch=None, total_epochs=None, prefix="val"):
    model.eval()
    running_loss = 0.0
    n_batches = len(dataloader)
    
    desc = f"Epoch [{prefix.capitalize()}]   {epoch}/{total_epochs}" if epoch else f"{prefix.capitalize()}"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for batch in pbar:
            msi1, hsi1, msi2, hsi2, mask = extract_batch_data(batch)
            
            batch_msi = msi1.to(device)
            batch_hsi = hsi1.to(device)
            batch_mask = mask.to(device) if mask is not None else None
            
            batch_msi2 = msi2.to(device) if msi2 is not None else None
            batch_hsi2 = hsi2.to(device) if hsi2 is not None else None
            
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    z_msi, z_hsi = model(batch_msi, batch_hsi)
                    if batch_msi2 is not None and batch_hsi2 is not None:
                        z_msi_other, z_hsi_other = model(batch_msi2, batch_hsi2)
                    else:
                        z_msi_other, z_hsi_other = z_msi, z_hsi

                    loss = compute_loss(
                        criterion_infonce, 
                        z1=z_hsi, z2=z_msi, 
                        z1_other=z_hsi_other, z2_other=z_msi_other, 
                        mask=batch_mask
                    )
            else:
                z_msi, z_hsi = model(batch_msi, batch_hsi)
                if batch_msi2 is not None and batch_hsi2 is not None:
                    z_msi_other, z_hsi_other = model(batch_msi2, batch_hsi2)
                else:
                    z_msi_other, z_hsi_other = z_msi, z_hsi

                loss = compute_loss(
                    criterion_infonce, 
                    z1=z_hsi, z2=z_msi, 
                    z1_other=z_hsi_other, z2_other=z_msi_other, 
                    mask=batch_mask
                )
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    return {f"{prefix}/loss": running_loss / n_batches}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entraînement Contrastif Siamois Hétérogène (Pur MSI/HSI)")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de config YAML")
    parser.add_argument("--output_tag", type=str, default="", help="Tag additionnel pour le run")
    parser.add_argument("--override_epochs", type=int, default=None, help="Nombre d'epochs à override")
    parser.add_argument("--override_lr", type=float, default=None, help="Learning rate à override")
    args = parser.parse_args()

    config = load_config(args.config)
    if "data" not in config:
        config["data"] = {}
    config["data"]["task"] = config["data"].get("task", "msi_hsi")

    if args.override_epochs:
        config["training"]["epochs"] = args.override_epochs
    if args.override_lr:
        config["training"]["lr"] = args.override_lr
    
    seed_val = config.get("experiment", {}).get("seed", 42)
    set_seed(seed_val)  
    device = get_device()
    
    use_amp = config["training"].get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None


    
    # --- AJOUT : Récupération des paramètres de seuillage depuis la config ---
    data_cfg = config.get("data", {})
    
    kept_indices = get_kept_wavelength_indices(WVL_PRS, config)
    patch_size = data_cfg.get("patch_size", 64)
    two_zone_flag = data_cfg.get("two_zone", True)
    
    dataset_kwargs = {
        "kept_indices": kept_indices,
        "patch_size": patch_size,
        "return_mask": True,
        "two_zone": two_zone_flag,
        "threshold_mode": data_cfg.get("threshold_mode", "empirical"),
        "tau_low_pct": data_cfg.get("tau_low_pct", 50.0),
        "tau_high_pct": data_cfg.get("tau_high_pct", 98.0),
        "gamma_low_pct": data_cfg.get("gamma_low_pct", 50.0),
        "sample_size_for_percentiles": data_cfg.get("sample_size_for_percentiles", 100),
    }

    train_dataset = UniversalRawMSIHSIDataset(data_dir=data_cfg["data_dir_train"], is_train=True, **dataset_kwargs)
    val_dataset = UniversalRawMSIHSIDataset(data_dir=data_cfg["data_dir_val"], is_train=False, **dataset_kwargs)
    test_dataset = UniversalRawMSIHSIDataset(data_dir=data_cfg["data_dir_test"], is_train=False, **dataset_kwargs)

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

    # Extraction des 5 premiers batches pour visualisations WandB
    num_log_batches = 5
    val_iter = iter(val_loader)
    fixed_val_batches = [next(val_iter) for _ in range(min(num_log_batches, len(val_loader)))]

    test_iter = iter(test_loader)
    fixed_test_batches = [next(test_iter) for _ in range(min(num_log_batches, len(test_loader)))]

    sample_msi, sample_hsi, _, _, _ = extract_batch_data(fixed_val_batches[0])
    n_msi = sample_msi.shape[1]
    n_hsi = sample_hsi.shape[1]
    print(f"📐 Formats d'entrée -> MSI: {sample_msi.shape} | HSI: {sample_hsi.shape} | Mode two_zone: {two_zone_flag}")

    model = build_model(config, n_msi, n_hsi).to(device)

    temperature = config["training"].get("temperature", 0.07)
    criterion_infonce = PixelWiseOtherZoneInfoNCE(temperature=temperature)

    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_lr_scheduler(optimizer, config)

    run_name = f"{config['model']['name']}_pure_lr{config['training']['lr']}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.output_tag:
        run_name = f"{run_name}_{args.output_tag}"
    run_name_wb = init_wandb(config, run_name=run_name)

    best_val_loss = float("inf")
    best_checkpoint_path = None
    total_epochs = config["training"]["epochs"]
    early_stopper = EarlyStopping(
        patience=config["training"].get("early_stopping_patience", 15),
        min_delta=config["training"].get("early_stopping_min_delta", 1e-4),
        start_epoch=config["training"].get("early_stopping_start_epoch", 10)
    )

    # ============================================================================
    # BOUCLE D'ENTRAÎNEMENT
    # ============================================================================
    for epoch in range(total_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion_infonce, device, 
            scaler=scaler, max_grad_norm=config["training"].get("max_grad_norm", 1.0),
            epoch=epoch + 1, total_epochs=total_epochs
        )
        
        val_metrics = validate(
            model, val_loader, criterion_infonce, device, 
            use_amp=use_amp, epoch=epoch + 1, total_epochs=total_epochs, prefix="val"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["val/loss"])
        else:
            scheduler.step()

        print(f"\n[Epoch {epoch+1}/{total_epochs}] — LR: {current_lr:.2e} | Train Loss: {train_metrics['train/loss']:.4f} | Val Loss: {val_metrics['val/loss']:.4f}")

        wandb.log({"epoch": epoch + 1, "lr": current_lr, **train_metrics, **val_metrics})
        
        # Logs visualisations WandB
        log_interval = config["training"].get("similarity_log_interval", 5)
        if (epoch + 1) % log_interval == 0:
            for b_idx, b_data in enumerate(fixed_val_batches):
                b_msi, b_hsi, _, _, b_mask = extract_batch_data(b_data)
                log_change_detection_map_to_wandb(
                    model, b_hsi, b_msi, batch_mask=b_mask, kept_indices=kept_indices,
                    device=device, tag_suffix=f"val_batch_{b_idx}_epoch_{epoch + 1}"
                )
        
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            best_checkpoint_path = save_checkpoint(model, config, run_name_wb, is_best=True)
            print(f"    ✨ Nouveau meilleur modèle sauvegardé !")

        early_stopper(val_metrics["val/loss"], epoch=epoch + 1)
        if early_stopper.early_stop:
            print(f"\n⏹ Arrêt précoce déclenché à l'époque {epoch+1} !")
            break

    # ============================================================================
    # PHASE DE TEST FINALE
    # ============================================================================
    print("\n" + "="*50)
    print("🧪 DÉBUT DE LA PHASE DE TEST FINALE")
    print("="*50)

    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        print(f"📥 Chargement du meilleur checkpoint : {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(
        model, test_loader, criterion_infonce, device, 
        use_amp=use_amp, epoch=None, total_epochs=None, prefix="test"
    )
    print(f"📊 Métriques finales de Test -> Loss: {test_metrics['test/loss']:.4f}")
    wandb.log(test_metrics)

    print("🖼️ Génération des cartes de détection de changement pour les 5 batches de test...")
    for b_idx, b_data in enumerate(fixed_test_batches):
        b_msi, b_hsi, _, _, b_mask = extract_batch_data(b_data)
        log_change_detection_map_to_wandb(
            model, b_hsi, b_msi, batch_mask=b_mask, kept_indices=kept_indices,
            device=device, tag_suffix=f"test_batch_{b_idx}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()