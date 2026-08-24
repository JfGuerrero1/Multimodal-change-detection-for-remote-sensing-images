import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import argparse
import random
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import wandb
import numpy as np
from torchmetrics.functional.image import (
    structural_similarity_index_measure as ssim_fn,
)

from src.models import GradualExpansionUNet_residual
from src.old.metrics_and_loss.loss import SpectralLoss
from src.old.metrics_and_loss.metrics import compute_ssim_multiband, compute_ergas
from src.prepare_data.prepare_patch import create_data_loaders_spectral
from src.visualise.visualisation import visualise_synthesis
from src.visualise.visualisation_spectre import trace_spectre
from src.constants import WVL_PRS
from src.prepare_data.dataset import SpectralDataset
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

def build_model(config: dict, n_msi: int, n_hsi: int) -> torch.nn.Module:
  """Instancie le modèle selon la configuration YAML."""
  model_cfg = config["model"]
  name = model_cfg.get("name", "").lower()

  if name in ["gradualexpansionunet_res"]:
    model = GradualExpansionUNet_residual(
        in_msi=n_msi,
        in_hsi=n_hsi,
        interpolation_mode=model_cfg.get("interpolation_mode", "Bilinear"),
        base_channel=model_cfg.get("base_channel", 64),
        activation=model_cfg.get("activation", "silu"),
        with_batch_norm=model_cfg.get("with_batch_norm", True),
        with_mlp_spectral=model_cfg.get("with_mlp_spectral", False),
        final_activation=model_cfg.get("final_activation", None),
    )
  else:
    raise ValueError(f"❌ Modèle non reconnu : '{name}'")

  return model


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    scaler: torch.amp.GradScaler = None,
) -> dict:
  """Entraîne le modèle sur une époque avec support AMP / Mixed Precision."""
  model.train()

  total_loss = 0.0
  total_mse = 0.0
  total_sam = 0.0
  total_mae = 0.0
  total_grad_norm = 0.0
  valid_grad_steps = 0

  use_amp = scaler is not None and device.type == "cuda"
  pbar = tqdm(loader, desc="Training")

  for x_init, x_interp, y, patch_ids in pbar:
    x_init = x_init.to(device, non_blocking=True)
    x_interp = x_interp.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    optimizer.zero_grad()

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
      pred = model(x_init, x_interp)
      loss, mse, mae, sam = criterion(pred, y)

    if use_amp:
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      grad_norm = torch.nn.utils.clip_grad_norm_(
          model.parameters(), max_norm=1.0
      )
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      grad_norm = torch.nn.utils.clip_grad_norm_(
          model.parameters(), max_norm=1.0
      )
      optimizer.step()

    gn_val = grad_norm.item()
    if not (np.isnan(gn_val) or np.isinf(gn_val)):
      total_grad_norm += gn_val
      valid_grad_steps += 1

    total_loss += loss.item()
    total_mse += mse.item()
    total_sam += sam.item()
    total_mae += mae.item()

    pbar.set_postfix(
        Loss=f"{loss.item():.4f}",
        MAE=f"{mae.item():.4f}",
        SAM_deg=f"{np.degrees(sam.item()):.2f}°",
        GradNorm=f"{gn_val:.2f}",
    )

  n = len(loader)
  metrics = {
      "train_loss": total_loss / n,
      "train_mse": total_mse / n,
      "train_mae": total_mae / n,
      "train_sam": total_sam / n,
      "train_sam_deg": np.degrees(total_sam / n),
      "train_grad_norm": total_grad_norm / max(1, valid_grad_steps),
  }

  return metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    use_amp: bool = False,
    compute_full_metrics: bool = True,
) -> dict:
  """Valide le modèle avec support AMP, PSNR, RMSE et métriques de synthèse."""
  model.eval()

  total_loss, total_mse, total_sam, total_mae = 0.0, 0.0, 0.0, 0.0
  total_ssim, total_ergas = 0.0, 0.0

  pbar = tqdm(loader, desc="Validation")
  for x_init, x_interp, y, patch_ids in pbar:
    x_init = x_init.to(device, non_blocking=True)
    x_interp = x_interp.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
      pred = model(x_init, x_interp)
      loss, mse, mae, sam = criterion(pred, y)

    total_loss += loss.item()
    total_mse += mse.item()
    total_sam += sam.item()
    total_mae += mae.item()

    batch_rmse = np.sqrt(mse.item())
    batch_psnr = (
        20 * np.log10(1.0 / batch_rmse) if batch_rmse > 1e-8 else 100.0
    )

    postfix_dict = {
        "Loss": f"{loss.item():.4f}",
        "PSNR": f"{batch_psnr:.2f}dB",
        "SAM_deg": f"{np.degrees(sam.item()):.2f}°",
    }

    if compute_full_metrics:
      pred_safe = torch.clamp(pred, 1e-6, 1.0)
      y_safe = torch.clamp(y, 1e-6, 1.0)

      batch_ssim = ssim_fn(pred_safe, y_safe, data_range=1.0)
      batch_ergas = compute_ergas(
          pred_safe.cpu().numpy(), y_safe.cpu().numpy()
      )

      total_ssim += batch_ssim.item()
      total_ergas += batch_ergas

      postfix_dict["SSIM"] = f"{batch_ssim.item():.3f}"
      postfix_dict["ERGAS"] = f"{batch_ergas:.1f}"

    pbar.set_postfix(**postfix_dict)

  n = len(loader)
  mean_mse = total_mse / n
  mean_rmse = np.sqrt(mean_mse)
  mean_psnr = 20 * np.log10(1.0 / mean_rmse) if mean_rmse > 1e-8 else 99.0

  metrics = {
      "val_loss": total_loss / n,
      "val_mse": mean_mse,
      "val_rmse": mean_rmse,
      "val_psnr": mean_psnr,
      "val_mae": total_mae / n,
      "val_sam": total_sam / n,
      "val_sam_deg": np.degrees(total_sam / n),
  }

  if compute_full_metrics:
    metrics["val_ssim"] = total_ssim / n
    metrics["val_ergas"] = total_ergas / n

  return metrics


@torch.no_grad()
def test(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
  """Évalue le modèle sur le test set avec la suite complète des métriques physiques."""
  model.eval()

  total_loss, total_mse, total_sam, total_mae = 0.0, 0.0, 0.0, 0.0
  total_ssim, total_ergas = 0.0, 0.0

  pbar = tqdm(loader, desc="Testing")
  for x_init, x_interp, y, patch_ids in pbar:
    x_init = x_init.to(device, non_blocking=True)
    x_interp = x_interp.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
      pred = model(x_init, x_interp)
      loss, mse, mae, sam = criterion(pred, y)

    pred_safe = torch.clamp(pred, 1e-6, 1.0)
    y_safe = torch.clamp(y, 1e-6, 1.0)

    batch_ssim = ssim_fn(pred_safe, y_safe, data_range=1.0)
    batch_ergas = compute_ergas(pred_safe.cpu().numpy(), y_safe.cpu().numpy())

    total_loss += loss.item()
    total_mse += mse.item()
    total_sam += sam.item()
    total_mae += mae.item()
    total_ssim += batch_ssim.item()
    total_ergas += batch_ergas

    batch_rmse = np.sqrt(mse.item())
    batch_psnr = 20 * np.log10(1.0 / batch_rmse) if batch_rmse > 1e-8 else 99.0

    pbar.set_postfix(
        Loss=f"{loss.item():.4f}",
        PSNR=f"{batch_psnr:.2f}dB",
        SAM_deg=f"{np.degrees(sam.item()):.2f}°",
        SSIM=f"{batch_ssim.item():.4f}",
        ERGAS=f"{batch_ergas:.2f}",
    )

  n = len(loader)
  mean_mse = total_mse / n
  mean_rmse = np.sqrt(mean_mse)
  mean_psnr = 20 * np.log10(1.0 / mean_rmse) if mean_rmse > 1e-8 else 100.0

  return {
      "loss": total_loss / n,
      "mse": mean_mse,
      "rmse": mean_rmse,
      "psnr": mean_psnr,
      "mae": total_mae / n,
      "sam": total_sam / n,
      "sam_deg": np.degrees(total_sam / n),
      "ssim": total_ssim / n,
      "ergas": total_ergas / n,
  }


@torch.no_grad()
@torch.no_grad()
def evaluate_and_log_random_patches(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    wvl_full: np.ndarray,
    kept_indices: np.ndarray = None,
    use_amp: bool = False,
    prefix: str = "Val Random",
    n_samples: int = 3,
):
  """Évalue le modèle, sélectionne N patchs de façon 100% déterministe par scène,
  et envoie les planches et les spectres de 4 pixels reproductibles sur WandB.
  """
  model.eval()
  
  # --- CALCUL SÉCURISÉ DE WVL_KEPT ---
  if kept_indices is not None and wvl_full is not None:
      kept_indices_arr = np.asarray(kept_indices)
      if kept_indices_arr.dtype == bool:
          wvl_kept = wvl_full[kept_indices_arr]
      else:
          wvl_kept = wvl_full[kept_indices_arr]
  else:
      wvl_kept = wvl_full

  scenes_dict = {}

  print(f"\n🔍 [1/2] Évaluation du jeu de données ({prefix}) et stockage des prédictions...")

  for x_init, x_interp, y, patch_ids in tqdm(loader, desc=f"Inférence {prefix}"):
    x_init = x_init.to(device, non_blocking=True)
    x_interp = x_interp.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
      pred = model(x_init, x_interp)

    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    x_init_np = x_init.detach().cpu().numpy()

    batch_size = y.size(0)

    for i in range(batch_size):
      p_id = patch_ids[i] if not hasattr(patch_ids[i], 'item') else patch_ids[i].item()

      if isinstance(p_id, int) or str(p_id).isdigit():
        p_id_int = int(p_id)
        scene_id = (p_id_int // 36) + 1
        local_patch_idx = p_id_int % 36
      else:
        parts = str(p_id).split("_")
        scene_id = parts[1] if len(parts) > 1 else "unknown"
        local_patch_idx = p_id

      gt_hwc = np.moveaxis(y_np[i], 0, -1) if y_np[i].shape[0] < y_np[i].shape[-1] else y_np[i]
      pred_hwc = np.moveaxis(pred_np[i], 0, -1) if pred_np[i].shape[0] < pred_np[i].shape[-1] else pred_np[i]
      msi_hwc = np.moveaxis(x_init_np[i], 0, -1) if x_init_np[i].shape[0] < x_init_np[i].shape[-1] else x_init_np[i]

      print(msi_hwc.shape)

      mae = float(np.mean(np.abs(gt_hwc - pred_hwc)))
      dot = np.sum(pred_hwc * gt_hwc, axis=-1)
      norm_p = np.linalg.norm(pred_hwc, axis=-1)
      norm_g = np.linalg.norm(gt_hwc, axis=-1)
      sam_map = np.arccos(np.clip(dot / (norm_p * norm_g + 1e-8), -1.0, 1.0))
      sam_rad = float(np.mean(sam_map))

      patch_data = {
          "patch_id": local_patch_idx,
          "sam_rad": sam_rad,
          "sam_deg": float(np.degrees(sam_rad)),
          "mae": mae,
          "cube_gt": gt_hwc,
          "cube_predict": pred_hwc,
          "cube_msi": msi_hwc,
      }

      if scene_id not in scenes_dict:
        scenes_dict[scene_id] = []
      scenes_dict[scene_id].append(patch_data)

  print(f"\n🎲 [2/2] Sélection déterministe et envoi des planches ({prefix}) sur WandB...")

  # Tri des scènes pour un parcours toujours identique
  for scene_id in sorted(scenes_dict.keys(), key=lambda x: str(x)):
    patches = scenes_dict[scene_id]
    
    # Tri des patchs par leur ID pour figer l'ordre de la liste
    patches = sorted(patches, key=lambda p: str(p["patch_id"]))
    
    # Fixer la graine de Python spécifiquement pour cette scène (garantit le même random.sample)
    scene_seed = abs(hash(str(scene_id))) % (2**32)
    random.seed(scene_seed)

    k = min(n_samples, len(patches))
    random_patches = random.sample(patches, k)

    selected = [(f"Random_{idx+1}", p_data) for idx, p_data in enumerate(random_patches)]

    for label, p_data in selected:
      p_id = p_data["patch_id"]
      save_name = f"{prefix}_Scene_{scene_id}_{label}_Patch_{p_id}_SAM_{p_data['sam_deg']:.2f}_deg.png"

      data_to_plot = {
          "cube_gt": p_data["cube_gt"],
          "cube_predict": p_data["cube_predict"],
          "cube_msi": p_data["cube_msi"],
          "model name": f"{prefix} — Scène {scene_id} ({label} Patch {p_id})",
          "img_mae": p_data["mae"],
          "img_sam": p_data["sam_rad"],
      }

      visualise_synthesis(
          data=data_to_plot,
          save_name=save_name,
          plot_dir=None,
          kept_indices=kept_indices,
          log_to_wandb=True,
      )

      # Génération reproductible de 4 pixels par patch (basée sur le hash de p_id)
      seed_val = abs(hash(str(p_id))) % (2**32) if not str(p_id).isdigit() else 42 + int(str(p_id))
      rng = np.random.default_rng(seed_val)
      
      H, W, _ = p_data["cube_predict"].shape
      num_pixels = 4
      y_indices = rng.integers(0, H, size=num_pixels)
      x_indices = rng.integers(0, W, size=num_pixels)

      # Appel de trace_spectre (3 lignes : RGB, Spectres séparés, Différence & Erreur moyennée ± std)
      trace_spectre(
          patch=p_data["cube_predict"],
          patch_gt=p_data["cube_gt"],
          y_indices=y_indices,
          x_indices=x_indices,
          scene=f"Scene_{scene_id}_{label}_Patch_{p_id}",
          output_dir_diag=None,
          wvl_prs=wvl_kept,  
          log_wandb=True,
          wandb_name=f"Suivi_Pixels/{prefix}_Scene_{scene_id}_{label}_Patch_{p_id}"
      )

def main():
  """Fonction principale d'entraînement."""
  parser = argparse.ArgumentParser(
      description="Entraîne le modèle résiduel MSI→HSI depuis une config YAML"
  )
  parser.add_argument(
      "--config",
      type=str,
      default="src/config/config_default_unet_res.yaml",
      help="Chemin vers le fichier de config YAML"
  )
  parser.add_argument(
      "--override_epochs",
      type=int,
      default=None,
      help="Override le nombre d'epochs"
  )
  parser.add_argument(
      "--override_lr",
      type=float,
      default=None,
      help="Override le learning rate"
  )
  parser.add_argument(
      "--output_tag",
      type=str,
      default="",
      help="Tag additionnel pour le nom du run"
  )
  
  args = parser.parse_args()

  # === CHARGEMENT DE LA CONFIG ===
  config = load_config(args.config)
  
  if args.override_epochs:
      config["training"]["epochs"] = args.override_epochs
  if args.override_lr:
      config["training"]["lr"] = args.override_lr
  
  print(f"\n{'='*60}")
  print(f"CONFIG CHARGÉE : {args.config}")
  print(f"{'='*60}\n")

  # === INITIALISATION ===
  seed_val = config.get("experiment", {}).get("seed", 42)
  set_seed(seed_val)  
  device = get_device()
  
  use_amp = config["training"].get("use_amp", True) and device.type == "cuda"
  scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None
  print(f"⚡ Précision mixte (AMP) : {'ACTIVÉE' if use_amp else 'DÉSACTIVÉE'}")

  kept_indices = get_kept_wavelength_indices(WVL_PRS, config)

  print(f"Bandes conservées : {kept_indices.sum()} / {len(WVL_PRS)} "
        f"({len(WVL_PRS) - kept_indices.sum()} bandes filtrées)")
  
  # Data loaders
  print("📦 Création des dataloaders...")
  train_loader, val_loader, test_loader = create_data_loaders_spectral(
      train_dir=config["data"]["data_dir_train"],
      val_dir=config["data"]["data_dir_val"],
      test_dir=config["data"]["data_dir_test"],
      simulated=config["data"]["simulated"],
      augment=config["data"]["augment"], 
      augment_illumination=config["data"]["augment_illumination"],
      batch_size=config["data"]["batch_size"],
      num_workers=config["data"]["num_workers"],
      is_residual=config["data"]["is_residual"],
      is_normalised=config["data"]["is_normalised"],
      kept_indices=kept_indices  
  )

  # Vérification des dimensions
  sample_x, sample_x_interp, sample_y, _ = next(iter(train_loader))
  n_msi = sample_x.shape[1]
  n_hsi = sample_y.shape[1]
  print(f"📐 MSI shape: {sample_x.shape} | HSI Interp shape: {sample_x_interp.shape} | HSI shape: {sample_y.shape}")

  # Modèle
  print("🧠 Instanciation du modèle...")
  model = build_model(config, n_msi, n_hsi).to(device)
  
  if config["model"]["load_model"]:
      checkpoint_path = Path(config["model"]["load_model"])
      if checkpoint_path.exists():
          state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
          state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
          model.load_state_dict(state_dict, strict=True)
          print(f"✅ Modèle chargé depuis {checkpoint_path}")
      else:
          print(f"⚠️ Checkpoint non trouvé : {checkpoint_path}")
  
  # Critère de perte
  print("🎯 Instanciation de la loss...")
  criterion = SpectralLoss(
      lambda_sam=config["training"]["lambda_sam"],
      lambda_mae=config["training"]["lambda_mae"],
      lambda_mse=config["training"]["lambda_mse"]
  ).to(device)

  # Optimiseur & Scheduler
  print("⚙️ Instanciation de l'optimiseur...")
  optimizer = build_optimizer(config, model.parameters())
  scheduler = build_lr_scheduler(optimizer, config)

  # Initialisation W&B
  print("🚀 Initialisation de Weights & Biases...")
  run_name = build_run_name(config)
  if args.output_tag:
      run_name = f"{run_name}_{args.output_tag}"

  run_name_wb = init_wandb(config, run_name=run_name)

  # === BOUCLE D'ENTRAÎNEMENT ===
  print(f"\n{'='*60}")
  print(f"🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
  print(f"Run name: {run_name_wb}")
  print(f"Epochs: {config['training']['epochs']}")
  print(f"LR: {config['training']['lr']}")
  print(f"{'='*60}\n")

  best_val_loss = float("inf")
  early_stopper = EarlyStopping(patience=15, min_delta=1e-4, start_epoch=10)

  for epoch in range(config["training"]["epochs"]):
      train_metrics = train_one_epoch(
          model, train_loader, optimizer, criterion, device, scaler=scaler
      )
      val_metrics = validate(
          model,
          val_loader,
          criterion,
          device,
          use_amp=use_amp,
          compute_full_metrics=True,
      )

      current_lr = optimizer.param_groups[0]["lr"]
      if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
          scheduler.step(val_metrics["val_loss"])
      else:
          scheduler.step()

      print(f"\n[Epoch {epoch+1}/{config['training']['epochs']}] — LR: {current_lr:.2e}")
      print(f"  Train → Loss: {train_metrics['train_loss']:.4f} | MAE: {train_metrics['train_mae']:.4f} | SAM: {train_metrics['train_sam_deg']:.2f}° | GradNorm: {train_metrics['train_grad_norm']:.2f}")
      print(f"  Val   → Loss: {val_metrics['val_loss']:.4f} | PSNR: {val_metrics['val_psnr']:.2f} dB | SAM: {val_metrics['val_sam_deg']:.2f}° | SSIM: {val_metrics.get('val_ssim', 0):.4f} | ERGAS: {val_metrics.get('val_ergas', 0):.2f}")

      wandb.log({"epoch": epoch + 1, "lr": current_lr, **train_metrics, **val_metrics})
      
      val_loss = val_metrics["val_loss"]
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, config, run_name_wb, is_best=True)

        root = get_project_root()
        checkpoint_path = root / config["experiment"]["output_dir"] / run_name_wb / "best_model.pth"
        artifact = wandb.Artifact(name=run_name_wb, type="model")
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
        print(f"   💾 Nouveau meilleur modèle sauvegardé !")

        print(f"   📊 Génération des patchs aléatoires (Val Epoch {epoch+1})...")
        # CORRECTION ICI (nom corrigé) :
        evaluate_and_log_random_patches(
            model=model,
            loader=val_loader,
            device=device,

            wvl_full=WVL_PRS,
            kept_indices=kept_indices,
            use_amp=use_amp,
            prefix=f"Val_Epoch_{epoch+1}",
        )

      early_stopper(val_loss, epoch=epoch + 1)

      if early_stopper.early_stop:
          print(f"\n⏹️ Arrêt précoce déclenché à l'époque {epoch+1} !")
          break

  print(f"\n{'='*60}")
  print(f"🏁 ENTRAÎNEMENT TERMINÉ")
  print(f"{'='*60}\n")

  # === ÉVALUATION FINALE SUR LE TEST SET ===
  print(f"\n{'='*60}")
  print("🧪 ÉVALUATION FINALE (TEST SET)")
  print(f"{'='*60}\n")

  root = get_project_root()
  best_ckpt = root / config["experiment"]["output_dir"] / run_name_wb / "best_model.pth"

  if best_ckpt.exists():
      model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
      print("   ✅ Meilleur checkpoint chargé pour l'évaluation.")
  else:
      print("   ⚠️ Checkpoint non trouvé, évaluation avec l'état actuel.")

  test_metrics = test(model, test_loader, criterion, device, use_amp=use_amp)

  print(f"\n📊 Résultats Test Globaux : MAE={test_metrics['mae']:.6f} | MSE={test_metrics['mse']:.6f} | SAM={test_metrics['sam']:.4f} rad ({np.degrees(test_metrics['sam']):.2f}°) | PSNR={test_metrics['psnr']:.2f} dB | SSIM={test_metrics['ssim']:.4f} | ERGAS={test_metrics['ergas']:.2f}")

  wandb.log({
      "test/loss": test_metrics["loss"],
      "test/mae": test_metrics["mae"],
      "test/mse": test_metrics["mse"],
      "test/rmse": test_metrics["rmse"],
      "test/sam_rad": test_metrics["sam"],
      "test/sam_deg": np.degrees(test_metrics["sam"]),
      "test/psnr": test_metrics["psnr"],
      "test/ssim": test_metrics["ssim"],
      "test/ergas": test_metrics["ergas"],
  })


  evaluate_and_log_random_patches(
      model=model,
      loader=test_loader,
      device=device,
      wvl_full=WVL_PRS,
      kept_indices=kept_indices,
      use_amp=use_amp,
      prefix="Test Final"
  )

  print(f"\n{'='*60}")
  print("🚀 ENTRAÎNEMENT ET ÉVALUATION TERMINÉS AVEC SUCCÈS")
  print(f"{'='*60}\n")

  wandb.finish()


if __name__ == "__main__":
    main()