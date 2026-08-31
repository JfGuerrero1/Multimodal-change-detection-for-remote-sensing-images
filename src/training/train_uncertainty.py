import argparse
import os
from pathlib import Path
import sys

# Ajoute la racine du projet au sys.path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy.stats import laplace, spearmanr
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from src.constants import WVL_PRS
from src.metrics_and_loss.loss import L1_uncertainty,LaplaceNLLLossDirect
from src.models import (
    DualBranchNAFNet,
    DualBranchUNet,
    GradualExpansionUNet,
    GradualExpansionUNet_residual,
)
from src.prepare_data.prepare_patch import create_data_loaders_spectral
from src.training.utils_train import (
    EarlyStopping,
    build_lr_scheduler,
    build_optimizer,
    get_device,
    get_kept_wavelength_indices,
    get_project_root,
    init_wandb,
    load_config,
    save_checkpoint,
    set_seed,
)
from src.visualise.visualise_uncertainty import visualise_synthesis_uncertainty


# =====================================================================
# METRIQUES D'INCERTITUDE OPTIMISÉES
# =====================================================================
def compute_fast_ause(
    E: np.ndarray, U: np.ndarray, num_bins: int = 100
) -> float:
  """Calcule l'AUSE (Area Under Sparsification Error) vectorisée sur CPU/NumPy."""
  n_samples = len(E)
  if n_samples > 100_000:
    idx = np.random.choice(n_samples, size=100_000, replace=False)
    E, U = E[idx], U[idx]
    n_samples = 100_000

  idx_m = np.argsort(U)[::-1]
  idx_o = np.argsort(E)[::-1]

  sum_m = np.cumsum(E[idx_m][::-1])[::-1]
  sum_o = np.cumsum(E[idx_o][::-1])[::-1]

  k_indices = np.linspace(
      0, n_samples - 1, num_bins, endpoint=False, dtype=int
  )
  counts = n_samples - k_indices

  curve_m = sum_m[k_indices] / counts
  curve_o = sum_o[k_indices] / counts

  x_axis = np.linspace(0.0, 1.0, num_bins)
  trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

  auc_m = trapz_func(curve_m, x=x_axis)
  auc_o = trapz_func(curve_o, x=x_axis)

  return float(auc_m - auc_o)


def build_model_reconstruction(
    model_cfg: dict, in_channels: int, n_hsi: int
) -> torch.nn.Module:
  """Instancie dynamiquement le modèle de reconstruction (MSI -> HSI)."""
  name = model_cfg.get('name', '').lower()

  if name == 'gradualexpansionunet_res':
    return GradualExpansionUNet_residual(
        in_msi=in_channels,
        in_hsi=n_hsi,
        interpolation_mode=model_cfg.get('interpolation_mode', 'Bilinear'),
        base_channel=model_cfg.get('base_channel', 64),
        activation=model_cfg.get('activation', 'silu'),
        with_batch_norm=model_cfg.get('with_batch_norm', True),
        with_mlp_spectral=model_cfg.get('with_mlp_spectral', False),
        final_activation=model_cfg.get('final_activation', None),
    )
  elif name == 'gradualexpansionunet':
    return GradualExpansionUNet(
        in_msi=in_channels,
        in_hsi=n_hsi,
        interpolation_mode=model_cfg.get('interpolation_mode', 'Bilinear'),
        base_channel=model_cfg.get('base_channel', 64),
        activation=model_cfg.get('activation', 'silu'),
        with_batch_norm=model_cfg.get('with_batch_norm', True),
        with_mlp_spectral=model_cfg.get('with_mlp_spectral', False),
        final_activation=model_cfg.get('final_activation', None),
    )
  else:
    raise ValueError(f"❌ Modèle de reconstruction non reconnu : '{name}'")


def build_model_uncertainty(
    model_cfg: dict, in_channels: int, n_hsi: int
) -> torch.nn.Module:
  """Instancie dynamiquement le modèle d'incertitude."""
  name = model_cfg.get('name', '').lower()

  if name in ['dualbranchunet', 'uncertainty_unet']:
    return DualBranchUNet(
        n_msi=in_channels,
        n_hsi=n_hsi,
        base_channels=model_cfg.get('base_channel', 64),
        interpolation_mode=model_cfg.get('interpolation_mode', 'Bilinear'),
        activation=model_cfg.get('activation', 'silu'),
        final_op=model_cfg.get('final_activation', 'softplus'),
        drop_out_rate=model_cfg.get('drop_out_rate', 0.0),
    )
  elif name in ['dualbranchnafnet', 'uncertainty_nafnet']:
    return DualBranchNAFNet(
        n_msi=in_channels,
        n_hsi=n_hsi,
        out_channels=n_hsi,
        width=model_cfg.get('base_channel', 64),
        drop_out_rate=model_cfg.get('drop_out_rate', 0.0),
        final_op=model_cfg.get('final_activation', 'softplus'),
    )
  else:
    raise ValueError(f"❌ Modèle d'incertitude non reconnu : '{name}'")


def build_run_name_uncertainty(config: dict) -> str:
  """Génère le nom du run spécifique aux modèles d'incertitude."""
  unc_cfg = config.get('model_uncertainty', {})
  train_cfg = config.get('training', {})

  model_name = unc_cfg.get('name', 'uncertainty_model')
  base_ch = unc_cfg.get('base_channel', 64)
  lr = train_cfg.get('lr', 0.0001)
  loss_type = train_cfg.get('loss_type', 'l1')

  return f'{model_name}_ch{base_ch}_{loss_type}_lr-{lr}'


def train_one_epoch(
    model_reconstruction: torch.nn.Module,
    model_uncertainty: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    scaler: torch.amp.GradScaler = None,
) -> dict:
  """Entraîne le modèle d'incertitude sur une époque."""
  model_reconstruction.eval()
  model_uncertainty.train()

  use_amp = scaler is not None and device.type == 'cuda'
  running_loss = 0.0
  pbar = tqdm(loader, desc='Training Uncertainty')

  for x_init, x_interp, y, patch_ids in pbar:
    x_init = x_init.to(device, non_blocking=True)
    x_interp = x_interp.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    optimizer.zero_grad()

    with torch.no_grad():
      y_pred = model_reconstruction(x_init, x_interp)
      E_map = torch.abs(y_pred - y)

    with torch.amp.autocast(device_type=device.type, enabled=bool(use_amp)):
      unc_input = torch.cat([x_init, y_pred.detach()], dim=1)
      u_hat = model_uncertainty(unc_input)
      loss = criterion(u_hat, E_map)

    if use_amp:
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(
          model_uncertainty.parameters(), max_norm=1.0
      )
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
          model_uncertainty.parameters(), max_norm=1.0
      )
      optimizer.step()

    running_loss += loss.item()
    pbar.set_postfix({'loss_unc': f'{loss.item():.6f}'})

  return {'train_loss_uncertainty': running_loss / len(loader)}


@torch.no_grad()
def validate(
    model_reconstruction: torch.nn.Module,
    model_uncertainty: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
  """Valide le modèle d'incertitude sans fuite de mémoire RAM."""
  model_reconstruction.eval()
  model_uncertainty.eval()

  running_loss = 0.0
  all_E = []
  all_U = []

  pbar = tqdm(loader, desc='Validation')
  for x_init, x_interp, y, _ in pbar:
    x_init = x_init.to(device, non_blocking=True)
    x_interp = x_interp.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.amp.autocast(device_type=device.type, enabled=bool(use_amp)):
      y_pred = model_reconstruction(x_init, x_interp)
      E_map = torch.abs(y_pred - y)
      unc_input = torch.cat([x_init, y_pred], dim=1)
      u_hat = model_uncertainty(unc_input)
      loss = criterion(u_hat, E_map)

    running_loss += loss.item()

    sub_E = E_map.detach().cpu().numpy().ravel()
    sub_U = u_hat.detach().cpu().numpy().ravel()

    if len(sub_E) > 50_000:
      idx = np.random.choice(len(sub_E), size=50_000, replace=False)
      sub_E, sub_U = sub_E[idx], sub_U[idx]

    all_E.append(sub_E.astype(np.float32))
    all_U.append(sub_U.astype(np.float32))

  E_flat = np.concatenate(all_E)
  U_flat = np.concatenate(all_U)

  picp_95 = float(np.mean(E_flat <= 3.0 * U_flat))
  spearman_corr, _ = spearmanr(U_flat[::100], E_flat[::100])

  return {
      'val_loss_uncertainty': running_loss / len(loader),
      'val_picp_95': picp_95,
      'val_spearman': float(spearman_corr),
  }


@torch.no_grad()
def test(
    model_reconstruction: torch.nn.Module,
    model_uncertainty: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
  """Évaluation de test complète avec AUSE et métriques d'incertitude."""
  model_reconstruction.eval()
  model_uncertainty.eval()

  running_loss = 0.0
  all_E, all_U = [], []

  pbar = tqdm(loader, desc='Testing')
  for x_init, x_interp, y, _ in pbar:
    x_init = x_init.to(device, non_blocking=True)
    x_interp = x_interp.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.amp.autocast(device_type=device.type, enabled=bool(use_amp)):
      y_pred = model_reconstruction(x_init, x_interp)
      E_map = torch.abs(y_pred - y)
      unc_input = torch.cat([x_init, y_pred], dim=1)
      u_hat = model_uncertainty(unc_input)
      loss = criterion(u_hat, E_map)

    running_loss += loss.item()

    sub_E = E_map.detach().cpu().numpy().ravel()
    sub_U = u_hat.detach().cpu().numpy().ravel()

    if len(sub_E) > 50_000:
      idx = np.random.choice(len(sub_E), size=50_000, replace=False)
      sub_E, sub_U = sub_E[idx], sub_U[idx]

    all_E.append(sub_E.astype(np.float32))
    all_U.append(sub_U.astype(np.float32))

  E_flat = np.concatenate(all_E)
  U_flat = np.concatenate(all_U)

  spearman_corr, _ = spearmanr(U_flat[::10], E_flat[::10])
  ause_score = compute_fast_ause(E_flat, U_flat)
  picp_95 = float(np.mean(E_flat <= 3.0 * U_flat))

  return {
      'test_loss_uncertainty': running_loss / len(loader),
      'test_spearman': float(spearman_corr),
      'test_ause': ause_score,
      'test_picp_95': picp_95,
  }


def get_fixed_random_patches(loader, n_per_scene: int = 2, seed: int = 42) -> dict:
    """Scanne le loader une seule fois pour pré-sélectionner N patchs par scène."""
    scene_to_patches = {}
    for _, _, _, patch_ids in loader:
        for p_id in patch_ids:
            p_id_val = p_id.item() if hasattr(p_id, 'item') else p_id
            p_str = str(p_id_val)
            
            # Extraction propre du nom de la scène (ex: "beheira_after_patch_27" -> "beheira_after")
            if "_patch_" in p_str:
                scene_id = p_str.rsplit("_patch_", 1)[0]
            elif "_" in p_str:
                scene_id = "_".join(p_str.split("_")[:-1])
            else:
                scene_id = "unknown"

            if scene_id not in scene_to_patches:
                scene_to_patches[scene_id] = []
            if p_id_val not in scene_to_patches[scene_id]:
                scene_to_patches[scene_id].append(p_id_val)

    rng = np.random.default_rng(seed)
    fixed_targets = {}
    for scene_id, p_ids in scene_to_patches.items():
        k = min(n_per_scene, len(p_ids))
        fixed_targets[scene_id] = list(rng.choice(p_ids, size=k, replace=False))
    return fixed_targets


@torch.no_grad()

def evaluate_and_log_uncertainty(
    model_rec: torch.nn.Module,
    model_unc: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    wvl_full: np.ndarray,
    kept_indices: np.ndarray = None,
    use_amp: bool = False,
    fixed_target_patches: dict = None,
    prefix: str = "Val",
):
  """Inférence ultra-rapide : ne traite et log que les patchs fixes pré-sélectionnés."""
  model_rec.eval()
  model_unc.eval()
  if not fixed_target_patches:
    return

  found_patches = {scene_id: set() for scene_id in fixed_target_patches.keys()}
  total_to_find = sum(len(v) for v in fixed_target_patches.values())
  found_count = 0

  print(f"\n Log ultra-rapide des {total_to_find} patchs fixes ({prefix})...")

  for x_init, x_interp, y, patch_ids in loader:
    if found_count >= total_to_find:
      break  # Arrêt immédiat dès que tous les patchs cibles sont trouvés

    batch_size = y.size(0)
    batch_indices_to_process = []

    for i in range(batch_size):
      raw_id = patch_ids[i]
      p_id = raw_id.item() if hasattr(raw_id, 'item') else raw_id
      
      try:
          p_id_int = int(p_id)
          scene_id = (p_id_int // 36) + 1
      except (ValueError, TypeError):
          scene_id = str(p_id).split("_")[1] if "_" in str(p_id) else "unknown"

      if scene_id in fixed_target_patches and p_id in fixed_target_patches[scene_id]:
        if p_id not in found_patches[scene_id]:
          batch_indices_to_process.append((i, p_id, scene_id))

    if not batch_indices_to_process:
      continue

    x_init_b = x_init.to(device, non_blocking=True)
    x_interp_b = x_interp.to(device, non_blocking=True)
    y_b = y.to(device, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", enabled=bool(use_amp)):
      pred = model_rec(x_init_b, x_interp_b)
      unc_input = torch.cat([x_init_b, pred], dim=1)
      u_hat = model_unc(unc_input)

    pred_np = pred.detach().cpu().numpy()
    y_np = y_b.detach().cpu().numpy()
    x_init_np = x_init_b.detach().cpu().numpy()
    u_hat_np = u_hat.detach().cpu().numpy()
    
    for idx_batch, p_id, scene_id in batch_indices_to_process:
      if p_id in found_patches[scene_id]:
        continue

      gt_hwc = np.moveaxis(y_np[idx_batch], 0, -1) if y_np[idx_batch].shape[0] < y_np[idx_batch].shape[-1] else y_np[idx_batch]
      pred_hwc = np.moveaxis(pred_np[idx_batch], 0, -1) if pred_np[idx_batch].shape[0] < pred_np[idx_batch].shape[-1] else pred_np[idx_batch]
      msi_hwc = np.moveaxis(x_init_np[idx_batch], 0, -1) if x_init_np[idx_batch].shape[0] < x_init_np[idx_batch].shape[-1] else x_init_np[idx_batch]
      unc_hwc = np.moveaxis(u_hat_np[idx_batch], 0, -1) if u_hat_np[idx_batch].shape[0] < u_hat_np[idx_batch].shape[-1] else u_hat_np[idx_batch]

      mae = float(np.mean(np.abs(gt_hwc - pred_hwc)))
      dot = np.sum(pred_hwc * gt_hwc, axis=-1)
      norm_p = np.linalg.norm(pred_hwc, axis=-1)
      norm_g = np.linalg.norm(gt_hwc, axis=-1)
      sam_map = np.arccos(np.clip(dot / (norm_p * norm_g + 1e-8), -1.0, 1.0))
      sam_rad = float(np.mean(sam_map))

      save_name = f"{prefix}_Scene_{scene_id}_Patch_{p_id}"

      data_to_plot = {
          "cube_gt": gt_hwc,
          "cube_predict": pred_hwc,
          "cube_msi": msi_hwc,
          "cube_uncertainty": unc_hwc,
          "model name": f"{prefix} — Scène {scene_id} (Patch {p_id})",
          "img_mae": mae,
          "img_sam": sam_rad,
      }

      visualise_synthesis_uncertainty(
          data=data_to_plot,
          save_name=save_name,
          plot_dir=None,
          kept_indices=kept_indices,
          log_to_wandb=True,
      )

      found_patches[scene_id].add(p_id)
      found_count += 1
  print("log des patchs terminés")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='src/config/config_uncertainty.yaml')
  parser.add_argument('--override_epochs', type=int, default=None)
  parser.add_argument('--override_lr', type=float, default=None)
  parser.add_argument('--output_tag', type=str, default='')
  args = parser.parse_args()

  config = load_config(args.config)
  if args.override_epochs: config['training']['epochs'] = args.override_epochs
  if args.override_lr: config['training']['lr'] = args.override_lr

  set_seed(config.get('experiment', {}).get('seed', 42))
  device = get_device()
  use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
  scaler = torch.amp.GradScaler(enabled=bool(use_amp)) if use_amp else None
  kept_indices = get_kept_wavelength_indices(WVL_PRS, config)

  train_loader, val_loader, test_loader = create_data_loaders_spectral(
      train_dir=config['data']['data_dir_train'],
      val_dir=config['data']['data_dir_val'],
      test_dir=config['data']['data_dir_test'],
      simulated=config['data']['simulated'],
      augment=config['data']['augment'],
      augment_illumination=config['data']['augment_illumination'],
      batch_size=config['data']['batch_size'],
      num_workers=config['data']['num_workers'],
      is_residual=config['data']['is_residual'],
      is_normalised=config['data']['is_normalised'],
      kept_indices=kept_indices,
  )

  fixed_val_patches = get_fixed_random_patches(val_loader, n_per_scene=2, seed=42)
  fixed_test_patches = get_fixed_random_patches(test_loader, n_per_scene=2, seed=42)

  sample_x, _, sample_y, _ = next(iter(train_loader))
  n_msi, n_hsi = sample_x.shape[1], sample_y.shape[1]

  model_reconstruction = build_model_reconstruction(config['model_reconstruction'], n_msi, n_hsi).to(device)
  rec_ckpt = Path(config['model_reconstruction'].get('load_model', 'checkpoints/best_model_rec.pth'))
  
  if rec_ckpt.exists():
    state_dict = {k.replace('module.', ''): v for k, v in torch.load(rec_ckpt, map_location=device, weights_only=True).items()}
    model_reconstruction.load_state_dict(state_dict, strict=True)
    model_reconstruction.eval()
    for param in model_reconstruction.parameters(): param.requires_grad = False
  else:
    raise FileNotFoundError(f"❌ Checkpoint non trouvé : {rec_ckpt}")


  model_uncertainty = build_model_uncertainty(config['model_uncertainty'], n_msi, n_hsi).to(device)

  loss_type = config["training"]["loss_type"]
  
  if loss_type == "laplace_nll":
    criterion = LaplaceNLLLossDirect()
  else:
    criterion = L1_uncertainty()

  optimizer = build_optimizer(config, model_uncertainty.parameters())
  scheduler = build_lr_scheduler(optimizer, config)
  run_name = build_run_name_uncertainty(config) + '_UNCERTAINTY' + (f'_{args.output_tag}' if args.output_tag else '')

  
  run = wandb.init(
    project="Multimodal-change-detection", # Le nom de ton projet W&B
    name=run_name,                          # Nom unique du run généré plus haut
    config=config,                          # Sauvegarde les hyperparamètres
    reinit=True
)
  run_name_wb = run.name

  best_val_loss = float('inf')
  early_stopper = EarlyStopping(patience=config['training'].get('patience', 15), start_epoch=10)

  for epoch in range(config['training']['epochs']):
    train_metrics = train_one_epoch(model_reconstruction, model_uncertainty, train_loader, optimizer, criterion, device, scaler=scaler)
    val_metrics = validate(model_reconstruction, model_uncertainty, val_loader, criterion, device, use_amp=use_amp)

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_metrics['val_loss_uncertainty'])
    elif scheduler is not None: scheduler.step()

    wandb.log({'epoch': epoch + 1, **train_metrics, **val_metrics})

    if val_metrics['val_loss_uncertainty'] < best_val_loss:
      best_val_loss = val_metrics['val_loss_uncertainty']
      save_checkpoint(model_uncertainty, config, run_name_wb, is_best=True)
      
      evaluate_and_log_uncertainty(
          model_rec=model_reconstruction,
          model_unc=model_uncertainty,
          loader=val_loader,
          device=device,
          wvl_full=WVL_PRS,
          kept_indices=kept_indices,
          use_amp=use_amp,
          fixed_target_patches=fixed_val_patches,
          prefix="Val_best"
      )

    early_stopper(val_metrics['val_loss_uncertainty'], epoch=epoch + 1)
    if early_stopper.early_stop: break

  best_ckpt = get_project_root() / config['experiment']['output_dir'] / run_name_wb / 'best_model.pth'
  if best_ckpt.exists(): model_uncertainty.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
  
  test_metrics = test(model_reconstruction, model_uncertainty, test_loader, criterion, device, use_amp=use_amp)
  wandb.log(test_metrics)

  evaluate_and_log_uncertainty(
      model_rec=model_reconstruction,
      model_unc=model_uncertainty,
      loader=test_loader,
      device=device,
      wvl_full=WVL_PRS,
      kept_indices=kept_indices,
      use_amp=use_amp,
      fixed_target_patches=fixed_test_patches,
      prefix="Test"
  )

  # --- Vérification rapide de l'hypothèse de Laplace ---
  # --- Vérification rapide de l'hypothèse de Laplace (Anti-OOM) ---
  print("\n📊 Vérification empirique de la loi de Laplace sur le Test Set...")
  all_errors = []
  model_reconstruction.eval()

  for x_init, x_interp, y, _ in test_loader:
    with torch.no_grad():
      y_pred = model_reconstruction(x_init.to(device), x_interp.to(device))
      signed_errors = (y_pred - y.to(device)).detach().cpu().numpy().ravel()
      
      # 🛡️ Sous-échantillonnage direct par batch pour éviter l'explosion RAM
      if len(signed_errors) > 10_000:
        idx = np.random.choice(len(signed_errors), size=10_000, replace=False)
        signed_errors = signed_errors[idx]
        
      all_errors.append(signed_errors.astype(np.float32))

  # On fusionne un échantillon robuste mais contrôlé (max ~quelques centaines de milliers de points)
  errors_flat = np.concatenate(all_errors)
  if len(errors_flat) > 200_000:
    idx = np.random.choice(len(errors_flat), size=200_000, replace=False)
    errors_flat = errors_flat[idx]

  loc, scale = laplace.fit(errors_flat)

  print(f"✅ Ajustement Laplace terminé :")
  print(f"   -> Position (loc / moyenne) : {loc:.5f}")
  print(f"   -> Échelle (scale / b)      : {scale:.5f}")

  wandb.log({"laplace_loc": loc, "laplace_scale": scale})
  wandb.log({"errors_histogram": wandb.Histogram(errors_flat)})

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.hist(
      errors_flat,
      bins=200,
      density=True,
      alpha=0.6,
      color="skyblue",
      label="Erreurs empiriques (échantillonnées)",
  )

  xmin, xmax = ax.get_xlim()
  x = np.linspace(xmin, xmax, 1000)
  p = laplace.pdf(x, loc, scale)
  # Remplacement de "r-" par "r--" (ou "r:") et réduction du linewidth
  ax.plot(x, p, linestyle="--", color="red", linewidth=1.0, label="Loi de Laplace théorique")

  ax.set_title(f"Ajustement Loi de Laplace\nloc={loc:.5f}, scale={scale:.5f}")
  ax.set_xlabel("Erreur signée (Prédiction - Vérité)")
  ax.set_ylabel("Densité")
  ax.legend()
  ax.grid(True, alpha=0.3)
  wandb.log({"laplace_fit_plot": wandb.Image(fig)})

  wandb.finish()

if __name__ == '__main__':
  main()