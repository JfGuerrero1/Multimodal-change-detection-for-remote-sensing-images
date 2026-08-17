import argparse
import os
from pathlib import Path
import sys
import tempfile

# Ajoute la racine du projet au sys.path de Python
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import wandb

from src.constants import WVL_PRS
from src.metrics_and_loss.loss import L1_uncertainty
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
    raise ValueError(
        f"❌ Modèle de reconstruction non reconnu : '{name}'"
    )


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
        drop_out_rate=model_cfg.get('drop_out_rate', 0.1),
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


@torch.no_grad()
def evaluate_and_log_uncertainty(
    model_rec: torch.nn.Module,
    model_unc: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    wvl_full: np.ndarray,
    kept_indices: np.ndarray = None,
    use_amp: bool = False,
    n_worst: int = 2,
    n_best: int = 1,
    prefix: str = "Val Best",
):
  """Évalue les modèles et gère les pires/meilleurs patchs via un stockage

  disque temporaire pour éviter toute saturation de la RAM (OOM).
  """
  model_rec.eval()
  model_unc.eval()

  with tempfile.TemporaryDirectory() as tmpdir:
    scenes_dict = {}
    patch_counter = 0

    print("\n🔍 [1/2] Inférence et sauvegarde sur disque temporaire...")

    for x_init, x_interp, y, patch_ids in tqdm(loader, desc="Inférence"):
      x_init = x_init.to(device, non_blocking=True)
      x_interp = x_interp.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      with torch.amp.autocast(device_type="cuda", enabled=bool(use_amp)):
        pred = model_rec(x_init, x_interp)
        unc_input = torch.cat([x_init, pred], dim=1)
        u_hat = model_unc(unc_input)

      pred_np = pred.detach().cpu().numpy()
      y_np = y.detach().cpu().numpy()
      x_init_np = x_init.detach().cpu().numpy()
      u_hat_np = u_hat.detach().cpu().numpy()

      batch_size = y.size(0)

      for i in range(batch_size):
        p_id = (patch_ids[i] if isinstance(patch_ids, list) else patch_ids[i].item())
        scene_id = (p_id // 36) + 1 if isinstance(p_id, int) else str(p_id).split("_")[1]

        gt_hwc = np.moveaxis(y_np[i], 0, -1) if y_np[i].shape[0] < y_np[i].shape[-1] else y_np[i]
        pred_hwc = np.moveaxis(pred_np[i], 0, -1) if pred_np[i].shape[0] < pred_np[i].shape[-1] else pred_np[i]
        msi_hwc = np.moveaxis(x_init_np[i], 0, -1) if x_init_np[i].shape[0] < x_init_np[i].shape[-1] else x_init_np[i]
        unc_hwc = np.moveaxis(u_hat_np[i], 0, -1) if u_hat_np[i].shape[0] < u_hat_np[i].shape[-1] else u_hat_np[i]

        mae = float(np.mean(np.abs(gt_hwc - pred_hwc)))

        dot = np.sum(pred_hwc * gt_hwc, axis=-1)
        norm_p = np.linalg.norm(pred_hwc, axis=-1)
        norm_g = np.linalg.norm(gt_hwc, axis=-1)
        sam_map = np.arccos(np.clip(dot / (norm_p * norm_g + 1e-8), -1.0, 1.0))
        sam_rad = float(np.mean(sam_map))

        temp_file_path = os.path.join(tmpdir, f"patch_{patch_counter}.npz")
        np.savez_compressed(
            temp_file_path,
            cube_gt=gt_hwc,
            cube_predict=pred_hwc,
            cube_msi=msi_hwc,
            cube_uncertainty=unc_hwc
        )

        patch_data = {
            "patch_id": p_id,
            "sam_rad": sam_rad,
            "sam_deg": float(np.degrees(sam_rad)),
            "mae": mae,
            "file_path": temp_file_path
        }

        if scene_id not in scenes_dict:
          scenes_dict[scene_id] = []
        scenes_dict[scene_id].append(patch_data)
        patch_counter += 1

    print(f"\n[2/2] Sélection des extrêmes et envoi sur WandB...")

    for scene_id, patches in scenes_dict.items():
      patches.sort(key=lambda x: x["sam_rad"], reverse=True)

      worst_patches = patches[:n_worst]
      best_patches = patches[-n_best:]

      to_log = []
      for rank, p in enumerate(worst_patches, start=1):
        to_log.append((f"Pire_#{rank}", p))
      for rank, p in enumerate(reversed(best_patches), start=1):
        to_log.append((f"Meilleur_#{rank}", p))

      for label, item in to_log:
        loaded = np.load(item["file_path"])

        save_name = f"Scene_{scene_id}_{label}_Patch_{item['patch_id']}_SAM_{item['sam_deg']:.2f}_deg.png"

        data_to_plot = {
            "cube_gt": loaded["cube_gt"],
            "cube_predict": loaded["cube_predict"],
            "cube_msi": loaded["cube_msi"],
            "cube_uncertainty": loaded["cube_uncertainty"],
            "model name": f"{prefix}_Scène {scene_id} — {label} (Patch {item['patch_id']})",
            "img_mae": item["mae"],
            "img_sam": item["sam_rad"],
        }

        visualise_synthesis_uncertainty(
            data=data_to_plot,
            save_name=save_name,
            plot_dir=None,
            kept_indices=kept_indices,
            log_to_wandb=True,
        )

  print("✅ Évaluation par scène terminée sans saturer la RAM !")


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
  
  criterion = L1_uncertainty()
  optimizer = build_optimizer(config, model_uncertainty.parameters())
  scheduler = build_lr_scheduler(optimizer, config)
  run_name = build_run_name_uncertainty(config) + '_UNCERTAINTY' + (f'_{args.output_tag}' if args.output_tag else '')
  run_name_wb = init_wandb(config, run_name=run_name)

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
          prefix="Val_best"
      )

    early_stopper(val_metrics['val_loss_uncertainty'], epoch=epoch + 1)
    if early_stopper.early_stop: break

  # Test final
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
      prefix="Test"
  )
  wandb.finish()


if __name__ == '__main__':
  main()