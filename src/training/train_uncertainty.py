import argparse
from pathlib import Path
import sys

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
    build_run_name,
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
        base_channels=model_cfg.get('base_channel', 64),
        interpolation_mode=model_cfg.get('interpolation_mode', 'Bilinear'),
        activation=model_cfg.get('activation', 'silu'),
        final_op=model_cfg.get('final_activation', 'softplus'),
        drop_out_rate=model_cfg.get('drop_out_rate', 0.0),
    )
  else:
    raise ValueError(f"❌ Modèle d'incertitude non reconnu : '{name}'")
  
def build_run_name_uncertainty(config: dict) -> str:
  """Génère le nom du run spécifique aux modèles d'incertitude."""
  unc_cfg = config.get("model_uncertainty", {})
  train_cfg = config.get("training", {})

  model_name = unc_cfg.get("name", "uncertainty_model")
  base_ch = unc_cfg.get("base_channel", 64)
  lr = train_cfg.get("lr", 0.0001)
  loss_type = train_cfg.get("loss_type", "l1")

  return f"{model_name}_ch{base_ch}_{loss_type}_lr-{lr}"

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

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
  """Valide le modèle d'incertitude."""
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

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
      y_pred = model_reconstruction(x_init, x_interp)
      E_map = torch.abs(y_pred - y)
      unc_input = torch.cat([x_init, y_pred], dim=1)
      u_hat = model_uncertainty(unc_input)
      loss = criterion(u_hat, E_map)

    running_loss += loss.item()
    all_E.append(E_map.cpu().numpy().ravel())
    all_U.append(u_hat.cpu().numpy().ravel())

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

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
      y_pred = model_reconstruction(x_init, x_interp)
      E_map = torch.abs(y_pred - y)
      unc_input = torch.cat([x_init, y_pred], dim=1)
      u_hat = model_uncertainty(unc_input)
      loss = criterion(u_hat, E_map)

    running_loss += loss.item()
    all_E.append(E_map.cpu().numpy().ravel())
    all_U.append(u_hat.cpu().numpy().ravel())

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
    model_rec,
    model_unc,
    loader,
    device,
    kept_indices,
    output_dir,
    run_name,
    log_to_wandb=True,
):
  """Sélectionne le meilleur et le pire patch selon le résidu (|MAE - b|) et génère les planches."""
  model_rec.eval()
  model_unc.eval()

  patches_score = []

  for x_init, x_interp, y, patch_ids in loader:
    x_init, x_interp, y = (
        x_init.to(device),
        x_interp.to(device),
        y.to(device),
    )
    y_pred = model_rec(x_init, x_interp)
    unc_input = torch.cat([x_init, y_pred], dim=1)
    u_hat = model_unc(unc_input)

    mae_map = torch.abs(y_pred - y).mean(dim=1).cpu().numpy()
    u_map = u_hat.mean(dim=1).cpu().numpy()
    residual = np.abs(mae_map - u_map).mean(axis=(1, 2))

    for i in range(len(patch_ids)):
      patches_score.append({
          'residual': residual[i],
          'patch_id': patch_ids[i],
          'data': {
              'cube_gt': y[i].cpu().numpy().transpose(1, 2, 0),
              'cube_predict': y_pred[i].cpu().numpy().transpose(1, 2, 0),
              'cube_msi': x_init[i].cpu().numpy().transpose(1, 2, 0),
              'cube_uncertainty': u_hat[i].cpu().numpy().transpose(1, 2, 0),
              'model name': 'DualBranchNAFNet (Uncertainty)',
          },
      })

  patches_score.sort(key=lambda x: x['residual'])

  best_patch = patches_score[0]
  worst_patch = patches_score[-1]

  plot_dir = Path(output_dir) / run_name / 'plots'

  visualise_synthesis_uncertainty(
      best_patch['data'],
      save_name=f'best_patch_{best_patch["patch_id"]}',
      plot_dir=plot_dir,
      kept_indices=kept_indices,
      log_to_wandb=log_to_wandb,
  )

  visualise_synthesis_uncertainty(
      worst_patch['data'],
      save_name=f'worst_patch_{worst_patch["patch_id"]}',
      plot_dir=plot_dir,
      kept_indices=kept_indices,
      log_to_wandb=log_to_wandb,
  )


def main():
  parser = argparse.ArgumentParser(
      description='Entraînement de la quantification d\'incertitude (MSI->HSI)'
  )
  parser.add_argument(
      '--config',
      type=str,
      default='src/config/config_uncertainty.yaml',
      help='Chemin YAML',
  )
  parser.add_argument(
      '--override_epochs', type=int, default=None, help='Override epochs'
  )
  parser.add_argument(
      '--override_lr', type=float, default=None, help='Override LR'
  )
  parser.add_argument(
      '--output_tag', type=str, default='', help='Tag additionnel run'
  )

  args = parser.parse_args()

  # 1. Configuration & Initialisation
  config = load_config(args.config)
  if args.override_epochs:
    config['training']['epochs'] = args.override_epochs
  if args.override_lr:
    config['training']['lr'] = args.override_lr

  seed_val = config.get('experiment', {}).get('seed', 42)
  set_seed(seed_val)
  device = get_device()

  use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
  scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

  kept_indices = get_kept_wavelength_indices(WVL_PRS, config)

  # 2. Dataloaders
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

  sample_x, sample_x_interp, sample_y, _ = next(iter(train_loader))
  n_msi = sample_x.shape[1]
  n_hsi = sample_y.shape[1]

  # 3. Chargement du modèle de reconstruction (Gelé)
  print('🔒 Chargement du modèle de reconstruction pré-entraîné...')
  rec_cfg = config['model_reconstruction']
  model_reconstruction = build_model_reconstruction(
      rec_cfg, in_channels=n_msi, n_hsi=n_hsi
  ).to(device)

  rec_ckpt_path = Path(
      rec_cfg.get('load_model', 'checkpoints/best_model_rec.pth')
  )

  if rec_ckpt_path.exists():
    state_dict = torch.load(rec_ckpt_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_reconstruction.load_state_dict(state_dict, strict=True)
    print(f'   ✅ Reconstruction chargée depuis : {rec_ckpt_path}')
  else:
    raise FileNotFoundError(
        f'❌ Checkpoint de reconstruction introuvable : {rec_ckpt_path}'
    )

  for param in model_reconstruction.parameters():
    param.requires_grad = False
  model_reconstruction.eval()

  # 4. Instanciation du modèle d'incertitude (Entraîné)
  print('🔥 Instanciation du modèle d\'incertitude...')
  unc_cfg = config['model_uncertainty']
  model_uncertainty = build_model_uncertainty(
      unc_cfg, in_channels=(n_msi + n_hsi), n_hsi=n_hsi
  ).to(device)

  load_unc_ckpt = unc_cfg.get('load_model_uncertainty')
  if load_unc_ckpt and Path(load_unc_ckpt).exists():
    state_dict = torch.load(load_unc_ckpt, map_location=device)
    model_uncertainty.load_state_dict(state_dict, strict=True)
    print(
        f'   ✅ Poids initiaux d\'incertitude chargés depuis : {load_unc_ckpt}'
    )

  # 5. Loss, Optimiseur et Scheduler
  criterion = L1_uncertainty
  optimizer = build_optimizer(config, model_uncertainty.parameters())
  scheduler = build_lr_scheduler(optimizer, config)

  # 6. Initialisation WandB
  config_for_name = config.copy()
  config_for_name['model'] = config.get(
      'model_uncertainty', config.get('model_reconstruction', {})
  )
  # 6. Initialisation WandB
  run_name = build_run_name_uncertainty(config) + "_UNCERTAINTY"
  if args.output_tag:
    run_name = f"{run_name}_{args.output_tag}"

  run_name_wb = init_wandb(config, run_name=run_name)
  # 7. Early Stopping & Boucle d'entraînement
  best_val_loss = float('inf')
  patience = config['training'].get('patience', 15)
  min_delta = config['training'].get('min_delta', 0.0001)
  early_stopper = EarlyStopping(
      patience=patience, min_delta=min_delta, start_epoch=10
  )

  for epoch in range(config['training']['epochs']):
    train_metrics = train_one_epoch(
        model_reconstruction,
        model_uncertainty,
        train_loader,
        optimizer,
        criterion,
        device,
        scaler=scaler,
    )
    val_metrics = validate(
        model_reconstruction,
        model_uncertainty,
        val_loader,
        criterion,
        device,
        use_amp=use_amp,
    )

    current_lr = optimizer.param_groups[0]['lr']
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
      scheduler.step(val_metrics['val_loss_uncertainty'])
    elif scheduler is not None:
      scheduler.step()

    print(
        f'\n[Epoch {epoch+1}/{config["training"]["epochs"]}] — LR:'
        f' {current_lr:.2e}'
    )
    print(f"  Train Loss : {train_metrics['train_loss_uncertainty']:.6f}")
    print(
        f"  Val Loss   : {val_metrics['val_loss_uncertainty']:.6f} | Spearman"
        f" : {val_metrics['val_spearman']:.4f} | PICP 95% :"
        f" {val_metrics['val_picp_95']*100:.1f}%"
    )

    wandb.log(
        {'epoch': epoch + 1, 'lr': current_lr, **train_metrics, **val_metrics}
    )

    val_loss = val_metrics['val_loss_uncertainty']
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(model_uncertainty, config, run_name_wb, is_best=True)
      print("   💾 Meilleur modèle d'incertitude sauvegardé !")

    early_stopper(val_loss, epoch=epoch + 1)
    if early_stopper.early_stop:
      print(f"⏹️ Early stopping déclenché à l'époque {epoch+1}")
      break

  # 8. Évaluation finale (Test Set)
  print('\n🧪 ÉVALUATION FINALE (TEST SET)')
  root = get_project_root()
  best_ckpt = (
      root
      / config['experiment']['output_dir']
      / run_name_wb
      / 'best_model.pth'
  )

  if best_ckpt.exists():
    model_uncertainty.load_state_dict(
        torch.load(best_ckpt, map_location=device)
    )

  test_metrics = test(
      model_reconstruction,
      model_uncertainty,
      test_loader,
      criterion,
      device,
      use_amp=use_amp,
  )
  print(
      f"📊 Test Loss: {test_metrics['test_loss_uncertainty']:.6f} | Spearman:"
      f" {test_metrics['test_spearman']:.4f} | AUSE:"
      f" {test_metrics['test_ause']:.6f} | PICP 95%:"
      f" {test_metrics['test_picp_95']*100:.2f}%"
  )

  wandb.log(test_metrics)

  # 9. Génération des planches d'extrêmes
  evaluate_and_log_uncertainty(
      model_reconstruction,
      model_uncertainty,
      test_loader,
      device,
      kept_indices,
      config['experiment']['output_dir'],
      run_name_wb,
      log_to_wandb=True,
  )

  wandb.finish()


if __name__ == '__main__':
  main()