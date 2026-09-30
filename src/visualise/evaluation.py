#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║             SCRIPT D'ÉVALUATION ET D'INFÉRENCE MSI / HSI                   ║
║  - Reconstruction spectrale (GradualExpansionUNet & Résiduel)              ║
║  - Quantification d'incertitude (DualBranchNAFNet)                         ║
║  - Exportation de métriques (CSV) et de memmaps synchronisés               ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Ajout de la racine du projet au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.constants import WVL_PRS
from src.metrics_and_loss.metrics import (
    compute_ergas,
    compute_mae,
    compute_psnr,
    compute_sam,
    compute_ssim_multiband,
)
from src.models.dual_branch import DualBranchNAFNet
from src.models.unet_residual import GradualExpansionUNet_residual
from src.models.unet_standard import GradualExpansionUNet
from src.prepare_data.prepare_patch import create_data_loaders_spectral
from src.training.utils_train import get_kept_wavelength_indices
from src.visualise.visualisation import visualise_synthesis
from src.visualise.visualisation_spectre import (
    supervise_analyse_spectrale,
    trace_spectre,
    visualise_curve,
)
from src.visualise.visualise_uncertainty import visualise_synthesis_uncertainty


# ============================================================================
# UTILS & CHARGEMENT
# ============================================================================

def load_config(config_path: str) -> dict:
    """Charge un fichier de configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_scene_name(patch_id: str | int) -> str:
    """Extrait le nom de la scène source à partir de l'ID du patch."""
    p_str = str(patch_id)
    scene = re.sub(r"(_patch_.*|_[\d]+)$", "", p_str)
    return scene if scene else "scene_inconnue"


def to_hwc(arr: np.ndarray) -> np.ndarray:
    """Convertit un tenseur/tableau du format CxHxW vers HxWxC pour la visualisation."""
    return np.moveaxis(arr, 0, -1) if arr.shape[0] < arr.shape[-1] else arr


def setup_evaluation_context(config: dict, device: torch.device):
    """
    Initialise les DataLoaders et extrait les dimensions caractéristiques des données.
    """
    eval_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})

    kept_indices = get_kept_wavelength_indices(WVL_PRS, config)

    print(" Chargement des DataLoaders d'évaluation...")
    train_loader, val_loader, test_loader = create_data_loaders_spectral(
        train_dir=data_cfg.get("data_dir_train", ""),
        val_dir=data_cfg.get("data_dir_val", ""),
        test_dir=data_cfg.get("data_dir_test", ""),
        proportion_simulated=data_cfg.get("proportion_simulated", 0.0),
        augment=False,
        augment_illumination=False,
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=0,
        is_residual=data_cfg.get("is_residual", False),
        is_normalised=data_cfg.get("is_normalised", False),
        kept_indices=kept_indices,
        training=False
    )

    dataloaders = {}
    if train_loader is not None and len(train_loader) > 0:
        dataloaders["train"] = train_loader
    if val_loader is not None and len(val_loader) > 0:
        dataloaders["val"] = val_loader
    if test_loader is not None and len(test_loader) > 0:
        dataloaders["test"] = test_loader

    if not dataloaders:
        raise ValueError(" Aucun DataLoader disponible (train, val et test sont vides).")

    # Analyse du premier échantillon pour déterminer dynamiquement les canaux
    first_loader = next(iter(dataloaders.values()))
    sample = first_loader.dataset[0]

    sample_x = sample[0]
    sample_y = None

    for elem in sample[1:]:
        if isinstance(elem, (torch.Tensor, np.ndarray)) and hasattr(elem, "shape"):
            sample_y = elem
            break

    if sample_y is None:
        raise ValueError(" Impossible de trouver le cube de vérité terrain (y) dans le dataset.")

    n_msi, n_hsi = sample_x.shape[0], sample_y.shape[0]
    print(f" Canaux détectés : MSI = {n_msi} | HSI = {n_hsi}")

    return dataloaders, kept_indices, n_msi, n_hsi


def load_model_weights(model: torch.nn.Module, weights_path: str, device: torch.device):
    """Charge les poids d'un modèle pré-entraîné de manière sécurisée."""
    if not weights_path or not Path(weights_path).exists():
        raise FileNotFoundError(f" Poids introuvables : '{weights_path}'")

    print(f"📥 Chargement des poids depuis : {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================================
# BUILDERS DE MODÈLES
# ============================================================================

def build_model_reconstruction(config: dict, n_msi: int, n_hsi: int) -> torch.nn.Module:
    """Instancie le modèle de reconstruction spectrale selon la config."""
    model_cfg = config["model"]
    name = model_cfg.get("name", "").lower()

    if name == "gradualexpansionunet":
        return GradualExpansionUNet(
            in_msi=n_msi, in_hsi=n_hsi,
            interpolation_mode=model_cfg.get("interpolation_mode", "Bilinear"),
            base_channel=model_cfg.get("base_channel", 64),
            activation=model_cfg.get("activation", "silu"),
            with_batch_norm=model_cfg.get("with_batch_norm", True),
            with_mlp_spectral=model_cfg.get("with_mlp_spectral", False),
            final_activation=model_cfg.get("final_activation", None),
        )
    elif name == "gradualexpansionunet_res":
        return GradualExpansionUNet_residual(
            in_msi=n_msi, in_hsi=n_hsi,
            interpolation_mode=model_cfg.get("interpolation_mode", "Bilinear"),
            base_channel=model_cfg.get("base_channel", 64),
            activation=model_cfg.get("activation", "silu"),
            with_batch_norm=model_cfg.get("with_batch_norm", True),
            with_mlp_spectral=model_cfg.get("with_mlp_spectral", False),
            final_activation=model_cfg.get("final_activation", None),
        )
    else:
        raise ValueError(f" Modèle de reconstruction non reconnu : '{name}'")


def build_model_uncertainty(config: dict, n_msi: int, out_channels: int) -> torch.nn.Module:
    """Instancie le modèle d'estimation d'incertitude."""
    model_cfg = config["model_uncertainty"]
    name = model_cfg.get("name", "").lower()

    if name == "dualbranchnafnet":
        return DualBranchNAFNet(
            n_msi=n_msi,
            n_hsi=out_channels,
            out_channels=out_channels,
            width=model_cfg.get("base_channel", 64),
            middle_blk_num=model_cfg.get("middle_blk_num", 1),
            enc_blk_nums=model_cfg.get("enc_blk_nums", []),
            dec_blk_nums=model_cfg.get("dec_blk_nums", []),
            drop_out_rate=model_cfg.get("drop_out_rate", 0.0),
            final_op=model_cfg.get("final_activation", "softplus"),
        )
    else:
        raise ValueError(f" Modèle d'incertitude non reconnu : '{name}'")


# ============================================================================
# PIPELINES D'ÉVALUATION
# ============================================================================

def evaluate_pipeline_reconstruction(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    output_dir: Path | str,
    kept_indices: list = None,
    model_name: str = "GradualExpansionUNet",
    set_name: str = "Test",
):
    """Évalue quantitativement les performances de reconstruction et sauvegarde un CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Inférence {set_name}"):
            if len(batch) == 4:
                x_init, x_interp, y, patch_ids = batch
                x_interp = x_interp.to(device, non_blocking=True)
                has_interp = True
            elif len(batch) == 3:
                x_init, y, patch_ids = batch
                has_interp = False
            else:
                raise ValueError(f" Format de batch inattendu : {len(batch)} éléments.")

            x_init = x_init.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x_init, x_interp) if has_interp else model(x_init)

            y_np = y.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            for b in range(y.size(0)):
                p_id = patch_ids[b].item() if hasattr(patch_ids[b], "item") else patch_ids[b]
                gt_b, pred_b = y_np[b], pred_np[b]
                scene_name = extract_scene_name(p_id)

                metrics = {
                    "patch_id": p_id,
                    "scene": scene_name,
                    "psnr": float(compute_psnr(gt_b, pred_b)),
                    "ssim": float(compute_ssim_multiband(pred_b, gt_b)),
                    "sam": float(compute_sam(gt_b, pred_b)),
                    "ergas": float(compute_ergas(pred_b, gt_b)),
                    "mae": float(compute_mae(gt_b, pred_b)),
                }
                results.append(metrics)

    if not results:
        print(f" Aucun échantillon évalué pour le set {set_name}.")
        return

    df = pd.DataFrame(results)
    csv_path = output_dir / f"metrics_results_{set_name}.csv"
    df.to_csv(csv_path, index=False)

    summary_stats = df[["psnr", "ssim", "sam", "ergas", "mae"]].agg(["mean", "std", "min", "max"]).T
    summary_stats.columns = ["Moyenne", "Écart-type", "Minimum", "Maximum"]
    print(f"\n SYNTHÈSE DES MÉTRIQUES ({set_name.upper()})\n" + summary_stats.to_string(float_format=lambda x: f"{x:.4f}"))


def save_all_patch_ordered(
    loader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
    n_msi: int,
    n_hsi: int,
    kept_indices: list = None,
):
    """Sauvegarde les prédictions et incertitudes dans des fichiers memmap synchronisés."""
    eval_cfg = config.get("evaluation", {})
    use_amp = eval_cfg.get("use_amp", True)
    
    eval_dataset = loader.dataset
    total_patches = len(eval_dataset)
    if total_patches == 0:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = eval_dataset[0]
    sample_y = sample[2] if eval_dataset.is_residual else sample[1]
    _, patch_h, patch_w = sample_y.shape

    unc_weights_path = eval_cfg.get("weights_path_uncertainty")
    checkpoint_unc = torch.load(unc_weights_path, map_location=device, weights_only=True)
    state_dict_unc = checkpoint_unc.get("model_state_dict", checkpoint_unc.get("state_dict", checkpoint_unc))

    unc_out_channels = state_dict_unc["ending.weight"].shape[0] if "ending.weight" in state_dict_unc else n_hsi

    shape_rec = (total_patches, n_hsi, patch_h, patch_w)
    shape_unc = (total_patches, unc_out_channels, patch_h, patch_w)

    fp_rec = np.memmap(output_dir / "reconstruction.bin", dtype="float32", mode="w+", shape=shape_rec)
    fp_unc = np.memmap(output_dir / "uncertainty.bin", dtype="float32", mode="w+", shape=shape_unc)

    model_rec = build_model_reconstruction(config, n_msi, n_hsi).to(device)
    model_rec = load_model_weights(model_rec, eval_cfg.get("weights_path_reconstruction"), device)

    model_unc = build_model_uncertainty(config, n_msi, unc_out_channels).to(device)
    model_unc.load_state_dict(state_dict_unc)
    model_unc.eval()

    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Génération des memmaps"):
            if eval_dataset.is_residual:
                x_init, x_interp, y, patch_ids = batch
                x_interp = x_interp.to(device, non_blocking=True)
                has_interp = True
            else:
                x_init, y, patch_ids = batch
                has_interp = False

            x_init = x_init.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model_rec(x_init, x_interp) if has_interp else model_rec(x_init)
                u_hat = model_unc(torch.cat([x_init, pred], dim=1))

            pred_np = pred.detach().cpu().numpy()
            u_hat_np = u_hat.detach().cpu().numpy()

            for b in range(pred.shape[0]):
                fp_rec[global_idx] = pred_np[b]
                fp_unc[global_idx] = u_hat_np[b]
                global_idx += 1

    fp_rec.flush()
    fp_unc.flush()
    print(f"✅ Fichiers memmap enregistrés dans : {output_dir}")


def run_evaluation_spectral(config: dict, device: torch.device):
    """Lance l'évaluation de reconstruction pour tous les splits disponibles."""
    print("\n---  ÉVALUATION SPECTRALE ---")
    eval_cfg = config.get("evaluation", {})
    output_dir = Path(eval_cfg.get("output_dir", "./results_eval")) / "reconstruction"

    dataloaders, kept_indices, n_msi, n_hsi = setup_evaluation_context(config, device)
    model_rec = build_model_reconstruction(config, n_msi, n_hsi).to(device)
    model_rec = load_model_weights(model_rec, eval_cfg.get("weights_path_reconstruction"), device)

    for split_name, loader in dataloaders.items():
        evaluate_pipeline_reconstruction(
            model=model_rec,
            test_loader=loader,
            device=device,
            output_dir=output_dir / split_name,
            kept_indices=kept_indices,
            set_name=split_name.capitalize(),
        )


def run_evaluation_uncertainty(config: dict, device: torch.device):
    """Lance l'évaluation de l'incertitude couplée au modèle de reconstruction."""
    print("\n--- ÉVALUATION DE L'INCERTITUDE ---")
    eval_cfg = config.get("evaluation", {})
    output_dir = Path(eval_cfg.get("output_dir", "./results_eval")) / "uncertainty"
    use_amp = eval_cfg.get("use_amp", True)
    log_to_wandb = eval_cfg.get("log_to_wandb", False)

    dataloaders, kept_indices, n_msi, n_hsi = setup_evaluation_context(config, device)

    model_rec = build_model_reconstruction(config, n_msi, n_hsi).to(device)
    model_rec = load_model_weights(model_rec, eval_cfg.get("weights_path_reconstruction"), device)

    unc_weights_path = eval_cfg.get("weights_path_uncertainty")
    checkpoint_unc = torch.load(unc_weights_path, map_location=device, weights_only=True)
    state_dict_unc = checkpoint_unc.get("model_state_dict", checkpoint_unc.get("state_dict", checkpoint_unc))
    unc_out_channels = state_dict_unc["ending.weight"].shape[0] if "ending.weight" in state_dict_unc else n_hsi

    model_unc = build_model_uncertainty(config, n_msi, unc_out_channels).to(device)
    model_unc.load_state_dict(state_dict_unc)
    model_unc.eval()

    for split_name, loader in dataloaders.items():
        prefix = split_name.capitalize()
        plot_dir = output_dir / split_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Inférence incertitude {prefix}"):
                if len(batch) == 4:
                    x_init, x_interp, y, patch_ids = batch
                    x_interp_b = x_interp.to(device, non_blocking=True)
                    has_interp = True
                else:
                    x_init, y, patch_ids = batch
                    has_interp = False

                x_init_b = x_init.to(device, non_blocking=True)
                y_b = y.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred = model_rec(x_interp_b, x_interp_b) if has_interp else model_rec(x_init_b)
                    u_hat = model_unc(torch.cat([x_init_b, pred], dim=1))

                pred_np = pred.detach().cpu().numpy()
                y_np = y_b.detach().cpu().numpy()
                x_init_np = x_init_b.detach().cpu().numpy()
                u_hat_np = u_hat.detach().cpu().numpy()

                for idx in range(y.size(0)):
                    p_id = patch_ids[idx].item() if hasattr(patch_ids[idx], "item") else patch_ids[idx]
                    scene_id = extract_scene_name(p_id)

                    gt_hwc, pred_hwc, msi_hwc, unc_hwc = (
                        to_hwc(y_np[idx]), to_hwc(pred_np[idx]),
                        to_hwc(x_init_np[idx]), to_hwc(u_hat_np[idx])
                    )

                    mae = float(np.mean(np.abs(gt_hwc - pred_hwc)))
                    dot = np.sum(pred_hwc * gt_hwc, axis=-1)
                    norm_p, norm_g = np.linalg.norm(pred_hwc, axis=-1), np.linalg.norm(gt_hwc, axis=-1)
                    sam_rad = float(np.mean(np.arccos(np.clip(dot / (norm_p * norm_g + 1e-8), -1.0, 1.0))))

                    visualise_synthesis_uncertainty(
                        data={
                            "cube_gt": gt_hwc, "cube_predict": pred_hwc,
                            "cube_msi": msi_hwc, "cube_uncertainty": unc_hwc,
                            "model name": f"{prefix} — Scène {scene_id} (Patch {p_id})",
                            "img_mae": mae, "img_sam": sam_rad,
                        },
                        save_name=f"{prefix}_Scene_{scene_id}_Patch_{p_id}",
                        plot_dir=plot_dir,
                        kept_indices=kept_indices,
                        log_to_wandb=log_to_wandb,
                    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'évaluation globale MSI-HSI")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de config YAML")
    parser.add_argument("--mode", type=str, choices=["spectral", "uncertainty", "all"], default="spectral", help="Type d'évaluation à exécuter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda ou cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dev = torch.device(args.device)

    if args.mode in ["spectral", "all"]:
        run_evaluation_spectral(cfg, dev)
    
    if args.mode in ["uncertainty", "all"]:
        run_evaluation_uncertainty(cfg, dev)