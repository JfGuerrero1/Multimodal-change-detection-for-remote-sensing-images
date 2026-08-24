import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Racine du projet
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.constants import WVL_PRS
from src.models.dual_branch import DualBranchNAFNet
from src.models.unet_residual import GradualExpansionUNet_residual
from src.models.unet_standard import GradualExpansionUNet
from src.old.metrics_and_loss.metrics import (
    compute_ergas,
    compute_mae,
    compute_psnr,
    compute_sam,
    compute_ssim_multiband,
)
from src.prepare_data.prepare_patch import create_data_loaders_spectral
from src.training.utils_train import get_kept_wavelength_indices
from src.visualise.visualisation import visualise_synthesis
from src.visualise.visualisation_spectre import (
    supervise_analyse_spectrale,
    trace_spectre,
    visualise_curve,
)
from src.visualise.visualise_uncertainty import (
    visualise_synthesis_uncertainty,
)

# --- UTILS ---


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_scene_name(patch_id: str | int) -> str:
    p_str = str(patch_id)
    scene = re.sub(r"(_patch_.*|_[\d]+)$", "", p_str)
    return scene if scene else "scene_inconnue"


def to_hwc(arr: np.ndarray) -> np.ndarray:
    return np.moveaxis(arr, 0, -1) if arr.shape[0] < arr.shape[-1] else arr


def setup_evaluation_context(config: dict, device: torch.device):
    eval_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})

    kept_indices = get_kept_wavelength_indices(WVL_PRS, config)

    print("📦 Chargement des DataLoaders...")
    _, val_loader, test_loader = create_data_loaders_spectral(
        train_dir=data_cfg.get("data_dir_train", ""),
        val_dir=data_cfg.get("data_dir_val", ""),
        test_dir=data_cfg.get("data_dir_test", ""),
        simulated=data_cfg.get("simulated", True),
        augment=False,
        augment_illumination=False,
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=0,
        is_residual=data_cfg.get("is_residual", False),
        is_normalised=data_cfg.get("is_normalised", False),
        kept_indices=kept_indices,
    )

    dataloaders = {}
    if val_loader is not None and len(val_loader) > 0:
        dataloaders["val"] = val_loader
    if test_loader is not None and len(test_loader) > 0:
        dataloaders["test"] = test_loader

    if not dataloaders:
        raise ValueError("❌ Aucun DataLoader disponible (val ou test vide).")

    first_loader = next(iter(dataloaders.values()))
    sample = first_loader.dataset[0]

    # Extraction sécurisée de x_init et y (indépendant de l'ordre str / Tensor)
    sample_x = sample[0]
    sample_y = None

    for elem in sample[1:]:
        if isinstance(elem, (torch.Tensor, np.ndarray)) and hasattr(
            elem, "shape"
        ):
            sample_y = elem

    if sample_y is None:
        raise ValueError(
            "❌ Impossible de trouver le cube de vérité terrain (y) dans le dataset."
        )

    n_msi, n_hsi = sample_x.shape[0], sample_y.shape[0]
    print(f"📐 Canaux détectés : MSI = {n_msi} | HSI = {n_hsi}")

    return dataloaders, kept_indices, n_msi, n_hsi


def load_model_weights(
    model: torch.nn.Module, weights_path: str, device: torch.device
):
    if not weights_path or not Path(weights_path).exists():
        raise FileNotFoundError(f"❌ Poids introuvables : '{weights_path}'")

    print(f"📥 Chargement des poids depuis : {weights_path}")
    checkpoint = torch.load(
        weights_path, map_location=device, weights_only=True
    )
    state_dict = checkpoint.get(
        "model_state_dict", checkpoint.get("state_dict", checkpoint)
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


# --- BUILDERS DE MODÈLES ---


def build_model_reconstruction(
    config: dict, n_msi: int, n_hsi: int
) -> torch.nn.Module:
    model_cfg = config["model"]
    name = model_cfg.get("name", "").lower()

    if name == "gradualexpansionunet":
        return GradualExpansionUNet(
            in_msi=n_msi,
            in_hsi=n_hsi,
            interpolation_mode=model_cfg.get("interpolation_mode", "Bilinear"),
            base_channel=model_cfg.get("base_channel", 64),
            activation=model_cfg.get("activation", "silu"),
            with_batch_norm=model_cfg.get("with_batch_norm", True),
            with_mlp_spectral=model_cfg.get("with_mlp_spectral", False),
            final_activation=model_cfg.get("final_activation", None),
        )
    elif name == "gradualexpansionunet_res":
        return GradualExpansionUNet_residual(
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


def build_model_uncertainty(
    config: dict, n_msi: int, out_channels: int
) -> torch.nn.Module:
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
        raise ValueError(f"❌ Modèle d'incertitude non reconnu : '{name}'")


# --- ÉVALUATIONS ---


def evaluate_pipeline_reconstruction(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    output_dir: Path | str,
    kept_indices: list = None,
    model_name: str = "GradualExpansionUNet",
    set_name: str = "Test",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    results, cache_data = [], {}

    n_pixels_total = 0
    sum_error, sum_sq_error, max_error = None, None, None

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(test_loader, desc=f"Inférence {set_name}")
        ):
            if len(batch) == 4:
                x_init, x_interp, y, patch_ids = batch
                x_interp = x_interp.to(device, non_blocking=True)
                has_interp = True
            elif len(batch) == 3:
                x_init, y, patch_ids = batch
                has_interp = False
            else:
                raise ValueError(
                    f"❌ Format de batch inattendu : {len(batch)} éléments reçus."
                )

            x_init = x_init.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x_init, x_interp) if has_interp else model(x_init)

            y_np = y.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            x_init_np = x_init.detach().cpu().numpy()

            for b in range(y.size(0)):
                p_id = (
                    patch_ids[b].item()
                    if hasattr(patch_ids[b], "item")
                    else patch_ids[b]
                )

                gt_b, pred_b, msi_b = y_np[b], pred_np[b], x_init_np[b]

                abs_err_chw = np.abs(gt_b - pred_b)
                num_pixels_patch = gt_b.shape[1] * gt_b.shape[2]

                if sum_error is None:
                    sum_error = np.sum(abs_err_chw, axis=(1, 2))
                    sum_sq_error = np.sum(abs_err_chw**2, axis=(1, 2))
                    max_error = np.max(abs_err_chw, axis=(1, 2))
                else:
                    sum_error += np.sum(abs_err_chw, axis=(1, 2))
                    sum_sq_error += np.sum(abs_err_chw**2, axis=(1, 2))
                    max_error = np.maximum(
                        max_error, np.max(abs_err_chw, axis=(1, 2))
                    )

                n_pixels_total += num_pixels_patch

                gt_hwc, pred_hwc, msi_hwc = (
                    to_hwc(gt_b),
                    to_hwc(pred_b),
                    to_hwc(msi_b),
                )
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
                cache_data[p_id] = {
                    "cube_gt": gt_hwc,
                    "cube_predict": pred_hwc,
                    "cube_msi": msi_hwc,
                    "metrics": metrics,
                }

    if not results:
        print(f"⚠️ Aucun échantillon dans le DataLoader {set_name}.")
        return

    df = pd.DataFrame(results)
    csv_path = output_dir / f"metrics_results_{set_name}.csv"
    df.to_csv(csv_path, index=False)

    # Statistiques
    summary_stats = (
        df[["psnr", "ssim", "sam", "ergas", "mae"]]
        .agg(["mean", "std", "min", "max"])
        .T
    )
    summary_stats.columns = [
        "Moyenne",
        "Écart-type (Std)",
        "Minimum",
        "Maximum",
    ]
    print(
        f"\n📊 TABLEAU RÉCAPITULATIF DES MÉTRIQUES (SET: {set_name.upper()})\n"
        + summary_stats.to_string(float_format=lambda x: f"{x:.4f}")
    )

    # Courbe spectrale globale
    if n_pixels_total > 0:
        mean_error_glob = sum_error / n_pixels_total
        mean_sq_error_glob = sum_sq_error / n_pixels_total
        variance_glob = np.maximum(
            0.0, mean_sq_error_glob - (mean_error_glob**2)
        )

        visualise_curve(
            mean_error=mean_error_glob,
            std_error=np.sqrt(variance_glob),
            max_error=max_error,
            name_curve=output_dir / f"Analyse_Spectrale_Globale_{set_name}.png",
            kept_indices=kept_indices,
            wvl_prs=WVL_PRS,
        )

    # Plots ciblés
    target_indices = set()
    np.random.seed(42)
    for _, group in df.groupby("scene"):
        scene_patches = group["patch_id"].unique()
        chosen = np.random.choice(
            scene_patches, min(3, len(scene_patches)), replace=False
        )
        target_indices.update(chosen)

    plot_sub_dir = output_dir / "targeted_patches"
    for p_id in tqdm(target_indices, desc="Plots Ciblés"):
        if p_id not in cache_data:
            continue
        p_data = cache_data[p_id]
        m = p_data["metrics"]

        data_dict = {
            "cube_gt": p_data["cube_gt"],
            "cube_predict": p_data["cube_predict"],
            "cube_msi": p_data["cube_msi"],
            "model_name": model_name,
            "img_psnr": m["psnr"],
            "img_ssim": m["ssim"],
            "img_sam": m["sam"],
            "img_ergas": m["ergas"],
            "img_mae": m["mae"],
        }
        safe_p_id_str = str(p_id).replace("/", "_")

        visualise_synthesis(
            data_dict,
            f"patch_{safe_p_id_str}",
            plot_sub_dir,
            kept_indices=kept_indices,
        )
        supervise_analyse_spectrale(
            data_dict,
            f"patch_{safe_p_id_str}",
            plot_sub_dir,
            kept_indices=kept_indices,
        )

        H, W, _ = p_data["cube_gt"].shape
        np.random.seed(42)
        trace_spectre(
            patch=p_data["cube_predict"],
            patch_gt=p_data["cube_gt"],
            y_indices=np.random.randint(0, H, 5),
            x_indices=np.random.randint(0, W, 5),
            scene=f"patch_{safe_p_id_str}",
            output_dir_diag=plot_sub_dir,
            kept_indices=kept_indices,
        )


def run_evaluation_spectral(config: dict, device: torch.device):
    print("\n--- 🚀 RUN EVALUATION SPECTRALE ---")
    eval_cfg = config.get("evaluation", {})
    output_dir = (
        Path(eval_cfg.get("output_dir", "./results_eval")) / "reconstruction"
    )

    dataloaders, kept_indices, n_msi, n_hsi = setup_evaluation_context(
        config, device
    )

    model_rec = build_model_reconstruction(config, n_msi, n_hsi).to(device)
    model_rec = load_model_weights(
        model_rec, eval_cfg.get("weights_path_reconstruction"), device
    )

    for split_name, loader in dataloaders.items():
        evaluate_pipeline_reconstruction(
            model=model_rec,
            test_loader=loader,
            device=device,
            output_dir=output_dir / split_name,
            kept_indices=kept_indices,
            model_name=eval_cfg.get("model_name", "GradualExpansionUNet"),
            set_name=split_name.capitalize(),
        )


def run_evaluation_uncertainty(config: dict, device: torch.device):
    print("\n--- 🚀 RUN EVALUATION INCERTITUDE ---")
    eval_cfg = config.get("evaluation", {})
    output_dir = (
        Path(eval_cfg.get("output_dir", "./results_eval")) / "uncertainty"
    )
    use_amp = eval_cfg.get("use_amp", True)
    log_to_wandb = eval_cfg.get("log_to_wandb", False)

    dataloaders, kept_indices, n_msi, n_hsi = setup_evaluation_context(
        config, device
    )

    # 1. Chargement modèle de reconstruction
    model_rec = build_model_reconstruction(config, n_msi, n_hsi).to(device)
    model_rec = load_model_weights(
        model_rec, eval_cfg.get("weights_path_reconstruction"), device
    )

    # 2. Détection dynamique des dimensions de sortie de l'incertitude depuis le checkpoint
    unc_weights_path = eval_cfg.get("weights_path_uncertainty")
    if not unc_weights_path or not Path(unc_weights_path).exists():
        raise FileNotFoundError(
            f"❌ Poids du modèle d'incertitude introuvables : '{unc_weights_path}'"
        )

    print(f"📥 Inspection des poids d'incertitude depuis : {unc_weights_path}")
    checkpoint_unc = torch.load(
        unc_weights_path, map_location=device, weights_only=True
    )
    state_dict_unc = checkpoint_unc.get(
        "model_state_dict", checkpoint_unc.get("state_dict", checkpoint_unc)
    )

    if "ending.weight" in state_dict_unc:
        unc_out_channels = state_dict_unc["ending.weight"].shape[0]
        print(
            f"🔍 Détection dynamique : le modèle d'incertitude attend {unc_out_channels} canaux en sortie."
        )
    else:
        unc_out_channels = config.get("model_uncertainty", {}).get(
            "out_channels", n_hsi
        )

    # 3. Instanciation du modèle d'incertitude aux bonnes dimensions
    model_unc = build_model_uncertainty(config, n_msi, unc_out_channels).to(
        device
    )
    model_unc.load_state_dict(state_dict_unc)
    model_unc.eval()

    # 4. Inférence
    for split_name, loader in dataloaders.items():
        prefix = split_name.capitalize()
        plot_dir = output_dir / split_name
        plot_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Inférence {prefix}"):
                if len(batch) == 4:
                    x_init, x_interp, y, patch_ids = batch
                    x_interp_b = x_interp.to(device, non_blocking=True)
                    has_interp = True
                else:
                    x_init, y, patch_ids = batch
                    has_interp = False

                x_init_b = x_init.to(device, non_blocking=True)
                y_b = y.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type, enabled=use_amp
                ):
                    pred = (
                        model_rec(x_init_b, x_interp_b)
                        if has_interp
                        else model_rec(x_init_b)
                    )
                    unc_input = torch.cat([x_init_b, pred], dim=1)
                    u_hat = model_unc(unc_input)

                pred_np = pred.detach().cpu().numpy()
                y_np = y_b.detach().cpu().numpy()
                x_init_np = x_init_b.detach().cpu().numpy()
                u_hat_np = u_hat.detach().cpu().numpy()

                for idx in range(y.size(0)):
                    raw_id = patch_ids[idx]
                    p_id = raw_id.item() if hasattr(raw_id, "item") else raw_id
                    scene_id = extract_scene_name(p_id)

                    gt_hwc, pred_hwc, msi_hwc, unc_hwc = (
                        to_hwc(y_np[idx]),
                        to_hwc(pred_np[idx]),
                        to_hwc(x_init_np[idx]),
                        to_hwc(u_hat_np[idx]),
                    )

                    mae = float(np.mean(np.abs(gt_hwc - pred_hwc)))
                    dot = np.sum(pred_hwc * gt_hwc, axis=-1)
                    norm_p, norm_g = np.linalg.norm(
                        pred_hwc, axis=-1
                    ), np.linalg.norm(gt_hwc, axis=-1)
                    sam_rad = float(
                        np.mean(
                            np.arccos(
                                np.clip(
                                    dot / (norm_p * norm_g + 1e-8), -1.0, 1.0
                                )
                            )
                        )
                    )

                    visualise_synthesis_uncertainty(
                        data={
                            "cube_gt": gt_hwc,
                            "cube_predict": pred_hwc,
                            "cube_msi": msi_hwc,
                            "cube_uncertainty": unc_hwc,
                            "model name": f"{prefix} — Scène {scene_id} (Patch {p_id})",
                            "img_mae": mae,
                            "img_sam": sam_rad,
                        },
                        save_name=f"{prefix}_Scene_{scene_id}_Patch_{p_id}",
                        plot_dir=plot_dir,
                        kept_indices=kept_indices,
                        log_to_wandb=log_to_wandb,
                    )


# --- MAIN ---


def main():
    parser = argparse.ArgumentParser(description="Script d'évaluation globale")
    parser.add_argument(
        "--config",
        type=str,
        default="evaluate_config.yaml",
        help="Fichier de configuration",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["spectral", "uncertainty", "all"],
        default="all",
        help="Type d'évaluation",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device de calcul : {device}")

    if args.mode in ["spectral", "all"]:
        run_evaluation_spectral(config, device)

    if args.mode in ["uncertainty", "all"]:
        run_evaluation_uncertainty(config, device)


if __name__ == "__main__":
    main()