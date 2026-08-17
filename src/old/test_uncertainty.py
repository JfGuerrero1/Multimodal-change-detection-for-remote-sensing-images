import os

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import UNet, NAFNet, DualBranchUNet, DualBranchNAFNet
from old.utils_dataset import create_dataloaders, Spectraldataset
from tqdm import tqdm

def build_model(args):

    if args.model == "unet":
        model = UNet(
            in_channels=242,
            activation=args.activation,
            interpolation_mode=args.interpolation_mode,
            learning_mode="standard"
        )

    elif args.model == "nafnet":
        model = NAFNet(
            in_channels=242, 
            out_channels=230, 
            width=args.width,          
            enc_blk_nums=args.enc_blk_nums,
            middle_blk_num=args.middle_blk_num,   
            dec_blk_nums=args.dec_blk_nums,
            drop_out_rate=args.drop_out_rate
        )

    elif args.model == 'dualbranchunet':
        model = DualBranchUNet(
            n_msi = 12, 
            n_hsi = 230, 
            base_features=64,
            interpolation_mode=args.interpolation_mode,
            activation=args.activation,
            drop_out_rate=args.drop_out_rate,
            final_op = args.final_op
        )

    elif args.model == "dualbranchnafnet":
        model = DualBranchNAFNet(
            n_msi = 12, 
            n_hsi = 230, 
            out_channels=230, 
            width=args.width,          
            enc_blk_nums=args.enc_blk_nums,
            middle_blk_num=args.middle_blk_num,   
            dec_blk_nums=args.dec_blk_nums,
            drop_out_rate=args.drop_out_rate,
            final_op = args.final_op
        )

    else:
        raise ValueError("Unknown model")

    return model

@torch.no_grad()
def evaluate_per_channel(model, loader, device, num_channels=230):
    model.eval()

    total_mse_per_channel = torch.zeros(num_channels, device=device)
    total_mae_per_channel = torch.zeros(num_channels, device=device)

    pbar = tqdm(loader, desc="Testing Uncertainty per Channel")
    
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        pred_uncertainty = model(x)

        # [B, 230, H, W]
        mse_raw = F.mse_loss(pred_uncertainty, y, reduction='none')
        mae_raw = F.l1_loss(pred_uncertainty, y, reduction='none')

        mse_per_channel = mse_raw.mean(dim=(0, 2, 3))
        mae_per_channel = mae_raw.mean(dim=(0, 2, 3))

        total_mse_per_channel += mse_per_channel
        total_mae_per_channel += mae_per_channel
        
        pbar.set_postfix(Global_MAE=f"{mae_per_channel.mean().item():.4f}")

    n = len(loader)

    final_mse = (total_mse_per_channel / n).cpu().numpy()
    final_mae = (total_mae_per_channel / n).cpu().numpy()

    return final_mse, final_mae

def save_errors(mae, mse, split_name, save_dir="data"):

    os.makedirs(save_dir, exist_ok=True)

    mae_path = f"{save_dir}/{split_name}_mae_per_channel.npy"
    mse_path = f"{save_dir}/{split_name}_mse_per_channel.npy"

    np.save(mae_path, mae)
    np.save(mse_path, mse)

    print(f"Saved {split_name} MAE in {mae_path}")
    print(f"Saved {split_name} MSE in {mse_path}")

def save_errors_with_mc(mse, mae, var, mean_pred, true_target, split_name, save_dir="data"):

    os.makedirs(save_dir, exist_ok=True)

    mse_path = f"{save_dir}/{split_name}_mse_per_channel.npy"
    mae_path = f"{save_dir}/{split_name}_mae_per_channel.npy"
    var_path = f"{save_dir}/{split_name}_var_per_channel.npy"
    mean_pred_path = f"{save_dir}/{split_name}_mean_pred_per_channel.npy"
    true_target_path = f"{save_dir}/{split_name}_true_target_per_channel.npy"

    np.save(mse_path, mse)
    np.save(mae_path, mae)
    np.save(var_path, var)
    np.save(mean_pred_path, mean_pred)
    np.save(true_target_path, true_target)

    print(f"Saved {split_name} MSE in {mse_path}")
    print(f"Saved {split_name} MAE in {mae_path}")
    print(f"Saved {split_name} Variance in {var_path}")
    print(f"Saved {split_name} Mean Prediction in {mean_pred_path}")
    print(f"Saved {split_name} True Target in {true_target_path}")

def plot_and_save_errors(mae_per_channel, mse_per_channel, split_name, save_dir="data"):

    os.makedirs(save_dir, exist_ok=True)

    channels = np.arange(len(mae_per_channel))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(channels, mae_per_channel, label='MAE', linewidth=2)
    plt.plot(channels, mse_per_channel, label='MSE', linestyle='--', alpha=0.7)
    
    plt.title(f"{split_name.capitalize()} Error Per Channel", fontsize=14)
    plt.xlabel("Channel", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()

    plot_path = f"{save_dir}/{split_name}_error_per_channel.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n Error plot saved in: {plot_path}")

def plot_confidence_interval(true_error, predicted_mean_error, var, split_name, save_dir="data"):

    os.makedirs(save_dir, exist_ok=True)

    channels = np.arange(len(true_error))

    std = np.sqrt(var)
    ci = 1.96 * std

    lower = np.maximum(0, predicted_mean_error - ci)
    upper = predicted_mean_error + ci

    plt.figure(figsize=(12, 6))

    plt.plot(channels, true_error, label='True error', linewidth=2)
    plt.plot(channels, predicted_mean_error, '--', label='Predicted mean error', linewidth=2)
    plt.fill_between(channels, lower, upper, alpha=0.2, label='95% CI')

    plt.title(f"{split_name.capitalize()} - Uncertainty per channel")
    plt.xlabel("Spectral channel")
    plt.ylabel("Error")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    path = f"{save_dir}/{split_name}_confidence_plot.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")

def plot_and_save_performance(mae_per_channel, mse_per_channel, split_name, save_dir="data"):

    os.makedirs(save_dir, exist_ok=True)
    channels = np.arange(len(mae_per_channel))
    
    plt.figure(figsize=(12, 6))
    plt.plot(channels, mae_per_channel, label='MAE', color='red', linewidth=2)
    
    plt.title(f"{split_name.capitalize()} MAE Per Channel", fontsize=14)
    plt.xlabel("Channel", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    mae_path = f"{save_dir}/{split_name}_mae_per_channel.png"
    plt.savefig(mae_path, dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(channels, mse_per_channel, label='MSE', color='orange', linestyle='--', linewidth=2)
    
    plt.title(f"{split_name.capitalize()} MSE Per Channel", fontsize=14)
    plt.xlabel("Channel", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    mse_path = f"{save_dir}/{split_name}_mse_per_channel.png"
    plt.savefig(mse_path, dpi=300)
    plt.close()

    print(f"MAE plot saved in: {mae_path}")
    print(f"MSE plot saved in: {mse_path}")

def enable_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

@torch.no_grad()
def evaluate_per_channel_mc(model, loader, device, num_channels=230, T=25):
    model.eval()
    enable_dropout(model)

    total_mse_per_channel = torch.zeros(num_channels, device=device)
    total_mae_per_channel = torch.zeros(num_channels, device=device)
    total_var_per_channel = torch.zeros(num_channels, device=device)
    total_mean_pred_per_channel = torch.zeros(num_channels, device=device) 
    total_true_target_per_channel = torch.zeros(num_channels, device=device)

    pbar = tqdm(loader, desc="MC Dropout Evaluation")

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        preds = []

        for _ in range(T):
            pred = model(x)
            preds.append(pred.unsqueeze(0))

        preds = torch.cat(preds, dim=0)

        mean_pred = preds.mean(dim=0)
        var_pred = preds.var(dim=0)

        mse_raw = F.mse_loss(mean_pred, y, reduction='none')
        mae_raw = F.l1_loss(mean_pred, y, reduction='none')

        mse_per_channel = mse_raw.mean(dim=(0, 2, 3))
        mae_per_channel = mae_raw.mean(dim=(0, 2, 3))
        var_per_channel = var_pred.mean(dim=(0, 2, 3))
        mean_val_per_channel = mean_pred.mean(dim=(0, 2, 3)) 
        true_val_per_channel = y.mean(dim=(0, 2, 3))

        total_mse_per_channel += mse_per_channel
        total_mae_per_channel += mae_per_channel
        total_var_per_channel += var_per_channel
        total_mean_pred_per_channel += mean_val_per_channel  
        total_true_target_per_channel += true_val_per_channel

        pbar.set_postfix(
            MAE=f"{mae_per_channel.mean().item():.4f}",
            UNC=f"{var_per_channel.mean().item():.4f}"
        )

    n = len(loader)

    return (
        (total_mse_per_channel / n).cpu().numpy(),
        (total_mae_per_channel / n).cpu().numpy(),
        (total_var_per_channel / n).cpu().numpy(),
        (total_mean_pred_per_channel / n).cpu().numpy(),
        (total_true_target_per_channel / n).cpu().numpy()
    )

def main():
    parser = argparse.ArgumentParser(description="Test Uncertainty Prediction Model")

    parser.add_argument("--activation",
                        type=str,
                        default="silu",
                        choices=["relu", "leakyrelu", "silu"])

    parser.add_argument("--interpolation_mode",
                        type=str,
                        default="Bilinear",
                        choices=["ConvTranspose2d", "Bilinear"])

    parser.add_argument("--batch_size",
                        type=int,
                        default=8)
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=4)
    
    parser.add_argument("--load_model",
                        type=str,
                        required=True, 
                        help="Path to the trained uncertainty model (.pth)")
    
    parser.add_argument("--model",
                        type=str,
                        default="unet",
                        choices=["unet", "nafnet", "dualbranchunet", "dualbranchnafnet"])
    
    parser.add_argument("--width",
                        type=int,
                        default=64)
    
    parser.add_argument("--enc_blk_nums",
                        type=int,
                        nargs="+",
                        default=[1, 2, 4, 8])
    
    parser.add_argument("--middle_blk_num",
                        type=int,
                        default=12)
    
    parser.add_argument("--dec_blk_nums",
                        type=int,
                        nargs="+",
                        default=[1, 1, 1, 1])
    
    parser.add_argument("--final_op",
                        type=str,
                        default='identity',
                        choices=["identity", "abs", "softplus", "square"]
                        )
    parser.add_argument("--drop_out_rate",
                        type=float,
                        default=0.)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)
    
    try:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Model loaded from  {args.load_model}")
    except Exception as e:
        print(f"Couldn't load model: {e}")
        return

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_class=SpectralDataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    for split_name, loader in [
        ("train", train_loader),
        #("val", val_loader),
        ("test", test_loader)
    ]:

        print("\n" + "="*50)
        print(f"Evaluating {split_name.upper()}")

        #mse_ch, mae_ch = evaluate_per_channel(model, loader, device)

        #print(f"{split_name} MSE : {mse_ch.mean():.6f}")
        #print(f"{split_name} MAE : {mae_ch.mean():.6f}")
#
        #print(f"Worst MAE channel : {np.argmax(mae_ch)} ({np.max(mae_ch):.6f})")
        #print(f"Best MAE channel  : {np.argmin(mae_ch)} ({np.min(mae_ch):.6f})")
        #print(f"Worst MSE channel : {np.argmax(mse_ch)} ({np.max(mse_ch):.6f})")
        #print(f"Best MSE channel  : {np.argmin(mse_ch)} ({np.min(mse_ch):.6f})")

        #save_errors(mae_ch, mse_ch, split_name)
        #plot_and_save_errors(mae_ch, mse_ch, split_name)

        if split_name == "train":
            subset_size = len(loader.dataset) // 8
            
            indices = np.random.choice(len(loader.dataset), subset_size, replace=False)
            small_train_ds = torch.utils.data.Subset(loader.dataset, indices)
            loader = torch.utils.data.DataLoader(
                small_train_ds, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                shuffle=False 
            )

        mse_ch, mae_ch, var_ch, mean_pred_ch, true_target_ch = evaluate_per_channel_mc(model, loader, device, T=10)

        save_errors_with_mc(mse_ch, mae_ch, var_ch, mean_pred_ch, true_target_ch, split_name)
        
        plot_confidence_interval(true_target_ch, mean_pred_ch, var_ch, split_name)
        plot_and_save_performance(mae_ch, mse_ch, split_name)
if __name__ == "__main__":
    main(