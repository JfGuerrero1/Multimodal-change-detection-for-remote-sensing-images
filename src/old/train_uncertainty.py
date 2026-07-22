import argparse
import torch
import torch.optim as optim

from models import UNet, NAFNet, DualBranchUNet, DualBranchNAFNet
from old.utils_dataset import create_dataloaders, UncertaintyDataset
from tqdm import tqdm

def build_model(args):

    if args.model == "unet":
        model = UNet(
            in_channels=242,
            activation=args.activation,
            interpolation_mode=args.interpolation_mode,
            learning_mode="standard",
            drop_out_rate=args.drop_out_rate
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
            final_op = args.final_op,
            drop_out_rate=args.drop_out_rate
        )

    else:
        raise ValueError("Unknown model")

    return model

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0

    pbar = tqdm(loader, desc="Training")
    
    for x, y in pbar:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)
        scale = 100.0
        loss = criterion(scale*pred, scale*y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        
        pbar.set_postfix(Loss=f"{loss.item():.4f}")

    n = len(loader)

    return total_loss / n

@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    pbar = tqdm(loader, desc="Validation")
    for x, y in pbar:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred, y)

        total_loss += loss.item()
        
        pbar.set_postfix(Loss=f"{loss.item():.4f}")

    n = len(loader)

    return total_loss / n

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--activation",
                        type=str,
                        default="silu",
                        choices=["relu", "leakyrelu", "silu"])

    parser.add_argument("--interpolation_mode",
                        type=str,
                        default="Bilinear",
                        choices=["ConvTranspose2d", "Bilinear"])

    parser.add_argument("--lr",
                        type=float,
                        default=1e-4)

    parser.add_argument("--epochs",
                        type=int,
                        default=100)

    parser.add_argument("--batch_size",
                        type=int,
                        default=8)
    
    parser.add_argument("--num_workers",
                        type=int,
                        default=4)
    
    parser.add_argument("--load_model",
                        type=str,
                        default=None)
    
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
    
    parser.add_argument("--warm_up",
                        type=bool,
                        default=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)
    
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Model loaded from {args.load_model}")

    criterion = torch.nn.L1Loss() #torch.nn.functional.mse_loss

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.warm_up:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=3) 

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_class=UncertaintyDataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss = validate(
            model, val_loader, criterion, device
        )

        if args.warm_up:
            scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss={train_loss:.6f}")
        print(f"Val Loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_uncertainty.pth")

    print("Training finished.")


if __name__ == "__main__":
    main()
