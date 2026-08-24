
import argparse
import torch
import torch.optim as optim

from models import UNet, GradualExpansionUNet,GradualExpansionUNet_residual
from src.old.metrics_and_loss.loss import SpectralLoss
from old.utils_dataset import create_data_loaders_spectral, SpectralDataset
from tqdm import tqdm
import wandb
import torch.optim.lr_scheduler as lr_scheduler

def build_model(args,n_msi,n_hsi):

    if args.model == "gradualexpansionunet_res":
        model = GradualExpansionUNet_residual(
            in_msi=n_msi,         
            in_hsi=n_hsi,
            interpolation_mode=args.interpolation_mode,
            activation=args.activation,
            augment=args.augment,
            with_mlp_spectral=args.with_mlp_spectral
          )
     

    else:
        raise ValueError("Unknown model")

    return model

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0
    total_mse = 0
    total_sam = 0
    total_mae= 0
    total_grad_norm=0

    pbar = tqdm(loader, desc="Training")
    
    # MSI_real, HSI_interp,HSI_real


    for x_init, x_interp, y in pbar:

        x_init = x_init.to(device)
        x_interp=x_interp.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x_init, x_interp)

        loss, mse,mae, sam = criterion(pred, y)

    
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_grad_norm += grad_norm.item()

        total_loss += loss.item()
        total_mse += mse.item()
        total_sam += sam.item()
        total_mae+= mae.item()
        
        pbar.set_postfix(Loss=f"{loss.item():.4f}",MAE=f"{mae.item()}", MSE=f"{mse.item():.4f}", SAM=f"{sam.item():.4f}")

    n = len(loader)

    return total_loss / n, total_mse / n ,total_mae/n, total_sam / n,total_grad_norm/n


@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    total_mse = 0
    total_sam = 0
    total_mae=0


    pbar = tqdm(loader, desc="Validation")
    for x_init, x_interp, y in pbar:

        x_init = x_init.to(device)
        x_interp=x_interp.to(device)
        y = y.to(device)

        pred = model(x_init,x_interp)

        loss, mse,mae, sam = criterion(pred, y)

        total_loss += loss.item()
        total_mse += mse.item()
        total_sam += sam.item()
        total_mae+=mae.item()
        
        pbar.set_postfix(Loss=f"{loss.item():.4f}",MAE=f"{mae.item()}", MSE=f"{mse.item():.4f}", SAM=f"{sam.item():.4f}")

    n = len(loader)

    return total_loss / n, total_mse / n, total_mae/n, total_sam / n


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model",type=str,default="gradualexpansionunet_res",
                        choices=["gradualexpansionunet_res"])

    parser.add_argument("--augment",action="store_true",
                        help="Activer l'augmentation de données géométriques sur le train")
    
    parser.add_argument("--simulated",action="store_true",
                        help="Activer données simulées sur la MSI")

    parser.add_argument("--activation",type=str,default="silu",
                        choices=["relu", "leakyrelu", "silu"])

    parser.add_argument("--interpolation_mode",type=str,default="Bilinear",
                        choices=["ConvTranspose2d", "Bilinear"])

    parser.add_argument("--lambda_sam",type=float,default=0.1)
    parser.add_argument("--lambda_mse",type=float,default=1.0)
    
    parser.add_argument("--lambda_mae",type=float,default=0.0 )

    parser.add_argument("--lr",type=float,default=1e-3)

    parser.add_argument("--epochs",type=int,default=100)

    parser.add_argument("--batch_size",type=int,default=8)
    
    parser.add_argument("--num_workers",type=int,default=4)
    
    parser.add_argument("--load_model",type=str,default=None)
    
    parser.set_defaults(is_residual=True) # Pas toucher 
    parser.add_argument('--no_keep_atm_wave', dest='keep_atm_wave', action='store_false', help="Supprime les longueurs d'onde atmosphériques")
    parser.set_defaults(keep_atm_wave=True)
    parser.set_defaults(with_mlp_spectral=False)
    parser.add_argument('--mlp',dest='with_mlp_spectral',action='store_true') 
    parser.set_defaults(is_normalised=False)
    parser.add_argument('--is_normalised',dest='is_normalised',action='store_true')

    parser.add_argument('--output_dir',type=str, default="checkpoints")
    args = parser.parse_args()

    
    train_loader, val_loader, test_loader = create_data_loaders_spectral(
        use_simulated_msi=args.simulated,
        augment=args.augment, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_residual=args.is_residual,
        keep_atm_wave=args.keep_atm_wave,
        is_normalised=args.is_normalised

    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Vérification rapide
    sample_x,sample_x_interp, sample_y = next(iter(train_loader))
    print(f"Bandes spectrales en entrée : {sample_x.shape[1]}")
    print(f"Bandes spectrales en sortie : {sample_y.shape[1]}")
    print(f"Bandes spectrales interp en sortie : {sample_x_interp.shape[1]}")

    n_msi = sample_x.shape[1]
    n_hsi = sample_y.shape[1]

    
    
    model = build_model(args,n_msi,n_hsi).to(device)
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Model loaded from {args.load_model}")
    
    """
    dummy_input = torch.randn(1, 12, 256, 256).to(device)
    dummy_interp= torch.randn(1, 230, 256, 256).to(device)
    try:
        print("Test de passage dans le modèle...")
        model(dummy_input, dummy_input) # x et hsi_interp
        print("Succès !")
    except Exception as e:
        print(f"Échec lors du test : {e}")
    """
    
    criterion = SpectralLoss(lambda_sam=args.lambda_sam,lambda_mae=args.lambda_mae,lambda_mse=args.lambda_mse).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

     # --- AJOUT SCHEDULER : WARMUP LINEAIRE + COSINE ANNEALING ---
    warmup_epochs = 5
    cosine_epochs = args.epochs - warmup_epochs

    # 1. Le Warmup : Part de lr*0.1 pour monter jusqu'à lr max en 5 époques
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    # 2. Le Cosine : Redescend de lr max jusqu'à 1e-6
    cosine_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)

    # 3. Enchaînement des deux schedulers
    scheduler = lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    
    ##########
    best_val_loss = float("inf")
    mode_str = "sim" if args.simulated else "real"
    aug_str = "aug" if args.augment else "noaug"
    mlp_str="mlp" if args.with_mlp_spectral else "no_mlp"
    run_name = f"{args.model}_{mode_str}_{aug_str}_SAM-{args.lambda_sam}_MSE{args.lambda_mse}_MAE{args.lambda_mae}_lr-{args.lr}_{mlp_str}_residual"
    #
    
    wandb.init(
    entity="JfGuerrero",                        
    project="Multimodal-change-detection",      
    name=run_name,                
    config=vars(args)                           
)




    for epoch in range(args.epochs):

        train_loss, train_mse,train_mae,train_sam,grad_norm = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_mse,val_mae, val_sam = validate(
            model, val_loader, criterion, device
        )

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss={train_loss:.6f} MAE={train_mae:.6f} MSE={train_mse:.6f} SAM={train_sam:.6f}")
        print(f"Val Loss={val_loss:.6f} MAE={val_mae:.6f} MSE={val_mse:.6f} SAM={val_sam:.6f}")

        scheduler.step()

        wandb.log({
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_sam": train_sam, 
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_sam": val_sam, 
            "lr": optimizer.param_groups[0]['lr'],
            "grad_norm": grad_norm,
        })


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            
            mode_str = "sim" if args.simulated else "real"
            aug_str = "aug" if args.augment else "noaug"
            model_name=f"{args.model}_{mode_str}_{aug_str}_SAM-{args.lambda_sam}_MSE{args.lambda_mse}_MAE{args.lambda_mae}_lr-{args.lr}_{mlp_str}_residual"
    
            
            torch.save(model.state_dict(), model_name)
            artifact = wandb.Artifact(name=run_name, type="model")
            artifact.add_file(model_name)
            wandb.log_artifact(artifact)
            print(f" Saved new best model to {model_name}")

    print("Training finished.")


if __name__ == "__main__":
    main()
