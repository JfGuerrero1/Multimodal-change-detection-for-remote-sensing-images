import os
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from tqdm import tqdm

from models import UNet, GradualExpansionUNet
from losses.spectral_loss import SpectralLoss
from old.utils_dataset import create_data_loaders_spectral, SpectralDataset

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent.parent

DATA_DIR = ROOT_DIR / "data" / "dataset"
CACHE_DIR = DATA_DIR / "patches_cache"
DEFAULT_SRF_PATH = DATA_DIR / "srf_matrix_norm_s2b.npy"
RESULT_DIR = ROOT_DIR  / "results"
PLOT_DIR = RESULT_DIR / "Result_plot"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT_DIR / "models"


def load_fixed_weights(model, path, device):
    state_dict = torch.load(path, map_location=device)
    
    # 1. Nettoyer le préfixe 'module.' si présent
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # 2. Charger les poids
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Chargement réussi !")
    except RuntimeError as e:
        print(f"Échec du chargement. Voici l'erreur : {e}")
        
    return model

def build_model(args,n_msi,n_hsi):
    if args.model == "unet":
        model = UNet(
            activation=args.activation,
            interpolation_mode=args.interpolation_mode,
           
        )
    elif args.model == "gradualexpansionunet":
        model = GradualExpansionUNet(
            activation=args.activation,
            in_msi=n_msi,
            in_hsi=n_hsi,
            interpolation_mode=args.interpolation_mode
        )
    else:
        raise ValueError("Modèle inconnu")
    return model

@torch.no_grad()
def test_model_metric(model,loader,criterion,device,save_dir):
    absolute_save_dir = os.path.abspath(str(save_dir))
    os.makedirs(absolute_save_dir, exist_ok=True)

    model.eval()

    all_mse=[]
    all_mrae=[]
    all_sam=[]

    eps=10**-6

    pbar=tqdm(loader, desc='Testing')

    for x, y in pbar: #(MSI,HSI_gt)
        x=x.to(device)
        y=y.to(device)
        pred=model(x) #HSI_pred

        for b in range(x.size(0)): #on parcourt le batch
            pred_patch=pred[b]
            y_patch=y[b] #(C,W,H)

            ### COmpute MSE for this patch
        
            mse_patch = torch.mean((pred_patch - y_patch) ** 2, dim=(1, 2))
            all_mse.append(mse_patch.cpu().numpy())
            # affichage en temps reele de la mse globale par patch
            mse_total_batch = torch.mean((pred_patch - y_patch) ** 2)
            print(f"MSE brute globale de ce patch : {mse_total_batch.item()}")

            ### MRAE

            mrae_map=torch.abs(pred_patch-y_patch)/(y_patch+eps)
            mrae_patch=torch.mean(mrae_map,dim=(1,2))
            all_mrae.append(mrae_patch.cpu().numpy())


            ####SAM
            dot_product = torch.sum(pred_patch * y_patch, dim=0)
            norm_pred = torch.norm(pred_patch, p=2, dim=0)
            norm_y = torch.norm(y_patch, p=2, dim=0)
                
            cos_sim = torch.clamp(dot_product / (norm_pred * norm_y + eps), -1.0 + eps, 1.0 - eps)
            sam_map_deg = torch.acos(cos_sim) 
                
            sam_patch_val = torch.mean(sam_map_deg).item()
            print(sam_patch_val)
            all_sam.append(sam_patch_val)

    matrix_mse=np.array(all_mse) #Nb_patch,C
    matrix_mrae=np.array(all_mrae) #(Nb_patch,C)
    array_sam=np.array(all_sam) #Nb_patch


    mse_mean = np.mean(matrix_mse, axis=0)
    mse_std = np.std(matrix_mse, axis=0)
    
    mrae_mean = np.mean(matrix_mrae, axis=0)
    mrae_std = np.std(matrix_mrae, axis=0)

    
    # SAM final (Scalaires uniques)
    sam_moyen = np.mean(array_sam)
    sam_std = np.std(array_sam)

    # Affichage rapide console
    print("\n" + "="*50)
    print(f"    BILAN STATISTIQUE SUR LES {len(array_sam)} PATCHS DU TEST SET    ")
    print("="*50)
    print(f"SAM Global Moyen  : {sam_moyen:.2f}° (± {sam_std:.2f}°)")
    print("="*50)

    
    np.save(save_dir / "hsi_mse_moyenne.npy", mse_mean)
    np.save(save_dir / "hsi_mse_std.npy", mse_std)
    np.save(save_dir / "hsi_mrae_moyenne.npy", mrae_mean)
    np.save(save_dir / "hsi_mrae_std.npy", mrae_std)

    #
    sam_stats = {
        "sam_moyen": float(np.mean(array_sam)),
        "sam_std": float(np.std(array_sam))
                 }
    with open(save_dir / "sam_global_stats.json", "w") as f:
        json.dump(sam_stats, f, indent=4)
        
    print(f"Évaluation terminée ! Fichiers sauvegardés dans {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "gradualexpansionunet"])
    parser.add_argument("--activation", type=str, default="silu", choices=["relu", "leakyrelu", "silu"])

    parser.add_argument("--interpolation_mode", type=str, default="Bilinear", choices=["ConvTranspose2d", "Bilinear"])
    parser.add_argument("--lambda_sam", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_model", type=str, default=None, required=False, help="Chemin vers les poids .pth")
    parser.add_argument("--simulated", action="store_true", help="Évaluer sur la MSI simulée")
    parser.set_defaults(is_residual=False) # Pas toucher 
    parser.add_argument('--no_keep_atm_wave', dest='keep_atm_wave', action='store_false', help="Supprime les longueurs d'onde atmosphériques")
    parser.set_defaults(keep_atm_wave=True) # Défini par défaut à True
    
    parser.add_argument("--is_residual", action="store_true", help="Approche résiduelle pour calculer la hsi")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    train_loader,val_loader,test_loader = create_data_loaders_spectral(
        use_simulated_msi=args.simulated,
        augment=False, #pour le test pas besoin
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_residual=args.is_residual,
        keep_atm_wave=args.keep_atm_wave,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Modele réel")
    poids_real= "/home/ids/jfguerrero/Multimodal-change-detection-for-remote-sensing-images/models/best_gradualexpansionunet_SAM-0.5_lr-0.0005_real_aug_final.pth" 
    model_real = GradualExpansionUNet(in_msi=12, in_hsi=230, interpolation_mode="Bilinear", activation="silu")

    model_real = load_fixed_weights(model_real, poids_real, device)

    model_real = model_real.to(device)
    
    #
    # print(f"Modèle chargé et transféré avec succès sur {device} depuis {args.load_model}")
    
    

    criterion = SpectralLoss(lambda_sam=args.lambda_sam).to(device)
    save_dir=RESULT_DIR/"gradualexpansionunet_real_aug_SAM-0.5_lr-0.0005_final_sans_kirtland"
    test_model_metric(model_real, test_loader, criterion, device, save_dir)



"""
######################################################################################################
#######################DEBOGUAGE######################################################################
######################################################################################################
def test_data_loaders():
    print("=== TEST DES DATALOADERS (MODE RÉEL) ===")
    # 1. On teste le loader en mode RÉEL
    _, _, real_loader = create_data_loaders(
        dataset_class=SpectralDataset,
        use_simulated_msi=False,  # <-- Mode Réel
        split='test'
        augment=False,
        batch_size=8,
        is_residual=False

    )
    
    # On prend le tout premier batch du loader réel
    real_x, real_y = next(iter(real_loader))
    print(f"[Réel] Shape de X (MSI) : {real_x.shape} (Attendu: [B, 12, H, W])")
    print(f"[Réel] Shape de y (HSI) : {real_y.shape} (Attendu: [B, 230, H, W])")
    
    print("\n=== TEST DES DATALOADERS (MODE SIMULÉ) ===")
    # 2. On teste le loader en mode SIMULÉ
    _, _, sim_loader = create_data_loaders(
        dataset_class=SpectralDataset,
        use_simulated_msi=True,  # <-- Mode Simulé
        split='test'
        augment=False,
        batch_size=8,
        is_residual=False
    )
    
    # On prend le tout premier batch du loader simulé
    sim_x, sim_y = next(iter(sim_loader))
    print(f"[Simulé] Shape de X (MSI) : {sim_x.shape} (Attendu: [B, 12, H, W])")
    print(f"[Simulé] Shape de y (HSI) : {sim_y.shape} (Attendu: [B, 230, H, W])")

    # 3. Double-check ultime : s'assurer que les batchs ne sont pas identiques
    are_loaders_identical = torch.equal(real_x, sim_x)
    print(f"\n[Verdict] Les tenseurs MSI X du premier batch sont-ils identiques ? {are_loaders_identical}")

"""


if __name__ == "__main__":
    #test_data_loaders()
    main()
