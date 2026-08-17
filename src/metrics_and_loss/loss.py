import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """Loss = lambda_mse * MSE + lambda_sam * SAM + lambda_mae * Smoothed MAE"""
    def __init__(
        self, 
        lambda_mse: float = 1.0, 
        lambda_mae: float = 0.0, 
        lambda_sam: float = 0.1, 
        eps: float = 1e-6  # Ajusté à 1e-6 pour la stabilité FP16/AMP
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_mae = lambda_mae
        self.lambda_sam = lambda_sam
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
    # 1. Calcul de la MSE et MAE globale
        mse = F.mse_loss(pred, target)
        smooth_mae =  F.smooth_l1_loss(pred, target, beta=0.01)

    # 2. Calcul du SAM (Spectral Angle Mapper)

        dot_product = torch.sum(pred * target, dim=1)

    # Sécurité : Clamp sur chaque norme individuelle pour éviter grad=Inf si norm=0
        norm_pred = torch.clamp(
        torch.linalg.vector_norm(pred, dim=1), min=self.eps
    )
        norm_target = torch.clamp(
        torch.linalg.vector_norm(target, dim=1), min=self.eps
    )

        denominator = norm_pred * norm_target

    # Sécurité : Clamp strict dans [-1.0, 1.0] pour éviter loss négative ou acos NaN
        cos_sim_map = torch.clamp(dot_product / denominator, -1.0 + self.eps, 1.0 - self.eps)

    # Loss differentiable (1 - cos(theta))
        sam_loss = torch.mean(1.0 - cos_sim_map)

    # Métrique réelle en radians pour le logging (hors graphe de calcul)
        with torch.no_grad():
            sam_rad = torch.mean(torch.acos(cos_sim_map))

    # 3. Loss Totale
        loss = (
            self.lambda_mse * mse
        + self.lambda_sam * sam_loss
        + self.lambda_mae * smooth_mae
    )

    # On retourne sam_rad pour les logs/wandb
        return loss, mse, smooth_mae, sam_rad


class L1_uncertainty(nn.Module):
    """Loss: L1 entre l'incertitude prédite U et l'erreur absolue E."""
    def __init__(self):
        super().__init__()

    def forward(self, u, error):
        return F.l1_loss(u, error)