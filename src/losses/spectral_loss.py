import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralLoss(nn.Module):
    """Loss = lambda_mse* MSE + lambda_sam * SAM+lambda_mae* MAE"""
    def __init__(self, lambda_mse=1.0,lambda_mae=0.0, lambda_sam=0.1, eps=1e-8):
        super().__init__()
        self.lambda_sam = lambda_sam
        self.eps = eps
        self.lambda_mse=lambda_mse
        self.lambda_mae=lambda_mae

    def forward(self, pred, target):
        # 1. Calcul de la MSE et MAE globale
        mse = F.mse_loss(pred, target)
        mae=F.l1_loss(pred,target)



        # 2. Calcul du SAM (Spectral Angle Mapper)
        # On calcule le produit scalaire le long de la dimension des canaux (dim=1)
        dot_product = torch.sum(pred * target, dim=1)
        norm_pred = torch.linalg.vector_norm(pred, dim=1)
        norm_target = torch.linalg.vector_norm(target, dim=1)
        
        cos_sim_map = dot_product / (norm_pred * norm_target + self.eps)
        cos_sim_map = torch.clamp(cos_sim_map, -1.0 + self.eps, 1.0 - self.eps)
        
        sam = torch.mean(torch.acos(cos_sim_map))
        
        # 3. Total
        loss = self.lambda_mse * mse + self.lambda_sam * sam +self.lambda_mae*mae
        return loss, mse, mae,  sam
