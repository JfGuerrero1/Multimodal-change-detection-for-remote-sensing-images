#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   FONCTIONS DE PERTE (LOSSES) POUR MSI/HSI                 ║
║  - Reconstruction spectrale (MSE, Smooth MAE, SAM)                         ║
║  - Quantification d'incertitude (L1, Laplace NLL)                          ║
║  - Apprentissage contrastif (InfoNCE standard et par pixels)               ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. LOSSES POUR LA RECONSTRUCTION SPECTRALE (MSI -> HSI)
# ============================================================================

class SpectralLoss(nn.Module):
    """
    Calcule une perte combinée pour la reconstruction spectrale hyperspectrale :
    Loss = lambda_mse * MSE + lambda_sam * SAM + lambda_mae * Smooth_MAE
    """
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
        """
        Args:
            pred (torch.Tensor): Cube prédit, shape [B, C, H, W] ou [B, C]
            target (torch.Tensor): Cube cible (vérité terrain), shape [B, C, H, W] or [B, C]
            
        Returns:
            tuple: (loss_totale, mse, smooth_mae, sam_rad)
        """
        # 1. Calcul de la MSE et de la Smooth MAE globale
        mse = F.mse_loss(pred, target)
        smooth_mae = F.smooth_l1_loss(pred, target, beta=0.01)

        # 2. Calcul du SAM (Spectral Angle Mapper)
        # Produit scalaire le long de la dimension spectrale (dim=1)
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

        # Loss différentiable (1 - cos(theta))
        sam_loss = torch.mean(1.0 - cos_sim_map)

        # Métrique réelle en radians pour le logging (hors graphe de calcul)
        with torch.no_grad():
            sam_rad = torch.mean(torch.acos(cos_sim_map))

        # 3. Loss Totale pondérée
        loss = (
            self.lambda_mse * mse
            + self.lambda_sam * sam_loss
            + self.lambda_mae * smooth_mae
        )

        return loss, mse, smooth_mae, sam_rad


# ============================================================================
# 2. LOSSES POUR LA QUANTIFICATION D'INCERTITUDE
# ============================================================================

class L1_uncertainty(nn.Module):
    """Loss L1 simple entre l'incertitude prédite U et l'erreur absolue E."""
    def __init__(self):
        super().__init__()

    def forward(self, u: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u (torch.Tensor): Carte d'incertitude prédite, shape [B, C, H, W]
            error (torch.Tensor): Erreur absolue réelle, shape [B, C, H, W]
        """
        return F.l1_loss(u, error)


class LaplaceNLLLossDirect(nn.Module):
    """Loss de Log-Vraisemblance Négative (NLL) sous hypothèse de distribution de Laplace."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, U: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        Args:
            U (torch.Tensor): Échelle d'incertitude (paramètre b de Laplace), shape [B, C, H, W]
            error (torch.Tensor): Erreur absolue (|pred - target|), shape [B, C, H, W]
        """
        # Sécurité pour empêcher U d'être égal ou inférieur à 0 (évite division par zéro et log(0))
        U_safe = torch.clamp(U, min=self.eps)
        
        # Formule de la NLL Laplace : log(U) + (E / U)
        nll = torch.log(U_safe) + (error / U_safe)
        
        return torch.mean(nll)


# ============================================================================
# 3. LOSSES CONTRASTIVES (InfoNCE & Variantes Spatiales)
# ============================================================================

class InfoNce(nn.Module):
    """InfoNCE Loss standard pour l'apprentissage contrastif au niveau batch."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i (torch.Tensor): Embeddings ancre, shape [B, D]
            z_j (torch.Tensor): Embeddings positifs, shape [B, D]
        """
        batch_size = z_i.size(0)
        
        # 1. Normalisation L2 des embeddings sur la dimensionnalité latente
        z_i_norm = F.normalize(z_i, dim=-1)
        z_j_norm = F.normalize(z_j, dim=-1)
        
        # 2. Matrice de similarité de taille [B, B]
        sim_matrix = torch.mm(z_i_norm, z_j_norm.t()) / self.temperature
        
        # 3. Stabilisation numérique (soustraction du max par ligne, détaché du graphe)
        sim_matrix_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
        sim_matrix = sim_matrix - sim_matrix_max.detach()
        
        # 4. Paires positives situées sur la diagonale
        labels = torch.arange(batch_size, device=z_i.device)
        
        return F.cross_entropy(sim_matrix, labels)


class PixelWiseInfoNCE(nn.Module):
    """InfoNCE Loss calculée au niveau des pixels individuels avec masquage spatial."""
    def __init__(
        self,
        temperature: float = 0.07,
        max_positive_pixels: int = 1024,
        max_negative_pixels: int = 1024,
    ):
        super().__init__()
        self.temperature = temperature
        self.max_positive_pixels = max_positive_pixels
        self.max_negative_pixels = max_negative_pixels

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        valid_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            z1, z2 (torch.Tensor): Cartes de features, shape [B, C, H, W]
            positive_mask (torch.Tensor): Masque booléen des positifs, shape [B, 1, H, W] ou [B, H, W]
            negative_mask (torch.Tensor): Masque booléen des négatifs, shape [B, 1, H, W] ou [B, H, W]
            valid_mask (torch.Tensor, optional): Masque de validité globale, shape [B, 1, H, W] ou [B, H, W]
        """
        if z1.shape != z2.shape:
            raise ValueError(f"z1 et z2 doivent avoir la même shape : {z1.shape} vs {z2.shape}")

        B, C, H, W = z1.shape
        device = z1.device

        # Préparation et application des masques spatiaux combinés
        pos_mask = positive_mask.bool()
        neg_mask = negative_mask.bool()
        if valid_mask is not None:
            val_mask = valid_mask.bool()
            pos_mask = pos_mask & val_mask
            neg_mask = neg_mask & val_mask

        # Aplatissement spatial en tenseurs de format [N_total, C]
        z1 = z1.permute(0, 2, 3, 1).reshape(-1, C)
        z2 = z2.permute(0, 2, 3, 1).reshape(-1, C)
        pos_mask = pos_mask.reshape(-1)
        neg_mask = neg_mask.reshape(-1)

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        if pos_indices.numel() == 0:
            return (z1.sum() * 0.0 + z2.sum() * 0.0)

        # Sous-échantillonnage aléatoire si les seuils maximums sont dépassés
        if pos_indices.numel() > self.max_positive_pixels:
            perm = torch.randperm(pos_indices.numel(), device=device)[:self.max_positive_pixels]
            pos_indices = pos_indices[perm]

        if neg_indices.numel() > self.max_negative_pixels:
            perm = torch.randperm(neg_indices.numel(), device=device)[:self.max_negative_pixels]
            neg_indices = neg_indices[perm]

        # Extraction et normalisation L2 en bloc
        anchor = F.normalize(z1[pos_indices], dim=1)      # Shape: [N_pos_sub, C]
        positive = F.normalize(z2[pos_indices], dim=1)    # Shape: [N_pos_sub, C]
        negatives = F.normalize(z2[neg_indices], dim=1)  # Shape: [N_neg_sub, C]

        # Calcul des similarités vectorisées
        pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True) / self.temperature  # Shape: [N_pos_sub, 1]
        neg_logits = torch.matmul(anchor, negatives.t()) / self.temperature              # Shape: [N_pos_sub, N_neg_sub]

        # Concaténation des logits [N_pos_sub, 1 + N_neg_sub]
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=device)

        return F.cross_entropy(logits, labels)


class PixelWiseOtherZoneInfoNCE(nn.Module):
    """InfoNCE avancée intégrant des négatifs croisés issus d'autres zones géographiques/temporelles."""
    def __init__(
        self, 
        temperature: float = 0.07, 
        max_positive_pixels: int = 10000, 
        max_negative_pixels: int = 10000
    ):
        super().__init__()
        self.temperature = temperature
        self.max_positive_pixels = max_positive_pixels
        self.max_negative_pixels = max_negative_pixels

    def forward(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        z1_other: torch.Tensor, 
        z2_other: torch.Tensor, 
        positive_mask: torch.Tensor, 
        negative_mask: torch.Tensor, 
        valid_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            z1, z2, z1_other, z2_other (torch.Tensor): Cartes de features, shape [B, C, H, W]
            positive_mask, negative_mask (torch.Tensor): Masques booléens, shape [B, 1, H, W] ou [B, H, W]
            valid_mask (torch.Tensor, optional): Masque de validité, shape [B, 1, H, W] ou [B, H, W]
        """
        if not (z1.shape == z2.shape == z1_other.shape == z2_other.shape):
            raise ValueError(f"Toutes les features doivent avoir la même shape : {z1.shape}")

        B, C, H, W = z1.shape
        device = z1.device

        # 1. Normalisation L2 spatiale par canal
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        z1_other_norm = F.normalize(z1_other, p=2, dim=1)
        z2_other_norm = F.normalize(z2_other, p=2, dim=1)

        # 2. Aplatissement en [B * H * W, C]
        z1_flat = z1_norm.permute(0, 2, 3, 1).reshape(-1, C)
        z2_flat = z2_norm.permute(0, 2, 3, 1).reshape(-1, C)
        z1_other_flat = z1_other_norm.permute(0, 2, 3, 1).reshape(-1, C)
        z2_other_flat = z2_other_norm.permute(0, 2, 3, 1).reshape(-1, C)

        # 3. Masquage combiné et aplatissement
        pos_mask = positive_mask.bool().reshape(-1)
        neg_mask = negative_mask.bool().reshape(-1)
        if valid_mask is not None:
            val_mask = valid_mask.bool().reshape(-1)
            pos_mask = pos_mask & val_mask
            neg_mask = neg_mask & val_mask

        #Pixel positifs on veut les rapprocher dans l'espace latent, pixle négatif on veut les eloigner 
        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        # Sécurité : si aucun pixel positif n'est présent, retourne une loss nulle avec gradient
        if pos_indices.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 4. Sous-échantillonnage optionnel pour limiter l'empreinte mémoire
        if pos_indices.numel() > self.max_positive_pixels:
            perm = torch.randperm(pos_indices.numel(), device=device)[:self.max_positive_pixels]
            pos_indices = pos_indices[perm]

        if neg_indices.numel() > self.max_negative_pixels:
            perm = torch.randperm(neg_indices.numel(), device=device)[:self.max_negative_pixels]
            neg_indices = neg_indices[perm]

        # 5. Extraction des pixels sélectionnés
        z1_pos = z1_flat[pos_indices]              # Shape: [N_pos_sub, C]
        z2_pos = z2_flat[pos_indices]              # Shape: [N_pos_sub, C]
        z1_other_neg = z1_other_flat[pos_indices]  # Shape: [N_pos_sub, C]
        z2_other_neg = z2_other_flat[pos_indices]  # Shape: [N_pos_sub, C]

        # 6. Logit Positif (Ancre et positif correspondants) -> Shape: [N_pos_sub, 1]
        pos_logits = torch.sum(z1_pos * z2_pos, dim=1, keepdim=True) / self.temperature

        # 7. Logits Négatifs Croisés (zones "other" aux mêmes positions spatiales)
        neg_logits_1 = torch.sum(z1_pos * z2_other_neg, dim=1, keepdim=True) / self.temperature  # Shape: [N_pos_sub, 1]
        neg_logits_2 = torch.sum(z1_other_neg * z2_pos, dim=1, keepdim=True) / self.temperature  # Shape: [N_pos_sub, 1]

        list_neg_logits = [neg_logits_1, neg_logits_2]

        # 8. Logits Négatifs Standards (pixels de zones négatives classiques)
        if neg_indices.numel() > 0:
            z2_neg = z2_flat[neg_indices]  # Shape: [N_neg_sub, C]
            # Produit matriciel entre les ancres positives et les négatifs standards -> Shape: [N_pos_sub, N_neg_sub]
            neg_logits_3 = torch.matmul(z1_pos, z2_neg.t()) / self.temperature
            list_neg_logits.append(neg_logits_3)

        # Concaténation de tous les blocs de négatifs sur la dimension des colonnes (dim=1)
        neg_logits_all = torch.cat(list_neg_logits, dim=1)  # Shape: [N_pos_sub, 1 + 1 + N_neg_sub]

        # 9. Logits finaux combinant le positif et tous les négatifs
        logits = torch.cat([pos_logits, neg_logits_all], dim=1)  # Shape: [N_pos_sub, 1 + N_neg_total]


        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        # 11. Calcul final de la perte d'entropie croisée
        loss = F.cross_entropy(logits, labels)

        return loss