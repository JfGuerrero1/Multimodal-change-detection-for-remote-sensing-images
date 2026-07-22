import torch
import torch.nn as nn
import torch.nn.functional as F
from .commons_model import DoubleConv, BilinearUpConv, MLP_spectral

class GradualExpansionUNet_residual(nn.Module):
    """
    Gradual Expansion U-Net with Residual Approach for MSI to HSI reconstruction.
    Input x:          [B, in_msi, H, W]  (e.g., [B, 12, H, W])
    Input hsi_interp: [B, in_hsi, H, W]  (e.g., [B, 230, H, W])
    Output:           [B, in_hsi, H, W]  (e.g., [B, 230, H, W])
    """
    def __init__(self, in_msi, in_hsi, interpolation_mode='Bilinear', base_channel=64, 
                 activation='silu', augment=True, with_batch_norm=True, 
                 with_mlp_spectral=False, final_activation=None):
        super().__init__()

        if interpolation_mode not in ['ConvTranspose2d', 'Bilinear']:
            raise ValueError("interpolation_mode must be 'ConvTranspose2d' or 'Bilinear'")

        self.interpolation_mode = interpolation_mode
        self.augment = augment 
        self.with_mlp_spectral = with_mlp_spectral

        # --- ENCODER (Downsampling) ---
        self.inc = DoubleConv(in_msi, base_channel, activation, with_batch_norm) 
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel, base_channel * 2, activation, with_batch_norm))  
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel * 2, base_channel * 4, activation, with_batch_norm)) 
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel * 4, base_channel * 8, activation, with_batch_norm)) 

        # --- DECODER UPSAMPLING LAYERS ---
        if self.interpolation_mode == 'ConvTranspose2d':
            self.up1a = nn.ConvTranspose2d(8 * base_channel, 4 * base_channel, kernel_size=2, stride=2) 
            self.up2a = nn.ConvTranspose2d(6 * base_channel, 4 * base_channel, kernel_size=2, stride=2) 
            self.up3a = nn.ConvTranspose2d(4 * base_channel, 3 * base_channel, kernel_size=2, stride=2) 
        else: 
            self.up1a = BilinearUpConv(8 * base_channel, 4 * base_channel)
            self.up2a = BilinearUpConv(6 * base_channel, 4 * base_channel) 
            self.up3a = BilinearUpConv(4 * base_channel, 3 * base_channel)
        
        # --- DECODER CONVOLUTIONAL BLOCKS ---
        self.up1b = DoubleConv(4 * base_channel + 4 * base_channel, 6 * base_channel, activation, with_batch_norm)    
        self.up2b = DoubleConv(4 * base_channel + 2 * base_channel, 4 * base_channel, activation, with_batch_norm)    
        self.up3b = DoubleConv(3 * base_channel + base_channel, in_hsi, activation, with_batch_norm)
       
        # --- OPTIONAL SPECTRAL MLP BLOCKS ---
        if with_mlp_spectral:
            self.mlp1b = MLP_spectral(6 * base_channel, 6 * base_channel, 6 * base_channel, activation)
            self.mlp2b = MLP_spectral(4 * base_channel, 4 * base_channel, 4 * base_channel, activation)
            self.mlp3b = MLP_spectral(in_hsi, in_hsi, in_hsi, activation)
        
        # Final Projection
        self.outc = nn.Conv2d(in_hsi, in_hsi, kernel_size=1) 

        # --- FINAL ACTIVATION LAYER---
        if final_activation is None:
            self.final_act = nn.Identity()
        elif final_activation.lower() == 'softplus':
            self.final_act = nn.Softplus()
        elif final_activation.lower() == 'relu':
            self.final_act = nn.ReLU(inplace=True)
        else:
            raise ValueError("final_activation must be one of [None, 'relu', 'softplus']")

    def forward(self, x, hsi_interp):
        # --- Encoder ---
        s1 = self.inc(x)              # [B, C, H, W]            (Skip connection 3)
        s2 = self.down1(s1)           # [B, 2C, H/2, W/2]       (Skip connection 2)
        s3 = self.down2(s2)           # [B, 4C, H/4, W/4]       (Skip connection 1)
        b = self.down3(s3)            # [B, 8C, H/8, W/8]       (Bottleneck)
        
        # --- Decoder Block 1 ---
        x = self.up1a(b)              # [B, 4C, H/4, W/4]       (Upsampling)
        x = torch.cat([x, s3], dim=1) # [B, 4C + 4C, H/4, W/4] -> [B, 8C, H/4, W/4] (Concat)
        x = self.up1b(x)              # [B, 6C, H/4, W/4]       (Double Conv)
        if self.with_mlp_spectral:
            x = x + self.mlp1b(x)     # [B, 6C, H/4, W/4]       (Spectral MLP Residual)
        
        # --- Decoder Block 2 ---
        x = self.up2a(x)              # [B, 4C, H/2, W/2]       (Upsampling)
        x = torch.cat([x, s2], dim=1) # [B, 4C + 2C, H/2, W/2] -> [B, 6C, H/2, W/2] (Concat)
        x = self.up2b(x)              # [B, 4C, H/2, W/2]       (Double Conv)
        if self.with_mlp_spectral:
            x = x + self.mlp2b(x)     # [B, 4C, H/2, W/2]       (Spectral MLP Residual)
        
        # --- Decoder Block 3 ---
        x = self.up3a(x)              # [B, 3C, H, W]           (Upsampling)
        x = torch.cat([x, s1], dim=1) # [B, 3C + C, H, W] -> [B, 4C, H, W]   (Concat)
        x = self.up3b(x)              # [B, in_hsi, H, W]       (Double Conv)
        if self.with_mlp_spectral:
            x = x + self.mlp3b(x)     # [B, in_hsi, H, W]       (Spectral MLP Residual)

        # --- Residual Mapping Addition ---
        res = self.outc(x)            # [B, in_hsi, H, W]       (Final Residual Prediction)
        out = res + hsi_interp        # [B, in_hsi, H, W]       (Combined Output)
        
        return self.final_act(out)    # [B, in_hsi, H, W]       (Final Output via Identity/Act)