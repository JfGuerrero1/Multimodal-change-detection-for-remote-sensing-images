
import torch
import torch.nn as nn
import torch.nn.functional as F
from .commons_model import NAFBlock,DoubleConv,BilinearUpConv

class DualBranchUNet(nn.Module):
    def __init__(self, n_msi=12, n_hsi=230, base_channels=64, interpolation_mode='ConvTranspose2d', activation='silu', final_op='identity', drop_out_rate=0.):
        """
        final_op in ['identity', 'abs', 'softplus', 'square']
        """
        super().__init__()

        if interpolation_mode not in ['ConvTranspose2d', 'Bilinear']:
            raise ValueError("interpolation_mode must be 'ConvTranspose2d' or 'Bilinear'")
        if final_op not in ['identity','abs', 'softplus', 'square']:
            raise ValueError("final_op must be 'identity', 'abs', 'softplus', or 'square'")
        
        self.interpolation_mode = interpolation_mode
        self.final_op = final_op

        # base_channels // 2 so we retrieve base_channels after concatenation
        self.branch_msi = DoubleConv(n_msi, base_channels // 2)
        self.branch_hsi = DoubleConv(n_hsi, base_channels // 2)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(base_channels, base_channels*2, activation))   # [B, 128, H/2, W/2]
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(base_channels*2, base_channels*4, activation),
            nn.Dropout2d(p=drop_out_rate)) # [B, 256, H/4, W/4]
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(base_channels*4, base_channels*8, activation),
            nn.Dropout2d(p=drop_out_rate)) # [B, 512, H/8, W/8]
        
        if self.interpolation_mode == 'ConvTranspose2d':
            self.up1a = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)  # [B, 256, H/4, W/4]
            self.up2a = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)  # [B, 128, H/2, W/2]
            self.up3a = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)    # [B, 64, H, W]
        else: # Bilinear
            self.up1a = BilinearUpConv(base_channels*8, base_channels*4)
            self.up2a = BilinearUpConv(base_channels*4, base_channels*2)
            self.up3a = BilinearUpConv(base_channels*2, base_channels)

        self.up1b = nn.Sequential(
            DoubleConv(base_channels*8, base_channels*4, activation),
            nn.Dropout2d(p=drop_out_rate))
        self.up2b = nn.Sequential(
            DoubleConv(base_channels*4, base_channels*2, activation),
            nn.Dropout2d(p=drop_out_rate))
        self.up3b = DoubleConv(base_channels*2, base_channels, activation)  

        self.outc = nn.Conv2d(base_channels, n_hsi, kernel_size=1) 

    def forward(self, x):
        x_msi = x[:, :12, :, :]   
        x_hsi = x[:, 12:, :, :]   

        feat_msi = self.branch_msi(x_msi) # [B, 32, H, W]
        feat_hsi = self.branch_hsi(x_hsi) # [B, 32, H, W]

        s1 = torch.cat([feat_msi, feat_hsi], dim=1) # [B, 64, H, W]

        s2 = self.down1(s1)           # [B, 128, H/2, W/2]
        s3 = self.down2(s2)           # [B, 256, H/4, W/4]
        b  = self.down3(s3)           # [B, 512, H/8, W/8] 
        
        out = self.up1a(b)            # [B, 256, H/4, W/4]
        out = torch.cat([out, s3], 1) # [B, 512, H/4, W/4]
        out = self.up1b(out)          # [B, 256, H/4, W/4]
        
        out = self.up2a(out)          # [B, 128, H/2, W/2]
        out = torch.cat([out, s2], 1) # [B, 256, H/2, W/2]
        out = self.up2b(out)          # [B, 128, H/2, W/2]
        
        out = self.up3a(out)          # [B, 64, H, W]
        out = torch.cat([out, s1], 1) # [B, 128, H, W] (On utilise le s1 fusionné !)
        out = self.up3b(out)          # [B, 64, H, W]
        
        res = self.outc(out)  
        
        if self.final_op == 'abs':
            return torch.abs(res)
        elif self.final_op == 'softplus':
            return F.softplus(res)
        elif self.final_op == 'square':
            return res ** 2
        else:
            return res
        
class DualBranchNAFNet(nn.Module):
    def __init__(self, n_msi=12, n_hsi=230, out_channels=230, width=64, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], drop_out_rate=0., final_op='abs', **usl_kwargs):
        super().__init__()
        self.n_msi = n_msi
        self.n_hsi = n_hsi
        self.out_channels = out_channels
        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums
        self.drop_out_rate = drop_out_rate
        self.final_op = final_op

        if final_op not in ['identity', 'abs', 'softplus', 'square']:
            raise ValueError("final_op must be 'identity', 'abs', 'softplus', or 'square'")

        width_msi = width // 2
        width_hsi = width - width_msi # Gère proprement le cas où width serait impair

        self.intro_msi = nn.Conv2d(in_channels=n_msi, out_channels=width_msi, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.intro_hsi = nn.Conv2d(in_channels=n_hsi, out_channels=width_hsi, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=0.) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=0.) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x_msi = inp[:, :self.n_msi, :, :]
        x_hsi = inp[:, self.n_msi:, :, :]
        feat_msi = self.intro_msi(x_msi) # [B, width//2, H_pad, W_pad]
        feat_hsi = self.intro_hsi(x_hsi) # [B, width//2, H_pad, W_pad]
       
        x = torch.cat([feat_msi, feat_hsi], dim=1) # -> [B, width, H_pad, W_pad]
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x[:, :, :H, :W]
        
        if self.final_op == 'abs':
            return torch.abs(x)
        elif self.final_op == 'softplus':
            return F.softplus(x)
        elif self.final_op == 'square':
            return x ** 2
        else:
            return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
