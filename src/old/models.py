import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################################################################################
def get_activation(name):
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)

    if name == "silu":
        return nn.SiLU(inplace=True)

    raise ValueError("Activation must be one of ['relu','leakyrelu','silu']")

##########################################################################################################
class DoubleConv(nn.Module):


    """(convolution => activation) * 2"""
    def __init__(self, in_channels, out_channels, activation="silu",with_batch_norm=True):
        super().__init__()
        if with_batch_norm:
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),           
                nn.BatchNorm2d(out_channels),
            get_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            get_activation(activation),
            )
        else:
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            get_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            get_activation(activation),
            )


    def forward(self, x):
        return self.double_conv(x)

##############################################################################################################
class MLP_spectral(nn.Module):
    

    def __init__(self,in_channels,out_channels,base_channel,activation='silu'):
        super().__init__()
        self.mlp_spectral=nn.Sequential(
            nn.Conv2d(in_channels, base_channel, kernel_size=1 ),
            get_activation(activation),
            nn.Conv2d(base_channel,2*base_channel,kernel_size=1),
            get_activation(activation),
            nn.Conv2d(2*base_channel,out_channels,kernel_size=1),
        )
    def forward(self,x):
        return self.mlp_spectral(x)

#############################################################################################################


class BilinearUpConv(nn.Module):
    """
        F.interpolate and a 1x1 Conv to reduce channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)

#############################################################################################################
    
class GradualExpansionUNet(nn.Module):
    """
    From MSI_simulated, MSI TO HSI
    U-NET
    input:[B,in_msi,H,W]
    output: [B,in_hsi,H,W]
    """
    def __init__(self, in_msi, in_hsi,base_channel=64,interpolation_mode='Bilinear', activation='silu',with_batch_norm=True,with_mlp_spectral=False):
    
        """
           the upsampling is learnable (and could therefore introduce artefacts). With Bilinear, 
           the upsampling is smoother.

           activation in ['relu', 'leakyrelu', 'silu']
        """

        super().__init__()

        if interpolation_mode not in ['ConvTranspose2d', 'Bilinear']:
            raise ValueError("interpolation_mode must be 'ConvTranspose2d' or 'Bilinear'")

        self.interpolation_mode = interpolation_mode
        #self.with_mlp_spectral=with_mlp_spectral

        # [B, in_msi, H, W], in_msi=12
        self.inc = DoubleConv(in_msi, base_channel, activation,with_batch_norm) # [B, base_channel, H, W]

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel,base_channel*2, activation,with_batch_norm))  # [B, 2*base_channel, H/2, W/2]
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel*2,base_channel*4, activation,with_batch_norm)) # [B, 4*base_channel, H/4, W/4]
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel*4, base_channel*8, activation,with_batch_norm)) # [B, 8*base_channel, H/8, W/8]

        if self.interpolation_mode == 'ConvTranspose2d':
            self.up1a = nn.ConvTranspose2d(8*base_channel, 4*base_channel, kernel_size=2, stride=2) # [B, 256, H/4, W/4]
            self.up2a = nn.ConvTranspose2d(2*base_channel+4*base_channel, 4*base_channel, kernel_size=2, stride=2) # [B, 256, H/2, W/2]
            self.up3a = nn.ConvTranspose2d(4*base_channel, 2*base_channel+base_channel, kernel_size=2, stride=2) # [B, 192, H, W]
        else: # Bilinear
            self.up1a = BilinearUpConv(8*base_channel,4*base_channel)
            self.up2a = BilinearUpConv(6*base_channel, 4*base_channel) 
            self.up3a = BilinearUpConv(4*base_channel, 3*base_channel)


        # Concat with down2: [B, 4C+4C, H/4, W/4]
        self.up1b = DoubleConv(4*base_channel+4*base_channel, 6*base_channel, activation,with_batch_norm)    # [B, 6*base_channel, H/4, W/4]
        # Concat with down1: [B, 4C+2C, H/2, W/2]
        self.up2b = DoubleConv(4*base_channel+2*base_channel, 4*base_channel, activation,with_batch_norm)    # [B, 4*base_channel, H/2, W/2]
        # Concat with inc: [B, 3C+C, H, W]
        self.up3b = DoubleConv(4*base_channel, in_hsi, activation,with_batch_norm)   # [B, in_hsi, H, W]

        #if   with_mlp_spectral:
            #self.mlp1b=MLP_spectral(6*base_channel,6*base_channel,6*base_channel,activation)
            #self.mlp2b=MLP_spectral(4*base_channel,4*base_channel,4*base_channel,activation)
            #self.mlp3=MLP_spectral(in_hsi,in_hsi,in_hsi,activation)
        
        self.outc = nn.Conv2d(in_hsi, in_hsi, kernel_size=1) # [B, in_hsi, H, W]

    def forward(self, x):
        s1 = self.inc(x)          # [B, C, H, W]
        s2 = self.down1(s1)       # [B, 2*C, H/2, W/2]
        s3 = self.down2(s2)       # [B, 4*C, H/4, W/4]
        b = self.down3(s3)        # [B, 8*C, H/8, W/8]
        
        x = self.up1a(b)          # [B, 4*C, H/4, W/4]
        x = torch.cat([x, s3], 1) # [B, 4*C+4*C, H/4, W/4] #skip connection on channel
        x = self.up1b(x)
        #if self.with_mlp_spectral:
            #x=self.mlp1b(x)       # [B, 6*C, H/4, W/4]
        #print(x.shape)
        
        x = self.up2a(x)          # [B, 4*C, H/2, W/2]
        x = torch.cat([x, s2], 1) # [B, 4*C+2*C, H/2, W/2] #skip connection
        x = self.up2b(x)
        #if self.with_mlp_spectral:
            #x=self.mlp2b(x)          # [B, 4*C, H/2, W/2]
        #print(x.shape)
        
        x = self.up3a(x)          # [B, 3*C, H, W]
        x = torch.cat([x, s1], 1) # [B, 3*C+1*C, H, W] #skip connection
        x = self.up3b(x)  
        #if self.with_mlp_spectral:
            #x=self.mlp3b(x)         # [B, in_hsi, H, W]
        #print(x.shape)
        
        #return torch.nn.functional.softplus(self.outc(x)) 
        return  self.outc(x)   # [B, in_hsi, H, W]


########################################################################################################

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, interpolation_mode='ConvTranspose2d', activation='silu', drop_out_rate=0.):
        """
           interpolation_mode in ['ConvTranspose2d', 'Bilinear']. With ConvTranspose2d, 
           the upsampling is learnable (and could therefore introduce artefacts). With Bilinear, 
           the upsampling is smoother.

           activation in ['relu', 'leakyrelu', 'silu']
        """
        super().__init__()

        if interpolation_mode not in ['ConvTranspose2d', 'Bilinear']:
            raise ValueError("interpolation_mode must be 'ConvTranspose2d' or 'Bilinear'")
        

        self.interpolation_mode = interpolation_mode

        self.inc = DoubleConv(in_channels, base_channels, activation)           # [B, base_channel, H, W]
        self.down1 = nn.Sequential(nn.MaxPool2d(2),DoubleConv(base_channels, base_channels*2, activation))   # [B, 2*C, H/2, W/2]
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels*2, base_channels*4, activation),
            nn.Dropout2d(p=drop_out_rate)) # [B, 4*C, H/4, W/4]
        self.down3 = nn.Sequential(nn.MaxPool2d(2),  DoubleConv(base_channels*4, base_channels*8, activation),
            nn.Dropout2d(p=drop_out_rate)) # [B, 8*C, H/8, W/8]

        if self.interpolation_mode == 'ConvTranspose2d':
            self.up1a = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)  # [B, 4*C, H/4, W/4]
            self.up2a = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)  # [B, 2*C, H/2, W/2]
            self.up3a = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)    # [B, C, H, W]
        else: # Bilinear
            self.up1a = BilinearUpConv(base_channels*8, base_channels*4)
            self.up2a = BilinearUpConv(base_channels*4, base_channels*2)
            self.up3a = BilinearUpConv(base_channels*2, base_channels)
 
        self.up1b = nn.Sequential(DoubleConv(base_channels*8, base_channels*4, activation),
            nn.Dropout2d(p=drop_out_rate))
        self.up2b = nn.Sequential(DoubleConv(base_channels*4, base_channels*2, activation),
            nn.Dropout2d(p=drop_out_rate))
        self.up3b = DoubleConv(base_channels*2, base_channels, activation)    

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1) 

    def forward(self, x):
        id = x 
        
        s1 = self.inc(x)              # [B, C, H, W]
        s2 = self.down1(s1)           # [B, 2*C, H/2, W/2]
        s3 = self.down2(s2)           # [B, 4*C, H/4, W/4]
        b = self.down3(s3)            # [B, 8*C, H/8, W/8] 
        
        
        out = self.up1a(b)            # [B, 256, H/4, W/4]
        out = torch.cat([out, s3], 1) # [B, 512, H/4, W/4]
        out = self.up1b(out)          # [B, 256, H/4, W/4]
        
        out = self.up2a(out)          # [B, 128, H/2, W/2]
        out = torch.cat([out, s2], 1) # [B, 256, H/2, W/2]
        out = self.up2b(out)          # [B, 128, H/2, W/2]
        
        out = self.up3a(out)          # [B, 64, H, W]
        out = torch.cat([out, s1], 1) # [B, 128, H, W]
        out = self.up3b(out)          # [B, 64, H, W]
        
        res = self.outc(out)

        return res
    
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], drop_out_rate=0., **usl_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums
        self.drop_out_rate = drop_out_rate

        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
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
        x = self.intro(inp)
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
        
        return F.softplus(x)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
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


class GradualExpansionUNet_residual(nn.Module):
    #from msi to hsi with residual approch
    
    def __init__(self, in_msi, in_hsi, interpolation_mode='Bilinear',base_channel=64, activation='silu',augment=True,with_batch_norm=True,with_mlp_spectral=False):
        """
           interpolation_mode in ['ConvTranspose2d', 'Bilinear']. With ConvTranspose2d, 
           the upsampling is learnable (and could therefore introduce artefacts). With Bilinear, 
           the upsampling is smoother.

           activation in ['relu', 'leakyrelu', 'silu']
        """
        super().__init__()

        if interpolation_mode not in ['ConvTranspose2d', 'Bilinear']:
            raise ValueError("interpolation_mode must be 'ConvTranspose2d' or 'Bilinear'")

        self.interpolation_mode = interpolation_mode
        self.augment=augment 
        self.with_mlp_spectral=with_mlp_spectral

                # [B, in_msi, H, W], in_msi=12
        self.inc = DoubleConv(in_msi, base_channel, activation) # [B, base_channel, H, W]

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel,base_channel*2, activation,with_batch_norm))  # [B, 2*base_channel, H/2, W/2]
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel*2,base_channel*4, activation,with_batch_norm)) # [B, 4*base_channel, H/4, W/4]
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channel*4, base_channel*8, activation,with_batch_norm)) # [B, 8*base_channel, H/8, W/8]

        if self.interpolation_mode == 'ConvTranspose2d':
            self.up1a = nn.ConvTranspose2d(8*base_channel, 4*base_channel, kernel_size=2, stride=2) # [B, 256, H/4, W/4]
            self.up2a = nn.ConvTranspose2d(2*base_channel+4*base_channel, 4*base_channel, kernel_size=2, stride=2) # [B, 256, H/2, W/2]
            self.up3a = nn.ConvTranspose2d(4*base_channel, 2*base_channel+base_channel, kernel_size=2, stride=2) # [B, 192, H, W]
        else: # Bilinear
            self.up1a = BilinearUpConv(8*base_channel,4*base_channel)
            self.up2a = BilinearUpConv(6*base_channel, 4*base_channel) 
            self.up3a = BilinearUpConv(4*base_channel, 3*base_channel)
        
        # Concat with down2: [B, 4C+4C, H/4, W/4]
        self.up1b = DoubleConv(4*base_channel+4*base_channel, 6*base_channel, activation,with_batch_norm)    # [B, 6*base_channel, H/4, W/4]
        # Concat with down1: [B, 4C+2C, H/2, W/2]
        
        self.up2b = DoubleConv(4*base_channel+2*base_channel, 4*base_channel, activation,with_batch_norm)    # [B, 4*base_channel, H/2, W/2]
        # Concat with inc: [B, 3C+C, H, W]
    
        self.up3b = DoubleConv(4*base_channel, in_hsi, activation,with_batch_norm)
       
        if   with_mlp_spectral:
            self.mlp1b=MLP_spectral(6*base_channel,6*base_channel,6*base_channel,activation)
            self.mlp2b=MLP_spectral(4*base_channel,4*base_channel,4*base_channel,activation)
            self.mlp3b=MLP_spectral(in_hsi,in_hsi,in_hsi,activation)
        
        self.outc = nn.Conv2d(in_hsi, in_hsi, kernel_size=1) # [B, in_hsi, H, W]

    def forward(self, x,hsi_interp):
        s1 = self.inc(x)          # [B, C, H, W]
        s2 = self.down1(s1)       # [B, 2*C, H/2, W/2]
        s3 = self.down2(s2)       # [B, 4*C, H/4, W/4]
        b = self.down3(s3)        # [B, 8*C, H/8, W/8]
        
        x = self.up1a(b)          # [B, 4*C, H/4, W/4]
        x = torch.cat([x, s3], 1) # [B, 4*C+4*C, H/4, W/4] #skip connection on channel
        x = self.up1b(x)
        if self.with_mlp_spectral:
            x=x+self.mlp1b(x)       # [B, 6*C, H/4, W/4]
        #print(x.shape)
        
        x = self.up2a(x)          # [B, 4*C, H/2, W/2]
        x = torch.cat([x, s2], 1) # [B, 4*C+2*C, H/2, W/2] #skip connection
        x = self.up2b(x)
        if self.with_mlp_spectral:
            x=x+self.mlp2b(x)          # [B, 4*C, H/2, W/2]
        #print(x.shape)
        
        x = self.up3a(x)          # [B, 3*C, H, W]
        x = torch.cat([x, s1], 1) # [B, 3*C+1*C, H, W] #skip connection
        x = self.up3b(x)  
        if self.with_mlp_spectral:
            x=x+self.mlp3b(x)         # [B, in_hsi, H, W]
        #print(x.shape)

        res=self.outc(x)

        
        #return torch.nn.functional.softplus(res+hsi_interp)
        return res+hsi_interp
