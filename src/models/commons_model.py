import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str) -> nn.Module:
    """Returns the activation module corresponding to the specified name.

    Args:
        name (str): The activation name ('relu', 'leakyrelu', 'silu').

    Returns:
        nn.Module: The corresponding PyTorch activation module.
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)

    if name == "silu":
        return nn.SiLU(inplace=True)

    raise ValueError("Activation must be one of ['relu', 'leakyrelu', 'silu']")


class BilinearUpConv(nn.Module):
    """Performs bilinear upsampling (F.interpolate) followed by a 1x1

    convolution to reduce channel dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H*2, W*2)
        """
        # x: (N, C_in, H, W)
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )
        # x after interpolate: (N, C_in, H*2, W*2)
        return self.conv(x)
        # Output: (N, C_out, H*2, W*2)


class DoubleConv(nn.Module):
    """Double convolution block: (Convolution => [BatchNorm] => Activation) * 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "silu",
        with_batch_norm: bool = True,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        # x: (N, in_channels, H, W)
        return self.double_conv(x)
        # Output: (N, out_channels, H, W)


class MLP_spectral(nn.Module):
    """Multi-Layer Perceptron (MLP) applied along the spectral/channel dimension

    using successive 1x1 convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channel: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.mlp_spectral = nn.Sequential(
            nn.Conv2d(in_channels, base_channel, kernel_size=1),
            get_activation(activation),
            nn.Conv2d(base_channel, 2 * base_channel, kernel_size=1),
            get_activation(activation),
            nn.Conv2d(2 * base_channel, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        # x: (N, in_channels, H, W)
        return self.mlp_spectral(x)
        # Output: (N, out_channels, H, W)


class LayerNormFunction(torch.autograd.Function):
    """Custom Layer Normalization autograd function optimized for 2D spatial

    tensors of shape (N, C, H, W).
    """

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
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    """nn.Module wrapper for the custom 2D Layer Normalization."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Normalized tensor of shape (N, C, H, W)
        """
        # x: (N, C, H, W)
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        # Output: (N, C, H, W)


class SimpleGate(nn.Module):
    """SimpleGate used in NAFNet architectures to split channels

    and perform an element-wise multiplication.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 2 * C_half, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_half, H, W)
        """
        # x: (N, 2 * C_half, H, W)
        x1, x2 = x.chunk(2, dim=1)
        # x1, x2: (N, C_half, H, W)
        return x1 * x2
        # Output: (N, C_half, H, W)


class NAFBlock(nn.Module):
    """Nonlinear Activation Free (NAF) Block from NAFNet."""

    def __init__(
        self,
        c: int, 
        DW_Expand: int = 2, #Depth-Wise Expansion)
        FFN_Expand: int = 2, # (Feed-Forward Network Expansion)
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(
            torch.zeros((1, c, 1, 1)), requires_grad=True
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            inp (torch.Tensor): Input tensor of shape (N, c, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, c, H, W)
        """
        # inp: (N, c, H, W)
        x = inp

        x = self.norm1(x)  # (N, c, H, W)
        x = self.conv1(x)  # (N, dw_channel, H, W)
        x = self.conv2(x)  # (N, dw_channel, H, W)
        x = self.sg(x)  # (N, dw_channel // 2, H, W)
        x = x * self.sca(x)  # (N, dw_channel // 2, H, W)
        x = self.conv3(x)  # (N, c, H, W)
        x = self.dropout1(x)  # (N, c, H, W)
        y = inp + x * self.beta  # (N, c, H, W)

        x = self.norm2(y)  # (N, c, H, W)
        x = self.conv4(x)  # (N, ffn_channel, H, W)
        x = self.sg(x)  # (N, ffn_channel // 2, H, W)
        x = self.conv5(x)  # (N, c, H, W)
        x = self.dropout2(x)  # (N, c, H, W)

        return y + x * self.gamma
        # Output: (N, c, H, W)


class NAFNet(nn.Module):
    """
    Nonlinear Activation Free Network (NAFNet) adapted for uncertainty estimation 
    in hyperspectral (HSI) data processing.
    """
    def __init__(self, in_channels=3, out_channels=3, width=16, middle_blk_num=1, 
                 enc_blk_nums=[], dec_blk_nums=[], drop_out_rate=0., 
                 uncertainty_activation='exp', **usl_kwargs):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # Corresponds to HSI bands or global/pixel-wise uncertainty
        self.uncertainty_activation = uncertainty_activation

        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums
        self.drop_out_rate = drop_out_rate

        # Initial feature extraction layer
        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        # Final projection layer to map features to raw uncertainty outputs
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Build encoder path (hierarchical feature extraction)
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=0.) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        # Build bottleneck / middle blocks
        self.middle_blks = nn.Sequential(*[NAFBlock(chan, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)])

        # Build decoder path (feature reconstruction with skip connections)
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop_out_rate=0.) for _ in range(num)]))

        # Padding alignment factor based on encoder depth
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        # Pad input to match U-Net scale constraints
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []

        # Encoder pass with feature caching for skip connections
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Bottleneck processing
        x = self.middle_blks(x)

        # Decoder pass combining upsampled features and skip connections
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        # Project to raw uncertainty values and crop back to original dimensions
        raw_unc = self.ending(x)
        raw_unc = raw_unc[:, :, :H, :W]
        
        # Apply modular activation function to ensure strictly positive uncertainty outputs
        eps = 1e-4
        if self.uncertainty_activation == 'softplus':
            uncertainty = F.softplus(raw_unc) + eps
        elif self.uncertainty_activation == 'exp':
            uncertainty = torch.exp(raw_unc) + eps
        elif self.uncertainty_activation == 'shifted_relu':
            uncertainty = F.relu(raw_unc) + eps
        else:
            raise ValueError(f"Unknown uncertainty activation: {self.uncertainty_activation}")
            
        return uncertainty

    def check_image_size(self, x):
        """Pads spatial dimensions (H, W) to be multiples of padder_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x