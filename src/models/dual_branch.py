import torch
import torch.nn as nn
import torch.nn.functional as F
from .commons_model import BilinearUpConv, DoubleConv, NAFBlock, NAFNet


class DualBranchUNet(nn.Module):
    """Dual-branch U-Net for joint MSI and HSI processing and prediction."""

    def __init__(
        self,
        n_msi: int = 12,
        n_hsi: int = 230,
        base_channels: int = 64,
        interpolation_mode: str = "ConvTranspose2d",
        activation: str = "silu",
        final_op: str = "identity",
        drop_out_rate: float = 0.0,
    ):
        """Args:

        n_msi (int): Number of MSI channels. n_hsi (int): Number of HSI
        channels. base_channels (int): Base number of channels.
        interpolation_mode (str): 'ConvTranspose2d' or 'Bilinear'. activation
        (str): Activation function name. final_op (str): Final operation
        ('identity', 'abs', 'softplus', 'square'). drop_out_rate (float):
        Dropout rate.
        """
        super().__init__()
        self.n_msi = n_msi
        self.n_hsi = n_hsi

        if interpolation_mode not in ["ConvTranspose2d", "Bilinear"]:
            raise ValueError(
                "interpolation_mode must be 'ConvTranspose2d' or 'Bilinear'"
            )
        if final_op not in ["identity", "abs", "softplus", "square"]:
            raise ValueError(
                "final_op must be 'identity', 'abs', 'softplus', or 'square'"
            )

        self.interpolation_mode = interpolation_mode
        self.final_op = final_op

        # base_channels // 2 so we retrieve base_channels after concatenation
        self.branch_msi = DoubleConv(n_msi, base_channels // 2)
        self.branch_hsi = DoubleConv(n_hsi, base_channels // 2)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2, activation)
        )  # [B, 2*C, H/2, W/2]
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, base_channels * 4, activation),
            nn.Dropout2d(p=drop_out_rate),
        )  # [B, 4*C, H/4, W/4]
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 4, base_channels * 8, activation),
            nn.Dropout2d(p=drop_out_rate),
        )  # [B, 8*C, H/8, W/8]

        if self.interpolation_mode == "ConvTranspose2d":
            self.up1a = nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4, kernel_size=2, stride=2
            )  # [B, 256, H/4, W/4]
            self.up2a = nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, kernel_size=2, stride=2
            )  # [B, 128, H/2, W/2]
            self.up3a = nn.ConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=2, stride=2
            )  # [B, 64, H, W]
        else:  # Bilinear
            self.up1a = BilinearUpConv(base_channels * 8, base_channels * 4)
            self.up2a = BilinearUpConv(base_channels * 4, base_channels * 2)
            self.up3a = BilinearUpConv(base_channels * 2, base_channels)

        self.up1b = nn.Sequential(
            DoubleConv(base_channels * 8, base_channels * 4, activation),
            nn.Dropout2d(p=drop_out_rate),
        )
        self.up2b = nn.Sequential(
            DoubleConv(base_channels * 4, base_channels * 2, activation),
            nn.Dropout2d(p=drop_out_rate),
        )
        self.up3b = DoubleConv(base_channels * 2, base_channels, activation)

        self.outc = nn.Conv2d(base_channels, n_hsi, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Combined input tensor of shape (N, n_msi + n_hsi, H,
              W)

        Returns:
            torch.Tensor: Output prediction tensor of shape (N, n_hsi, H, W)
        """
        # x: (N, n_msi + n_hsi, H, W)
        x_msi = x[:, : self.n_msi, :, :]  # (N, n_msi, H, W)
        x_hsi = x[:, self.n_msi : self.n_msi + self.n_hsi, :, :]  # (N, n_hsi, H, W)

        feat_msi = self.branch_msi(x_msi)  # (N, base_channels // 2, H, W)
        feat_hsi = self.branch_hsi(x_hsi)  # (N, base_channels // 2, H, W)

        s1 = torch.cat(
            [feat_msi, feat_hsi], dim=1
        )  # (N, base_channels, H, W)

        s2 = self.down1(s1)  # (N, base_channels * 2, H/2, W/2)
        s3 = self.down2(s2)  # (N, base_channels * 4, H/4, W/4)
        b = self.down3(s3)  # (N, base_channels * 8, H/8, W/8)

        out = self.up1a(b)  # (N, base_channels * 4, H/4, W/4)
        out = torch.cat([out, s3], 1)  # (N, base_channels * 8, H/4, W/4)
        out = self.up1b(out)  # (N, base_channels * 4, H/4, W/4)

        out = self.up2a(out)  # (N, base_channels * 2, H/2, W/2)
        out = torch.cat([out, s2], 1)  # (N, base_channels * 4, H/2, W/2)
        out = self.up2b(out)  # (N, base_channels * 2, H/2, W/2)

        out = self.up3a(out)  # (N, base_channels, H, W)
        out = torch.cat(
            [out, s1], 1
        )  # (N, base_channels * 2, H, W) (Using fused s1!)
        out = self.up3b(out)  # (N, base_channels, H, W)

        res = self.outc(out)  # (N, n_hsi, H, W)

        if self.final_op == "abs":
            return torch.abs(res)
        elif self.final_op == "softplus":
            return F.softplus(res)
        elif self.final_op == "square":
            return res**2
        else:
            return res

import torch
import torch.nn as nn
import torch.nn.functional as F
from .commons_model import NAFBlock


class DualBranchNAFNet(nn.Module):
    """Dual-branch NAFNet (Nonlinear Activation Free Network) architecture

    extended for joint multi-modality regression and pixel-wise uncertainty estimation.

    Instead of relying on heavy non-linear activations (ReLU/GELU), NAFNet relies
    on LayerNorm and SimpleGate structures, achieving high performance with lower
    computational overhead. This version outputs both the predicted HSI image
    and its associated log-variance (uncertainty).
    """

    def __init__(
        self,
        n_msi: int = 12,
        n_hsi: int = 230,
        out_channels: int = 230,
        width: int = 64,
        middle_blk_num: int = 1,
        enc_blk_nums: list = [],
        dec_blk_nums: list = [],
        drop_out_rate: float = 0.0,
        final_op: str = "abs",
        return_uncertainty: bool = True,
        **usl_kwargs,
    ):
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
        self.return_uncertainty = return_uncertainty

        if final_op not in ["identity", "abs", "softplus", "square"]:
            raise ValueError(
                "final_op must be 'identity', 'abs', 'softplus', or 'square'"
            )

        # Split width allocation between MSI and HSI introductory branches
        width_msi = width // 2
        width_hsi = width - width_msi  # Handles odd width values cleanly

        # Separate feature intro layers to handle heterogeneous input dimensions
        self.intro_msi = nn.Conv2d(
            in_channels=n_msi,
            out_channels=width_msi,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )
        self.intro_hsi = nn.Conv2d(
            in_channels=n_hsi,
            out_channels=width_hsi,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        # Main network ending head for the target prediction (e.g., HSI reconstruction)
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )

        # Optional secondary ending head to predict pixel-wise log-variance (uncertainty)
        if self.return_uncertainty:
            self.ending_uncertainty = nn.Conv2d(
                in_channels=width,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
            )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Build Encoder-Decoder NAFNet backbone stages
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, drop_out_rate=0.0) for _ in range(num)]
                )
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(chan, drop_out_rate=drop_out_rate)
                for _ in range(middle_blk_num)
            ]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, drop_out_rate=0.0) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(
        self, inp: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            inp (torch.Tensor): Combined input tensor of shape (N, n_msi + n_hsi, H, W)

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]:
                - If return_uncertainty=False: Predicted tensor of shape (N, out_channels, H, W)
                - If return_uncertainty=True: Tuple of (prediction, log_variance) each of shape (N, out_channels, H, W)
        """
        # inp: (N, n_msi + n_hsi, H, W)
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)  # (N, n_msi + n_hsi, H_pad, W_pad)

        # Split input into MSI and HSI components
        x_msi = inp[:, : self.n_msi, :, :]  # (N, n_msi, H_pad, W_pad)
        x_hsi = inp[:, self.n_msi :, :, :]  # (N, n_hsi, H_pad, W_pad)

        # Independent feature extraction followed by channel-wise concatenation
        feat_msi = self.intro_msi(x_msi)  # (N, width_msi, H_pad, W_pad)
        feat_hsi = self.intro_hsi(x_hsi)  # (N, width_hsi, H_pad, W_pad)
        x = torch.cat(
            [feat_msi, feat_hsi], dim=1
        )  # (N, width, H_pad, W_pad)

        encs = []

        # Encoder path
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)  # Downsamples spatial dimensions by 2

        # Bottleneck processing
        x = self.middle_blks(x)

        # Decoder path with skip-connection additions
        for decoder, up, enc_skip in zip(
            self.decoders, self.ups, encs[::-1]
        ):
            x = up(x)  # Upsamples spatial dimensions by 2
            x = x + enc_skip  # Residual skip connection
            x = decoder(x)

        # Final predictions cropped back to the original image dimensions (H, W)
        pred_feat = self.ending(x)  # (N, out_channels, H_pad, W_pad)
        pred_feat = pred_feat[:, :, :H, :W]  # (N, out_channels, H, W)

        # Apply final activation to the main prediction
        if self.final_op == "abs":
            prediction = torch.abs(pred_feat)
        elif self.final_op == "softplus":
            prediction = F.softplus(pred_feat)
        elif self.final_op == "square":
            prediction = pred_feat**2
        else:
            prediction = pred_feat

        if self.return_uncertainty:
            unc_feat = self.ending_uncertainty(x)  # (N, out_channels, H_pad, W_pad)
            log_var = unc_feat[:, :, :H, :W]  # (N, out_channels, H, W)
            return prediction, log_var

        return prediction

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pads input tensor to match the required network downsampling alignment."""
        _, _, h, w = x.size()
        mod_pad_h = (
            self.padder_size - h % self.padder_size
        ) % self.padder_size
        mod_pad_w = (
            self.padder_size - w % self.padder_size
        ) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x