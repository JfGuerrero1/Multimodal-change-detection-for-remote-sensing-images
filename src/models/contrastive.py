import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Standard Residual Block with two 3x3 convolutions and a shortcut connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H_out, W_out)
        """
        # x: (N, in_channels, H, W)
        identity = self.shortcut(x)  # (N, out_channels, H_out, W_out)
        out = self.conv1(x)  # (N, out_channels, H_out, W_out)
        out = self.bn1(out)  # (N, out_channels, H_out, W_out)
        out = self.relu(out)  # (N, out_channels, H_out, W_out)
        out = self.conv2(out)  # (N, out_channels, H_out, W_out)
        out = self.bn2(out)  # (N, out_channels, H_out, W_out)
        out += identity  # (N, out_channels, H_out, W_out)
        out = self.relu(out)  # (N, out_channels, H_out, W_out)
        return out


class AdaptedResNet(nn.Module):
    """Adapted ResNet Encoder-Decoder architecture for pixel-wise dense

    representation learning.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        out_channels: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # ================================================================
        # ENCODER
        # ================================================================
        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # Resolution: H x W
        self.layer1 = nn.Sequential(
            ResBlock(base_channels, base_channels, stride=1),
            ResBlock(base_channels, base_channels),
        )

        # Resolution: H/2 x W/2
        self.layer2 = nn.Sequential(
            ResBlock(base_channels, base_channels * 2, stride=2),
            ResBlock(base_channels * 2, base_channels * 2),
        )

        # Resolution: H/4 x W/4
        self.layer3 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels * 4, stride=2),
            ResBlock(base_channels * 4, base_channels * 4),
        )

        # Resolution: H/8 x W/8
        self.layer4 = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 8, stride=2),
            ResBlock(base_channels * 8, base_channels * 8),
        )

        # ================================================================
        # DECODER
        # Progressive recovery back to original resolution
        # ================================================================
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(
                base_channels * 8,
                base_channels * 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Final pixel-wise embedding layer
        self.output = nn.Conv2d(
            base_channels, out_channels, kernel_size=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        # x: (N, in_channels, H, W)
        input_size = x.shape[2:]

        # ================================================================
        # ENCODING
        # ================================================================
        x = self.conv1(x)  # (N, base_channels, H, W)
        x = self.bn1(x)  # (N, base_channels, H, W)
        x = self.relu(x)  # (N, base_channels, H, W)

        x = self.layer1(x)  # (N, base_channels, H, W)
        x = self.layer2(x)  # (N, base_channels * 2, H/2, W/2)
        x = self.layer3(x)  # (N, base_channels * 4, H/4, W/4)
        x = self.layer4(x)  # (N, base_channels * 8, H/8, W/8)

        # ================================================================
        # DECODING
        # ================================================================
        # H/8 -> H/4
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # (N, base_channels * 8, H/4, W/4)
        x = self.decoder_conv1(x)  # (N, base_channels * 4, H/4, W/4)

        # H/4 -> H/2
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # (N, base_channels * 4, H/2, W/2)
        x = self.decoder_conv2(x)  # (N, base_channels * 2, H/2, W/2)

        # H/2 -> H
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # (N, base_channels * 2, H, W)
        x = self.decoder_conv3(x)  # (N, base_channels, H, W)

        # ================================================================
        # EXACT RESOLUTION GUARANTEE
        # ================================================================
        if x.shape[2:] != input_size:
            x = F.interpolate(
                x, size=input_size, mode="bilinear", align_corners=False
            )  # (N, base_channels, H, W)

        # ================================================================
        # PIXEL-WISE EMBEDDING
        # ================================================================
        x = self.output(x)  # (N, out_channels, H, W)

        return x


class SiameseNetwork_heterogene(nn.Module):
    """Heterogeneous Siamese Network for processing two different modalities

    (e.g., MSI and HSI).
    """

    def __init__(
        self,
        in_channel_msi: int,
        in_channel_hsi: int,
        base_channels: int = 64,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.encoder_msi = AdaptedResNet(
            in_channel_msi, base_channels, latent_dim
        )
        self.encoder_hsi = AdaptedResNet(
            in_channel_hsi, base_channels, latent_dim
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x1 (torch.Tensor): First modality input tensor of shape (N,
              in_channel_msi, H, W)
            x2 (torch.Tensor): Second modality input tensor of shape (N,
              in_channel_hsi, H, W)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Embeddings (z1, z2) of shape (N,
            latent_dim, H, W)
        """
        # x1: (N, in_channel_msi, H, W), x2: (N, in_channel_hsi, H, W)
        z1 = self.encoder_msi(x1)  # (N, latent_dim, H, W)
        z2 = self.encoder_hsi(x2)  # (N, latent_dim, H, W)
        return z1, z2

#################################################################################
#not tested
class SiameseNetwork_homogene(nn.Module):
    """Homogeneous Siamese Network for processing two inputs of the same

    modality.
    """

    def __init__(
        self,
        in_channel_hsi: int,
        base_channels: int = 64,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.encoder_msi = AdaptedResNet(
            in_channel_hsi, base_channels, latent_dim
        )
        self.encoder_hsi = AdaptedResNet(
            in_channel_hsi, base_channels, latent_dim
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x1 (torch.Tensor): First input tensor of shape (N, in_channel_hsi, H,
              W)
            x2 (torch.Tensor): Second input tensor of shape (N, in_channel_hsi,
              H, W)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Embeddings (z1, z2) of shape (N,
            latent_dim, H, W)
        """
        # x1: (N, in_channel_hsi, H, W), x2: (N, in_channel_hsi, H, W)
        z1 = self.encoder_msi(x1)  # (N, latent_dim, H, W)
        z2 = self.encoder_hsi(x2)  # (N, latent_dim, H, W)
        return z1, z2


class MeanTeacher(nn.Module):
    """Mean Teacher framework for semi-supervised representation learning

    consisting of a student network and a teacher network updated via EMA.
    """

    def __init__(
        self, in_channels: int, base_channels: int = 64, latent_dim: int = 128
    ):
        super().__init__()
        self.student = AdaptedResNet(in_channels, base_channels, latent_dim)
        self.teacher = AdaptedResNet(in_channels, base_channels, latent_dim)

        # Freeze teacher parameters (no gradient updates via backpropagation)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(self, alpha: float = 0.99):
        """Updates teacher parameters using Exponential Moving Average (EMA)."""
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                teacher_param.data.mul_(alpha).add_(
                    student_param.data, alpha=(1.0 - alpha)
                )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Student and Teacher embeddings
            (z_student, z_teacher) of shape (N, latent_dim, H, W)
        """
        # x: (N, in_channels, H, W)
        z_student = self.student(x)  # (N, latent_dim, H, W)
        z_teacher = self.teacher(x)  # (N, latent_dim, H, W)
        return z_student, z_teacher