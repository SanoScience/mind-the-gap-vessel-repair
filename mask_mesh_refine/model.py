from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet3D(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc0 = ConvBlock(in_channels, c)
        self.enc1 = ConvBlock(c, c * 2)
        self.enc2 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.dec2 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec1 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec0 = ConvBlock(c * 2 + c, c)
        self.out = nn.Conv3d(c, 1, kernel_size=1)

    @staticmethod
    def _down(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=tuple(ref.shape[2:]), mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self._down(e0))
        e2 = self.enc2(self._down(e1))
        b = self.bottleneck(self._down(e2))
        d2 = self._up_to(b, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._up_to(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = self._up_to(d1, e0)
        d0 = self.dec0(torch.cat([d0, e0], dim=1))
        return self.out(d0)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))
    inter = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def batch_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits) >= 0.5
    target_b = target >= 0.5
    dims = tuple(range(1, pred.dim()))
    inter = torch.sum((pred & target_b).float(), dim=dims)
    denom = torch.sum(pred.float(), dim=dims) + torch.sum(target_b.float(), dim=dims)
    return ((2.0 * inter + eps) / (denom + eps)).mean()


class RefinerLightningModule(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        lr: float = 1e-3,
        lambda_bce: float = 0.5,
        lambda_dice: float = 1.0,
        crop_margin_mm: float = 16.0,
        sdf_clip_mm: float = 16.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = SmallUNet3D(in_channels=in_channels, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x = batch["x"].float()
        y = batch["y"].float()
        logits = self(x)
        bce = F.binary_cross_entropy_with_logits(logits, y)
        dice_loss = soft_dice_loss(logits, y)
        loss = float(self.hparams.lambda_bce) * bce + float(self.hparams.lambda_dice) * dice_loss
        dice = batch_dice(logits.detach(), y.detach())
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        self.log(f"{stage}_bce", bce, on_step=False, on_epoch=True)
        self.log(f"{stage}_dice_loss", dice_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_dice", dice, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=float(self.hparams.lr))
