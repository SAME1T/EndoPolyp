import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.conv2 = ConvBNReLU(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SwinUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        img_size: int = 512,
    ):
        super().__init__()
        # Swin encoder
        self.encoder = create_model(
            backbone,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=img_size,
        )
        self.chs = self.encoder.feature_info.channels()  # örn: [96, 192, 384, 768]
        c1, c2, c3, c4 = self.chs

        # Decoder
        self.lateral4 = ConvBNReLU(c4, 512, k=1, s=1, p=0)
        self.dec3 = UpBlock(512, c3, 256)
        self.dec2 = UpBlock(256, c2, 128)
        self.dec1 = UpBlock(128, c1, 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _to_nchw_with_expected(self, t: torch.Tensor, expected_c: int) -> torch.Tensor:
        """
        Eğer tensör [B,H,W,C] ise ve C==expected_c, NCHW'ye çevir.
        Zaten [B,C,H,W] ise ve C==expected_c, dokunma.
        """
        if t.ndim != 4:
            raise RuntimeError(f"Beklenmeyen tensör boyutu: {t.shape}")
        # NCHW mi?
        if t.shape[1] == expected_c:
            return t
        # NHWC mi?
        if t.shape[-1] == expected_c:
            return t.permute(0, 3, 1, 2).contiguous()
        # Fallback: çoğunlukla NHWC'yi NCHW'ye çevir
        return t.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        feats = self.encoder(x)   # [c1,c2,c3,c4]
        # Her seviyeyi beklenen kanal sayısına göre güvenli biçimde NCHW'ye getir
        c1 = self._to_nchw_with_expected(feats[0], self.chs[0])
        c2 = self._to_nchw_with_expected(feats[1], self.chs[1])
        c3 = self._to_nchw_with_expected(feats[2], self.chs[2])
        c4 = self._to_nchw_with_expected(feats[3], self.chs[3])

        x = self.lateral4(c4)     # (B,512,H/32,W/32)
        x = self.dec3(x, c3)      # (B,256,H/16,W/16)
        x = self.dec2(x, c2)      # (B,128,H/8 ,W/8)
        x = self.dec1(x, c1)      # (B, 64,H/4 ,W/4)
        x = self.head(x)          # (B,num_classes,H/4,W/4)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return x
