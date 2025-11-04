import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Albumentations ile dönüşümler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------
# PolypDataset: image-mask çiftlerini okur ve augment/normalize eder.
# Maske 0/255 gelebilir → 0/1'e çevrilir (float32).
# ---------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transforms(img_size=(512, 512), is_train=True):
    h, w = int(img_size[0]), int(img_size[1])
    if is_train:
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

class PolypDataset(Dataset):
    def __init__(self, pairs, img_size=(512, 512), is_train=True, transform=None):
        """
        pairs: [(img_path, mask_path), ...]
        """
        self.items = pairs
        self.img_size = img_size
        self.is_train = is_train
        self.tfm = transform if transform is not None else build_transforms(img_size, is_train)

    def __len__(self):
        return len(self.items)

    def _read_image(self, p):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, p):
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Maske okunamadı: {p}")
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        # 0/255 → 0/1
        m = (m > 0).astype("float32")
        return m

    def __getitem__(self, idx):
        img_p, msk_p = self.items[idx]
        img = self._read_image(img_p)
        msk = self._read_mask(msk_p)

        # Albumentations birlikte image & mask
        out = self.tfm(image=img, mask=msk)
        img_t = out["image"]                      # (3,H,W) float32 normalized
        msk_t = out["mask"].unsqueeze(0)          # (1,H,W) float32 {0,1}
        return {
            "image": img_t,
            "mask":  msk_t,
            "path":  str(img_p),
        }
