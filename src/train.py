import argparse
import os
from pathlib import Path
import random
import yaml
import glob
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- import yolunu garantiye almak için ----
import sys
ROOT = Path(__file__).resolve().parents[1]          # .../EndoPolyp
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from models.swin_unet import SwinUNet
from datasets.polyp_dataset import PolypDataset, build_transforms

# -------------------------------
# Basit BCE+DICE loss ve metrikler
# -------------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice = 1.0 - dice_coeff(probs, targets, smooth=self.smooth)  # (1 - dice)
        return bce + dice

@torch.no_grad()
def dice_coeff(prob, target, smooth=1.0):
    # prob, target: (B,1,H,W) float
    prob = (prob > 0.5).float()
    target = (target > 0.5).float()
    inter = (prob * target).sum(dim=(1,2,3))
    union = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + smooth) / (union + smooth)
    return dice.mean()

@torch.no_grad()
def iou_coeff(prob, target, smooth=1.0):
    prob = (prob > 0.5).float()
    target = (target > 0.5).float()
    inter = (prob * target).sum(dim=(1,2,3))
    union = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou.mean()

# -------------------------------
# Yardımcılar
# -------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def find_mask_for_image(img_path: Path, masks_dir: Path) -> Path | None:
    """Aynı 'stem' adına sahip maskeyi 'masks' klasöründe ara."""
    stem = img_path.stem
    for ext in IMG_EXTS:
        cand = masks_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    # *_mask.png vb. adlandırmalar için en yakın eşleşmeyi dene
    gl = list(masks_dir.glob(f"{stem}*"))
    return gl[0] if gl else None

def collect_pairs(data_root: Path, images_subdirs: list[str], masks_subdirs: list[str]) -> list[tuple[Path, Path]]:
    pairs = []
    for img_sub, msk_sub in zip(images_subdirs, masks_subdirs):
        img_dir = (data_root / img_sub).resolve()
        msk_dir = (data_root / msk_sub).resolve()
        assert img_dir.exists(), f"Görüntü klasörü yok: {img_dir}"
        assert msk_dir.exists(), f"Maske klasörü yok: {msk_dir}"

        imgs = []
        for ext in IMG_EXTS:
            imgs.extend(img_dir.rglob(f"*{ext}"))
        imgs = sorted(imgs)

        for ip in imgs:
            mp = find_mask_for_image(ip, msk_dir)
            if mp is None:
                alt = Path(str(ip).replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep))
                if alt.exists():
                    mp = alt
            if mp is None:
                print(f"[Uyarı] Mask bulunamadı, atlandı: {ip}")
                continue
            pairs.append((ip, mp))
    return pairs

def split_pairs(pairs, val_ratio=0.2, seed=42):
    rnd = random.Random(seed)
    pairs = pairs.copy()
    rnd.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val

# -------------------------------
# Eğitim
# -------------------------------
import contextlib

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)
        gts  = batch["mask"].to(device, non_blocking=True)

        # GPU'da channels_last belleğe geç (hız için)
        if device.type == "cuda":
            imgs = imgs.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        use_amp = (device.type == "cuda")
        ctx = torch.autocast('cuda', dtype=torch.float16) if use_amp else contextlib.nullcontext()
        with ctx:
            logits = model(imgs)
            loss = loss_fn(logits, gts)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / max(1, len(loader))



@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    d_all, i_all = [], []
    pbar = tqdm(loader, desc="Valid", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)
        gts  = batch["mask"].to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        d = dice_coeff(probs, gts).item()
        i = iou_coeff(probs, gts).item()
        d_all.append(d); i_all.append(i)
        pbar.set_postfix(dice=f"{d:.4f}", iou=f"{i:.4f}")
    return float(np.mean(d_all)), float(np.mean(i_all))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",       type=str, default=str(ROOT / "configs" / "data_polyp.yaml"))
    ap.add_argument("--train",      type=str, default=str(ROOT / "configs" / "train_swin_unet.yaml"))
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--val_ratio",  type=float, default=0.2)  # %20 doğrulama
    ap.add_argument("--early_stop_patience", type=int, default=0)  # 0 = kapalı
    ap.add_argument("--val_every", type=int, default=1)  # her epoch doğrula (1). Hızı artırmak için 2-3 yap.
    args = ap.parse_args()

    # ---------------- load configs ----------------
    data_cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(Path(args.train).read_text(encoding="utf-8"))

    data_root = (ROOT / data_cfg["data_root"]).resolve()
    images_sub = data_cfg["train_images"]
    masks_sub  = data_cfg["train_masks"]
    img_size   = data_cfg.get("image_size", [512, 512])

    epochs     = train_cfg["train"]["epochs"]
    batch_size = train_cfg["train"]["batch_size"]
    lr         = train_cfg["train"]["lr"]
    wd         = train_cfg["train"]["weight_decay"]
    model_name = train_cfg["model"]["name"]
    pretrained = train_cfg["model"].get("pretrained", True)
    in_ch      = train_cfg["model"].get("in_channels", 3)
    num_cls    = train_cfg["model"].get("num_classes", 1)
    model_img = int(train_cfg["model"].get("img_size", 512))


    # ---------------- seed ----------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------- data ----------------
    all_pairs = collect_pairs(data_root, images_sub, masks_sub)
    assert all_pairs, "Hiç (görüntü, maske) çifti bulunamadı!"
    tr_pairs, va_pairs = split_pairs(all_pairs, val_ratio=args.val_ratio, seed=args.seed)

    print(f"Toplam çift: {len(all_pairs)} | Train: {len(tr_pairs)} | Val: {len(va_pairs)} (val_ratio={args.val_ratio})")

    tfm_tr = build_transforms(img_size, is_train=True)
    tfm_va = build_transforms(img_size, is_train=False)
    ds_tr = PolypDataset(tr_pairs, img_size, is_train=True, transform=tfm_tr)
    ds_va = PolypDataset(va_pairs, img_size, is_train=False, transform=tfm_va)

   

    # ---------------- model ----------------
    assert model_name.lower() == "swin_unet", "Şimdilik yalnızca 'swin_unet' destekleniyor."
    model = SwinUNet(in_channels=in_ch, num_classes=num_cls, pretrained=pretrained, img_size=model_img)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # --- GPU hız ayarları ---
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True           # conv autotune
        torch.set_float32_matmul_precision("high")      # matmul hız
        print(">> CUDA ENABLED | cuDNN benchmark ON")
    else:
        print(">> CPU MODE")

    # --- DataLoader hız ayarları ---
    max_w = os.cpu_count() or 4
    num_workers = max(2, min(8, max_w - 1))
    persist = num_workers > 0
    pin = (device.type == "cuda")

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        prefetch_factor=(2 if persist else None),
        drop_last=True
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        prefetch_factor=(2 if persist else None)
    )
    print(f">> DataLoader workers={num_workers} | prefetch={(2 if persist else 0)} | pin_memory={pin}")

      
    # ---------------- optim ----------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    loss_fn = BCEDiceLoss()

    # ---------------- save dirs ----------------
    weights_dir = ROOT / "runs" / "weights"
    logs_dir    = ROOT / "runs" / "logs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    patience = args.early_stop_patience
    wait = 0


    best_dice = -1.0
    start = time.time()
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        tr_loss = train_one_epoch(model, dl_tr, optimizer, scaler, device, loss_fn)

        do_val = (epoch % args.val_every == 0)
        if do_val:
            va_dice, va_iou = validate(model, dl_va, device)
            print(f"train_loss: {tr_loss:.4f}  |  val_dice: {va_dice:.4f}  val_iou: {va_iou:.4f}")
        else:
            va_dice, va_iou = -1.0, -1.0
            print(f"train_loss: {tr_loss:.4f}  |  (val atlandı, val_every={args.val_every})")


        start = time.time()
        print(f"train_loss: {tr_loss:.4f}  |  val_dice: {va_dice:.4f}  val_iou: {va_iou:.4f}")
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"[GPU] peak_mem={peak_gb:.2f} GB")
        print(f"[timing] epoch_time={(time.time()-start):.1f}s")
        start = time.time()


        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"[GPU] peak_mem={peak_gb:.2f} GB")
        print(f"[timing] epoch_time={(time.time()-start):.1f}s")
        start = time.time()


        torch.save({"model": model.state_dict()}, weights_dir / "last.pt")
        if do_val and va_dice > best_dice:
            best_dice = va_dice
            torch.save({"model": model.state_dict()}, weights_dir / "best.pt")
            print(f"[+] best.pt güncellendi (dice={best_dice:.4f})")
            wait = 0
        elif do_val:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"[EarlyStopping] {patience} epoch iyileşme yok, duruyoruz.")
                break



    dur = time.time() - start
    print(f"\nEğitim tamamlandı. Süre: {dur/60:.1f} dk | En iyi Dice: {best_dice:.4f}")
    print(f"Kaydedildi: {weights_dir/'best.pt'}  ve  {weights_dir/'last.pt'}")

if __name__ == "__main__":
    main()
