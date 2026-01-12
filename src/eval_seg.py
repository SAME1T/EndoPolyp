import argparse
import os
from pathlib import Path
import yaml
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------
# import yolunu ayarla (train.py ile aynı mantık)
# ---------------------------------
import sys
ROOT = Path(__file__).resolve().parents[1]     # .../EndoPolyp
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# Projede zaten var olan bileşenler
from models.swin_unet import SwinUNet
from datasets.polyp_dataset import PolypDataset, build_transforms
from train import collect_pairs, split_pairs, dice_coeff, iou_coeff


def build_model(train_cfg, device):
    """
    Eğitimde kullandığın model yapılandırmasını okuyup,
    aynı Swin-UNet mimarisini tekrar kuruyoruz.
    Burada pretrained=False diyorum, çünkü ağırlıkları
    zaten .pt dosyasından yükleyeceğiz.
    """
    model_name = train_cfg["model"]["name"]
    in_ch      = train_cfg["model"].get("in_channels", 3)
    num_cls    = train_cfg["model"].get("num_classes", 1)
    model_img  = int(train_cfg["model"].get("img_size", 512))

    assert model_name.lower() == "swin_unet", "Şimdilik sadece SwinUNet bekliyoruz."

    model = SwinUNet(
        in_channels=in_ch,
        num_classes=num_cls,
        pretrained=False,      # Ağırlıkları checkpoint'ten alacağız
        img_size=model_img
    )
    model.to(device)
    return model

def make_blocks(css: str):
    """
    Eski Gradio sürümleri theme parametresini desteklemez.
    Bu fonksiyon yeni sürümde theme kullanır, eskide otomatik kapatır.
    """
    # Yeni gradio + theme varsa dene
    try:
        if hasattr(gr, "themes") and hasattr(gr.themes, "Soft"):
            return gr.Blocks(theme=gr.themes.Soft(), css=css)
    except TypeError:
        pass
    except Exception:
        pass

    # Eski gradio fallback
    return gr.Blocks(css=css)


def main():
    # ----------------- argümanlar -----------------
    ap = argparse.ArgumentParser(description="Swin-UNet polip segmentasyon değerlendirme script'i")
    ap.add_argument("--data",      type=str, default=str(ROOT / "configs" / "data_polyp.yaml"),
                    help="Train/val için kullanılan veri config'i")
    ap.add_argument("--train_cfg", type=str, default=str(ROOT / "configs" / "train_swin_unet.yaml"),
                    help="Model ve eğitim hyperparametrelerini içeren config")
    ap.add_argument("--weights",   type=str, default=str(ROOT / "runs" / "weights" / "best.pt"),
                    help="Değerlendirilecek ağırlık dosyası (genelde best.pt)")
    ap.add_argument("--seed",      type=int, default=42,
                    help="Train.py ile aynı seed (aynı val split için)")
    ap.add_argument("--val_ratio", type=float, default=0.2,
                    help="Train.py ile aynı val oranı")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Değerlendirme batch size (val'de genelde küçük olabilir)")
    args = ap.parse_args()

    # ----------------- configleri yükle -----------------
    data_cfg  = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(Path(args.train_cfg).read_text(encoding="utf-8"))

    data_root  = (ROOT / data_cfg["data_root"]).resolve()
    images_sub = data_cfg["train_images"]
    masks_sub  = data_cfg["train_masks"]
    img_size   = data_cfg.get("image_size", [512, 512])

    # ----------------- seed ayarla -----------------
    # Burada amaç: train.py'deki ile *aynı* random sıralama ve split'i elde etmek.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ----------------- veri çiftlerini topla -----------------
    all_pairs = collect_pairs(data_root, images_sub, masks_sub)
    assert all_pairs, "Hiç (görüntü, maske) çifti bulunamadı!"

    tr_pairs, va_pairs = split_pairs(all_pairs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Toplam çift: {len(all_pairs)} | Train: {len(tr_pairs)} | Val/Test: {len(va_pairs)}")

    # Bu script'te sadece val kümesini 'test' gibi kullanacağız.
    # Sebep: Ayrı bir test klasörü ayırmadık. Küçük veri setinde
    # çoğu makalede train/val üzerinden raporlanıyor. İleride istersen
    # ayrıca test klasörü de ayırabiliriz.

    tfm_val = build_transforms(img_size, is_train=False)
    ds_val  = PolypDataset(va_pairs, img_size, is_train=False, transform=tfm_val)

    # DataLoader ayarlarını train.py'ye benzetiyoruz:
    max_w = os.cpu_count() or 4
    num_workers = max(2, min(8, max_w - 1))
    pin = (torch.cuda.is_available())

    dl_val = DataLoader(
        ds_val,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None)
    )
    print(f">> Eval DataLoader | batch_size={args.batch_size} | num_workers={num_workers} | pin_memory={pin}")

    # ----------------- cihaz & model -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Device: {device}")

    model = build_model(train_cfg, device)

    # Ağırlıkları yükle
    ckpt = torch.load(args.weights, map_location=device)
    state_dict = ckpt.get("model", ckpt)   # hem {'model': ...} hem direkt state_dict için güvenli
    model.load_state_dict(state_dict)
    model.eval()

    # ----------------- değerlendirme döngüsü -----------------
    dice_list, iou_list = [], []

    start = time.time()
    with torch.no_grad():
        pbar = tqdm(dl_val, desc="Eval", leave=True)
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            gts  = batch["mask"].to(device, non_blocking=True)

            logits = model(imgs)
            probs  = torch.sigmoid(logits)

            # train.py'deki ile aynı metrik fonksiyonları
            d = dice_coeff(probs, gts).item()
            i = iou_coeff(probs, gts).item()

            dice_list.append(d)
            iou_list.append(i)
            pbar.set_postfix(dice=f"{d:.4f}", iou=f"{i:.4f}")

    mean_dice = float(np.mean(dice_list))
    mean_iou  = float(np.mean(iou_list))
    dur = time.time() - start

    print("\n====== [EVAL SONUCU] ======")
    print(f"Val/Test Dice: {mean_dice:.4f}")
    print(f"Val/Test IoU : {mean_iou:.4f}")
    print(f"Süre: {dur:.1f} sn ({dur/60:.1f} dk)")
    print("===========================\n")


if __name__ == "__main__":
    main()
