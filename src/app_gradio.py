import sys
from pathlib import Path
import yaml

import numpy as np
import torch
from PIL import Image

import gradio as gr  # önce: pip install gradio

# ----------------------------
#  Yol ayarları (train.py ile uyumlu)
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]   # .../EndoPolyp
SRC  = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from models.swin_unet import SwinUNet
from datasets.polyp_dataset import build_transforms


# ----------------------------
#  Config ve model yükleme
# ----------------------------

def load_configs():
    """
    data_polyp.yaml ve train_swin_unet.yaml dosyalarını okuyoruz.
    Amaç: image_size ve model parametrelerini eğitimdeki ile aynı
    olacak şekilde burada da kullanmak.
    """
    data_cfg_path  = ROOT / "configs" / "data_polyp.yaml"
    train_cfg_path = ROOT / "configs" / "train_swin_unet.yaml"

    with open(data_cfg_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    return data_cfg, train_cfg


def build_model(train_cfg, device):
    """
    Eğitimde kullandığın Swin-UNet mimarisini burada tekrar kuruyoruz.
    Ağırlıkları .pt dosyasından yükleyeceğimiz için pretrained=False.
    """
    model_cfg = train_cfg["model"]
    name      = model_cfg.get("name", "swin_unet")
    in_ch     = model_cfg.get("in_channels", 3)
    num_cls   = model_cfg.get("num_classes", 1)
    img_size  = int(model_cfg.get("img_size", 512))

    assert name.lower() == "swin_unet", "Şimdilik sadece Swin-UNet destekliyoruz."

    model = SwinUNet(
        in_channels=in_ch,
        num_classes=num_cls,
        img_size=img_size,
        pretrained=False
    )
    model.to(device)
    return model


def load_model_and_transforms():
    """
    - Cihazı (cuda / cpu) seç
    - configleri oku
    - modeli kur
    - best.pt ağırlıklarını yükle
    - eval moduna al
    - validation için kullandığımız transform'u hazırla
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg, train_cfg = load_configs()

    # data_polyp.yaml içindeki image_size'i al (örn. [512, 512])
    img_size = data_cfg.get("image_size", [512, 512])
    if isinstance(img_size, int):
        img_size = [img_size, img_size]

    # Eğitimde kullandığımız val transform'un aynısını alıyoruz
    tfm_val = build_transforms(img_size, is_train=False)

    model = build_model(train_cfg, device)

    # En iyi modeli yükle
    weights_path = ROOT / "runs" / "weights" / "best.pt"
    ckpt = torch.load(weights_path, map_location=device)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tfm_val, device, img_size


# Global olarak bir kez yükleyelim (her çağrıda tekrar kurulmaması için)
MODEL, TFM_VAL, DEVICE, IMG_SIZE = load_model_and_transforms()


# ----------------------------
#  Yardımcı fonksiyonlar
# ----------------------------

def preprocess_image(np_img: np.ndarray) -> torch.Tensor:
    """
    Gradio'dan gelen görüntü numpy (H, W, 3, uint8) şeklinde gelir (RGB).
    Albumentations tabanlı build_transforms fonksiyonunu kullanarak
    eğitime benzer biçimde boyutlandırma + normalize + tensor çevirme yapıyoruz.

    Dönüş:
        (1, C, H, W) boyutlu torch.Tensor
    """
    # Albumentations, image paramı numpy (H, W, C) bekler
    transformed = TFM_VAL(image=np_img)
    img_tensor = transformed["image"]  # ToTensorV2 varsa CHW tensor döner

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)  # batch boyutu ekle

    return img_tensor.to(DEVICE)


def predict_mask(np_img: np.ndarray, threshold: float = 0.5):
    """
    Tek bir görüntü için:
    - orijinal resmi sakla
    - modeli çalıştır (TFM_VAL ile preprocess)
    - verilen threshold'a göre maske çıkar
    - polip alan oranı ve olasılık istatistiklerini hesapla
    - doktora yönelik kısa bir yorum üret
    """
    # 1) Orijinal görüntü (rapor/sunum için)
    orig = np_img.astype(np.uint8)

    # 2) Modele ver → olasılık haritası
    with torch.no_grad():
        inp = preprocess_image(orig)  # TFM_VAL kullanıyor
        logits = MODEL(inp)
        probs = torch.sigmoid(logits)
        prob_map = probs[0, 0].cpu().numpy()   # (H, W)

    # 3) Eşikleme ile binary maske
    bin_mask = (prob_map >= threshold).astype(np.uint8)  # (H, W)

    # 4) Maske görselleştirme: 0-255 + 3 kanal (Gradio uyumu)
    mask_vis = (bin_mask * 255).astype(np.uint8)
    mask_rgb = np.stack([mask_vis] * 3, axis=-1)  # (H, W, 3)

    # 5) Overlay için orijinali maske boyutuna getir
    H, W = bin_mask.shape
    base = np.array(Image.fromarray(orig).resize((W, H)))
    overlay = base.copy()

    color_mask = np.zeros_like(base)
    color_mask[..., 1] = 255  # yeşil

    alpha = 0.5
    overlay[bin_mask == 1] = (
        alpha * color_mask[bin_mask == 1]
        + (1 - alpha) * overlay[bin_mask == 1]
    ).astype(np.uint8)

    # 6) Alan ve olasılık istatistikleri
    area_ratio = float(bin_mask.mean())  # [0,1] arası
    if bin_mask.any():
        mean_prob = float(prob_map[bin_mask == 1].mean())
        max_prob = float(prob_map[bin_mask == 1].max())
    else:
        mean_prob = 0.0
        max_prob = float(prob_map.max())

    # 7) Basit karar mantığı (tam klinik değil, sadece rehber)
    if area_ratio < 0.005 and max_prob < 0.6:
        decision = "Polip saptanmadı (düşük olasılık)"
    elif area_ratio < 0.02 and max_prob < 0.75:
        decision = "Şüpheli küçük odak (düşük-orta olasılık)"
    else:
        decision = "Polip şüphesi yüksek"

    info_text = (
        f"Polip alanı: {area_ratio * 100:.2f}%\n"
        f"Maske içi ortalama olasılık: {mean_prob:.2f}\n"
        f"Maske içi maksimum olasılık: {max_prob:.2f}\n"
        f"Karar: {decision}"
    )

    return orig, mask_rgb, overlay, info_text




# ----------------------------
#  Gradio arayüzü
# ----------------------------

def gradio_predict(image: np.ndarray, threshold: float):
    """
    Gradio callback'i.
    image: (H, W, 3) RGB
    threshold: Slider'dan gelen eşik değeri
    """
    if image is None:
        return None, None, None, "Görüntü yüklenmedi."

    orig, mask_rgb, overlay, info_text = predict_mask(image, threshold)
    return orig, mask_rgb, overlay, info_text



def main():
    """
    Gradio arayüzünü ayağa kaldırır.
    Tarayıcı üzerinden:
      - input: tek bir endoskopik görüntü
      - output: orijinal, maske, overlay
    """
    title = "EndoPolyp - Swin-UNet Polip Segmentasyonu"
    description = (
        "Eğitilmiş Swin-UNet modelini kullanarak endoskopi görüntülerinde "
        "polip bölgesini otomatik olarak segment eder. "
        "Sol tarafa bir görüntü yükleyin."
    )

    demo = gr.Interface(
        fn=gradio_predict,
        inputs=[
            gr.Image(
                type="numpy",
                label="Girdi görüntü (endoskopi)",
                image_mode="RGB",
            ),
            gr.Slider(
                minimum=0.3,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Eşik (threshold)",
            ),
        ],
        outputs=[
            gr.Image(type="numpy", label="Orijinal"),
            gr.Image(type="numpy", label="Tahmin maskesi (0-255)"),
            gr.Image(type="numpy", label="Overlay (maske bindirilmiş)"),
            gr.Textbox(label="Model Özeti", lines=4),
        ],
        title=title,
        description=description,
        flagging_mode="manual",  # Flag dursun
    )


    demo.launch()


if __name__ == "__main__":
    main()
