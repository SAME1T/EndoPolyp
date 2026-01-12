# -*- coding: utf-8 -*-
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
import yaml
from typing import Optional, List, Tuple

import numpy as np
import torch
from PIL import Image

import gradio as gr

# opsiyonel: strict mod için component filtreleme (cv2 varsa kullanır)
try:
    import cv2
except Exception:
    cv2 = None


# ----------------------------
#  Gradio uyumluluk: theme desteklemeyen sürümler için
# ----------------------------
def make_blocks(css: str):
    """
    Eski Gradio sürümleri theme parametresini desteklemez.
    Yeni sürümde theme kullanır, eskide otomatik kaldırır.
    """
    try:
        if hasattr(gr, "themes") and hasattr(gr.themes, "Soft"):
            return gr.Blocks(theme=gr.themes.Soft(), css=css)
    except TypeError:
        pass
    except Exception:
        pass
    return gr.Blocks(css=css)


# ----------------------------
#  Yol ayarları
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]   # .../EndoPolyp
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from models.swin_unet import SwinUNet
from datasets.polyp_dataset import build_transforms


# ----------------------------
#  Config + model yükleme
# ----------------------------
def load_configs():
    data_cfg_path  = ROOT / "configs" / "data_polyp.yaml"
    train_cfg_path = ROOT / "configs" / "train_swin_unet.yaml"

    with open(data_cfg_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    return data_cfg, train_cfg


def build_model(train_cfg, device):
    model_cfg = train_cfg["model"]
    name      = model_cfg.get("name", "swin_unet")
    in_ch     = model_cfg.get("in_channels", 3)
    num_cls   = model_cfg.get("num_classes", 1)
    img_size  = int(model_cfg.get("img_size", 512))

    assert name.lower() == "swin_unet", "Şimdilik sadece Swin-UNet destekli."

    model = SwinUNet(
        in_channels=in_ch,
        num_classes=num_cls,
        img_size=img_size,
        pretrained=False
    )
    model.to(device)
    return model


def load_model_and_transforms():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg, train_cfg = load_configs()

    img_size = data_cfg.get("image_size", [512, 512])
    if isinstance(img_size, int):
        img_size = [img_size, img_size]

    tfm_val = build_transforms(img_size, is_train=False)
    model = build_model(train_cfg, device)

    weights_path = ROOT / "runs" / "weights" / "best.pt"
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)

    model.load_state_dict(state_dict)
    model.eval()

    # hız
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return model, tfm_val, device, img_size


MODEL, TFM_VAL, DEVICE, IMG_SIZE = load_model_and_transforms()


# ----------------------------
#  Yardımcılar (boyut güvenli)
# ----------------------------
def safe_rgb(img: np.ndarray) -> np.ndarray:
    """Gradio bazen (H,W), bazen (H,W,4) getirebilir. Burada güvenle RGB yapıyoruz."""
    if img is None:
        return img
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def preprocess_image(np_img_rgb: np.ndarray) -> torch.Tensor:
    transformed = TFM_VAL(image=np_img_rgb)
    img_tensor = transformed["image"]
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.to(DEVICE)


def resize_float_map(prob_map: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    prob_map = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    im = Image.fromarray(prob_map)  # mode 'F'
    im = im.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.array(im).astype(np.float32)


def resize_mask_nearest(mask01: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    m = (mask01.astype(np.uint8) * 255)
    im = Image.fromarray(m, mode="L")
    im = im.resize((out_w, out_h), resample=Image.NEAREST)
    return (np.array(im) >= 128).astype(np.uint8)


def make_overlay(orig_rgb: np.ndarray, bin_mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    KRİTİK: Maske boyutu ne olursa olsun overlay boyutuna NEAREST ile oturtulur.
    Böylece boolean index mismatch ASLA olmaz.
    """
    overlay = safe_rgb(orig_rgb).copy()
    h, w = overlay.shape[:2]

    m = bin_mask01
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != (h, w):
        m = resize_mask_nearest(m, w, h)

    idx = (m == 1)
    if not idx.any():
        return overlay

    color = np.zeros_like(overlay, dtype=np.uint8)
    color[..., 1] = 255  # yeşil
    overlay[idx] = (alpha * color[idx] + (1.0 - alpha) * overlay[idx]).astype(np.uint8)
    return overlay


def keep_large_components(mask01: np.ndarray, min_area_px: int, keep_largest: bool = True) -> Tuple[np.ndarray, int]:
    """
    Strict mod: küçük parçaları temizle.
    """
    m = mask01.astype(np.uint8)

    if cv2 is None:
        return m, 1 if m.any() else 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m * 0, 0

    comps = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            comps.append((area, i))

    if not comps:
        return m * 0, 0

    if keep_largest:
        _, best_i = max(comps, key=lambda x: x[0])
        out = (labels == best_i).astype(np.uint8)
        return out, 1

    out = np.zeros_like(m, dtype=np.uint8)
    for _, i in comps:
        out[labels == i] = 1
    return out, len(comps)


# ----------------------------
#  PDF: Türkçe karakter fix + amblem
# ----------------------------
_PDF_FONT_NAME = "EndoPolypFontTR"
_PDF_FONT_READY = False


def _find_ttf_font() -> Optional[str]:
    candidates: List[Path] = []

    # Proje içinde font koyarsan buradan da yakalar:
    candidates += [
        ROOT / "assets" / "fonts" / "DejaVuSans.ttf",
        ROOT / "assets" / "fonts" / "NotoSans-Regular.ttf",
    ]

    # Windows fontları
    win_font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates += [
        win_font_dir / "arial.ttf",
        win_font_dir / "Arial.ttf",
        win_font_dir / "calibri.ttf",
        win_font_dir / "Calibri.ttf",
        win_font_dir / "segoeui.ttf",
        win_font_dir / "SegoeUI.ttf",
        win_font_dir / "arialuni.ttf",
        win_font_dir / "ARIALUNI.TTF",
    ]

    # Linux (olursa)
    candidates += [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    ]

    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _ensure_pdf_font():
    global _PDF_FONT_READY
    if _PDF_FONT_READY:
        return

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        _PDF_FONT_READY = False
        return

    ttf_path = _find_ttf_font()
    if ttf_path is None:
        _PDF_FONT_READY = False
        return

    try:
        pdfmetrics.registerFont(TTFont(_PDF_FONT_NAME, ttf_path))
        _PDF_FONT_READY = True
    except Exception:
        _PDF_FONT_READY = False


def _pdf_font_name() -> str:
    return _PDF_FONT_NAME if _PDF_FONT_READY else "Helvetica"


def try_make_pdf(orig_rgb: np.ndarray, overlay_rgb: np.ndarray, info_lines: List[str]) -> Optional[str]:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.lib import colors
    except Exception:
        return None

    _ensure_pdf_font()

    tmpdir = tempfile.mkdtemp(prefix="endopolyp_report_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(tmpdir, f"EndoPolyp_Rapor_{ts}.pdf")

    orig_path = os.path.join(tmpdir, f"orig_{ts}.png")
    ov_path   = os.path.join(tmpdir, f"overlay_{ts}.png")
    Image.fromarray(safe_rgb(orig_rgb)).save(orig_path)
    Image.fromarray(safe_rgb(overlay_rgb)).save(ov_path)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4

    font_title = _pdf_font_name()
    font_body  = _pdf_font_name()

    # Üst şerit
    c.setFillColor(colors.HexColor("#0b1220"))
    c.rect(0, H - 90, W, 90, stroke=0, fill=1)

    # Amblem
    cx, cy = 55, H - 45
    c.setFillColor(colors.white)
    c.circle(cx, cy, 18, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#ef4444"))
    c.rect(cx - 4, cy - 12, 8, 24, stroke=0, fill=1)
    c.rect(cx - 12, cy - 4, 24, 8, stroke=0, fill=1)

    # Başlık
    c.setFillColor(colors.white)
    c.setFont(font_title, 16)
    c.drawString(85, H - 55, "EndoPolyp - Klinik Destek Raporu")

    c.setFont(font_body, 10)
    c.drawString(85, H - 73, f"Tarih/Saat: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")

    # İçerik kutusu
    c.setFillColor(colors.HexColor("#111827"))
    c.roundRect(35, H - 330, W - 70, 220, 12, stroke=0, fill=1)

    c.setFillColor(colors.white)
    c.setFont(font_body, 11)

    y = H - 125
    for line in info_lines:
        c.drawString(55, y, line)
        y -= 18
        if y < H - 310:
            break

    # Görseller
    img_w = 250
    img_h = 180

    c.setFont(font_body, 11)
    c.setFillColor(colors.black)
    c.drawString(55, 255, "Orijinal")
    c.drawString(325, 255, "Overlay (Tahmin)")

    c.drawImage(ImageReader(orig_path), 55, 55, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
    c.drawImage(ImageReader(ov_path),   325, 55, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")

    # Dipnot
    c.setFont(font_body, 9)
    c.setFillColor(colors.HexColor("#374151"))
    c.drawString(55, 35, "Not: Bu rapor klinik karar yerine geçmez. Nihai değerlendirme hekimindir.")

    c.showPage()
    c.save()
    return pdf_path


# ----------------------------
#  Tahmin (UI)
# ----------------------------
def predict_for_ui(image: np.ndarray, threshold: float, strict_mode: bool):
    if image is None:
        return None, "### Bilgi Formu\nGörüntü yüklenmedi.", None

    orig = safe_rgb(image)
    orig_h, orig_w = orig.shape[:2]

    # Model output genelde 512x512 olur (transform resize)
    with torch.inference_mode():
        inp = preprocess_image(orig)
        logits = MODEL(inp)
        probs_512 = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    # KRİTİK: prob_map'i ORİJİNAL BOYUTA taşı
    prob_map = probs_512
    if prob_map.shape != (orig_h, orig_w):
        prob_map = resize_float_map(prob_map, orig_w, orig_h)

    thr = float(threshold)
    bin_mask = (prob_map >= thr).astype(np.uint8)

    # strict mod ile FP azalt
    max_prob_global = float(prob_map.max()) if prob_map.size else 0.0
    kept_count = 1 if bin_mask.any() else 0

    if strict_mode:
        min_area_px = max(250, int(0.0018 * (orig_h * orig_w)))
        bin_mask, kept_count = keep_large_components(bin_mask, min_area_px=min_area_px, keep_largest=True)

        # ek güvenlik kapıları (FP azaltmak için)
        if max_prob_global < 0.75:
            bin_mask[:] = 0
            kept_count = 0
        if float(bin_mask.mean()) < 0.0012:
            bin_mask[:] = 0
            kept_count = 0

    overlay = make_overlay(orig, bin_mask, alpha=0.45)

    # istatistik
    area_ratio = float(bin_mask.mean())
    if bin_mask.any():
        mean_prob = float(prob_map[bin_mask == 1].mean())
        max_prob  = float(prob_map[bin_mask == 1].max())
    else:
        mean_prob = 0.0
        max_prob  = float(prob_map.max()) if prob_map.size else 0.0

    if not bin_mask.any():
        sonuc = "Polip şüphesi YOK"
        yorum = "Maske bulunamadı (katı mod açıksa filtrelenmiş olabilir)."
    else:
        if area_ratio < 0.01 and max_prob < 0.80:
            sonuc = "ŞÜPHELİ (düşük-orta)"
            yorum = "Küçük odak(lar) tespit edildi, teyit önerilir."
        else:
            sonuc = "Polip şüphesi VAR"
            yorum = "Odak(lar) belirgin, doktor değerlendirmesi önerilir."

    info_md = f"""
### Bilgi Formu

**Sonuç:** {sonuc}  
**Yorum:** {yorum}

**Eşik (threshold):** {thr:.2f}  
**Katı mod:** {"Açık" if strict_mode else "Kapalı"}  
**Cihaz:** {DEVICE}

**Tahmini maske alanı:** %{area_ratio * 100:.2f}  
**Maske içi ortalama olasılık:** %{mean_prob * 100:.1f}  
**Maske içi maksimum olasılık:** %{max_prob * 100:.1f}  
**Odak sayısı (tahmini):** {kept_count}

> Not: Bu sistem bir destek aracıdır. Klinik karar için doktor değerlendirmesi esastır.
"""

    info_lines = [
        f"Sonuç: {sonuc}",
        f"Yorum: {yorum}",
        f"Eşik (threshold): {thr:.2f}",
        f"Katı mod: {'Açık' if strict_mode else 'Kapalı'}",
        f"Cihaz: {DEVICE}",
        f"Maske alanı (%): {area_ratio * 100:.2f}",
        f"Maske ort. olasılık (%): {mean_prob * 100:.1f}",
        f"Maske maks. olasılık (%): {max_prob * 100:.1f}",
        f"Odak sayısı (tahmini): {kept_count}",
    ]

    pdf_path = try_make_pdf(orig, overlay, info_lines)
    if pdf_path is None:
        info_md += "\n\n⚠️ PDF üretimi için: `pip install reportlab`"

    return overlay, info_md, pdf_path


# ----------------------------
#  UI (Blocks)
# ----------------------------
CUSTOM_CSS = """
:root {
  --bg: #0b1220;
  --card: rgba(255,255,255,.06);
  --text: rgba(255,255,255,.92);
  --muted: rgba(255,255,255,.70);
}
.gradio-container {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(249,115,22,.18), transparent 60%),
              radial-gradient(800px 500px at 80% 20%, rgba(14,165,233,.16), transparent 60%),
              var(--bg) !important;
  color: var(--text) !important;
}
#topbar {
  background: linear-gradient(90deg, rgba(255,255,255,.10), rgba(255,255,255,.04));
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px;
  padding: 18px 18px;
  display:flex;
  align-items:center;
  gap:14px;
  margin-bottom: 14px;
}
.card {
  background: var(--card);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px;
  padding: 14px;
}
.small {
  color: var(--muted);
  font-size: 12px;
  margin-top: 6px;
}
"""

HEADER_HTML = """
<div id="topbar">
  <div style="width:42px;height:42px;border-radius:14px;background:rgba(255,255,255,.12);display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,255,255,.14)">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" fill="rgba(255,255,255,.92)"/>
      <path d="M11 6h2v12h-2z" fill="#ef4444"/>
      <path d="M6 11h12v2H6z" fill="#ef4444"/>
    </svg>
  </div>
  <div style="flex:1">
    <div style="font-size:20px;font-weight:800;letter-spacing:.2px">EndoPolyp - Klinik Destek Ekranı</div>
    <div class="small">Swin-UNet tabanlı polip segmentasyonu • Sonuç overlay + rapor (PDF)</div>
  </div>
  <div style="font-size:12px;padding:6px 10px;border-radius:999px;background:rgba(239,68,68,.14);border:1px solid rgba(239,68,68,.25)">⚕️ Sağlık / Hastane</div>
</div>
"""


def main():
    with make_blocks(CUSTOM_CSS) as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown("#### Girdi", elem_classes="small")
                    inp = gr.Image(type="numpy", image_mode="RGB", label="Girdi görüntü (endoskopi)")
                    thr = gr.Slider(0.3, 0.9, value=0.55, step=0.05, label="Eşik (threshold)")
                    strict = gr.Checkbox(value=True, label="Katı mod (yanlış pozitif azalt)")
                    with gr.Row():
                        btn = gr.Button("Analiz Et")
                        clr = gr.Button("Temizle")
                    gr.Markdown(
                        "🔒 **Gizlilik:** Görüntüler sadece yerelde işlenir.\n\n"
                        "⚠️ **Uyarı:** Bu sistem destek amaçlıdır; klinik karar hekimindir.",
                        elem_classes="small"
                    )

            with gr.Column(scale=6):
                with gr.Group():
                    out_img = gr.Image(type="numpy", label="Sonuç (Overlay)")
                with gr.Group():
                    info = gr.Markdown()
                    pdf = gr.File(label="PDF Raporu")

        def _clear():
            return None, "### Bilgi Formu\n—", None

        btn.click(
            fn=predict_for_ui,
            inputs=[inp, thr, strict],
            outputs=[out_img, info, pdf]
        )
        clr.click(
            fn=_clear,
            inputs=[],
            outputs=[out_img, info, pdf]
        )

    demo.launch()


if __name__ == "__main__":
    main()
