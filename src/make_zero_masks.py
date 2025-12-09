from pathlib import Path
from PIL import Image
import numpy as np

# Proje kökünü bul ( .../EndoPolyp )
ROOT = Path(__file__).resolve().parents[1]

# NonPolyp klasörleri
NONPOLYP_IMG = ROOT / "data" / "raw" / "NonPolyp" / "images"
NONPOLYP_MSK = ROOT / "data" / "raw" / "NonPolyp" / "masks"


def main():
    # Maske klasörü yoksa oluştur
    NONPOLYP_MSK.mkdir(parents=True, exist_ok=True)

    img_paths = list(NONPOLYP_IMG.glob("*.*"))
    print(f"{len(img_paths)} adet non-polyp görüntü bulundu.")

    for img_path in img_paths:
        # Sadece boyut bilgisi için açıyoruz
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # h x w boyutunda tamamen siyah (0) maske
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        mask = Image.fromarray(mask_arr)

        # Çıkış ismi: aynı isim, .png uzantılı
        out_path = NONPOLYP_MSK / (img_path.stem + ".png")
        mask.save(out_path)

    print("Tüm maskeler oluşturuldu.")


if __name__ == "__main__":
    main()
