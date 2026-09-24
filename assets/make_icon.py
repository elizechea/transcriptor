"""Genera assets/icon.png, icon.ico (Windows) e icon.icns (macOS) (onda de audio sobre un documento)."""

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
S = 512


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((16, 16, S - 16, S - 16), radius=110, fill=(31, 78, 121))
    # Onda de audio
    bars = [0.25, 0.5, 0.8, 0.55, 0.95, 0.6, 0.35]
    w, gap, x = 34, 20, 70
    for h in bars:
        half = h * 120
        d.rounded_rectangle((x, 190 - half, x + w, 190 + half), radius=17, fill=(120, 200, 255))
        x += w + gap
    # Líneas de texto
    for i, width in enumerate((372, 300, 340)):
        y = 355 + i * 42
        d.rounded_rectangle((70, y, 70 + width, y + 20), radius=10, fill=(255, 255, 255))
    return img


if __name__ == "__main__":
    img = draw()
    img.save(HERE / "icon.png")
    img.save(HERE / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                       (128, 128), (256, 256)])
    img.resize((1024, 1024), Image.LANCZOS).save(HERE / "icon.icns")
