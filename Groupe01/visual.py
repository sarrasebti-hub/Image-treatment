"""Fonctions pixel partagées (hash perceptuel, couleurs nommées)."""
import numpy as np
from collections import Counter
from PIL import Image


def dhash(img: Image.Image, size: int = 8) -> str:
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)  # 64 bits ; distance = Hamming


def named_colors(rgb_small: np.ndarray) -> dict:
    """Répartition des pixels dans 11 couleurs nommées (règles HSV) -> attributs visuels sans modèle."""
    x = rgb_small.reshape(-1, 3) / 255.0
    mx, mn = x.max(1), x.min(1)
    v, d = mx, mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    h = np.zeros(len(x))
    m = d > 1e-9
    rc = np.where(m & (mx == r), ((g - b) / np.maximum(d, 1e-9)) % 6, 0)
    gc = np.where(m & (mx == g) & (mx != r), (b - r) / np.maximum(d, 1e-9) + 2, 0)
    bc = np.where(m & (mx == b) & (mx != r) & (mx != g), (r - g) / np.maximum(d, 1e-9) + 4, 0)
    h = (rc + gc + bc) * 60.0
    name = np.empty(len(x), dtype=object)
    name[:] = "gray"
    name[v < 0.18] = "black"
    chroma = (s >= 0.25) & (v >= 0.18)
    name[(s < 0.12) & (v > 0.85)] = "white"
    for lo, hi, nm in [(0, 15, "red"), (15, 40, "orange"), (40, 70, "yellow"), (70, 170, "green"), (170, 260, "blue"),
                       (260, 300, "purple"), (300, 345, "pink"), (345, 361, "red")]:
        name[chroma & (h >= lo) & (h < hi)] = nm
    brown = chroma & (h >= 15) & (h < 45) & (v < 0.6)
    name[brown] = "brown"
    cnt = Counter(name)
    tot = len(x)
    return {k: round(c / tot, 4) for k, c in cnt.most_common()}
