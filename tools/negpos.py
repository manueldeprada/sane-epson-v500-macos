"""Color negative to positive conversion (defaults chosen on a ColorPlus 200 frame).

Input: a linear 16-bit RGB negative (gamma 1.0). Steps:
  1. film base per channel = 99.9th percentile of transmittance,
  2. density D = -log10(T / base),
  3. scene-linear value 10^(K (D - D_hi)) per channel, with black and white points at the
     0.3th and 99.7th percentile of D,
  4. gray-world gains (green is the reference), then sRGB encoding.
"""
import numpy as np

K = 1.4
BASE_PERCENTILE = 99.9
BLACK_PERCENTILE = 0.3
WHITE_PERCENTILE = 99.7
CROP = 50        # pixels trimmed from every edge of the 6400 dpi frame
ROTATE = 1       # quarter turns counterclockwise: film sits sideways in the holder


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def convert(neg, k=K, crop=CROP, rotate=ROTATE, log=print):
    """Return an sRGB-encoded uint16 positive. `neg` is a uint16 (H, W, 3) array."""
    if crop:
        neg = neg[crop:-crop, crop:-crop]
    H, W = neg.shape[:2]
    sub = neg[::8, ::8].astype(np.float32) / 65535.0
    flat = sub.reshape(-1, 3)
    base = np.percentile(flat, BASE_PERCENTILE, axis=0)
    dsub = -np.log10(np.clip(sub / base, 1e-4, 1.0))
    dflat = dsub.reshape(-1, 3)
    lo = np.percentile(dflat, BLACK_PERCENTILE, axis=0)
    hi = np.percentile(dflat, WHITE_PERCENTILE, axis=0)
    floor = 10 ** (k * (lo - hi))
    dl = np.clip((10 ** (k * (dsub - hi)) - floor) / (1 - floor), 0, 1)
    mean = dl.reshape(-1, 3).mean(0)
    gains = (mean[1] / mean).astype(np.float32)
    log('  film base %s, gray-world gains %s' % (np.round(base, 4), np.round(gains, 3)))

    out = np.empty((H, W, 3), np.uint16)
    step = 512
    for y in range(0, H, step):
        blk = neg[y:y + step].astype(np.float32) / 65535.0
        d = -np.log10(np.clip(blk / base, 1e-4, 1.0))
        lin = (10 ** (k * (d - hi)) - floor) / (1 - floor)
        lin = np.clip(lin * gains, 0, 1)
        out[y:y + step] = np.rint(srgb(lin) * 65535).astype(np.uint16)
    if rotate:
        out = np.ascontiguousarray(np.rot90(out, rotate))
    return out
