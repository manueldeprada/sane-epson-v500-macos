"""Invert linear RGB negatives with separate film response, white balance, and tone rendering.

Version 2 uses fixed per-channel exponents for a roll, separate shadow-color and highlight-WB
measurements, and one achromatic final black offset. A common shoulder preserves RGB ratios.
Neutral rectangles can replace automatic white balance. They use normalized coordinates in the
cropped negative: (left, top, right, bottom). No sharpening or spatial image filtering is applied.

Version 1 recipes remain reproducible through apply(). The default ratios are a starting point
evaluated on this V500 roll, whose stock is believed to be ColorPlus 200. They are not an ICC
profile or a measured characterization of every ColorPlus roll. A `color-profile.json` next to the
negatives overrides the defaults.
"""
import json

import numpy as np

K = 1.4
BASE_PERCENTILE = 99.9
BLACK_PERCENTILE = 0.3
WHITE_PERCENTILE = 99.7
CROP = 50        # pixels trimmed from every edge of the 6400 dpi frame
ROTATE = 1       # quarter turns counterclockwise: film sits sideways in the holder
RED_GAIN = 0.95  # slightly cooler than the gray-world balance (chosen by eye on frame 10)
BLUE_GAIN = 1.05
SATURATION = 1.2


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _trim(neg, crop):
    return neg[crop:-crop, crop:-crop] if crop else neg


def measure_legacy(neg, k=K, crop=CROP):
    """The recipe (a plain dict of floats) for a negative, measured on its cropped pixels."""
    neg = _trim(neg, crop)
    sub = neg[::8, ::8].astype(np.float32) / 65535.0
    base = np.percentile(sub.reshape(-1, 3), BASE_PERCENTILE, axis=0)
    dsub = -np.log10(np.clip(sub / base, 1e-4, 1.0))
    dflat = dsub.reshape(-1, 3)
    lo = np.percentile(dflat, BLACK_PERCENTILE, axis=0)
    hi = np.percentile(dflat, WHITE_PERCENTILE, axis=0)
    floor = 10 ** (k * (lo - hi))
    dl = np.clip((10 ** (k * (dsub - hi)) - floor) / (1 - floor), 0, 1)
    mean = dl.reshape(-1, 3).mean(0)
    gains = (mean[1] / mean).astype(np.float32)
    return {
        'k': float(k), 'crop_px': int(crop),
        'film_base': [float(x) for x in base],
        'density_black': [float(x) for x in lo],
        'density_white': [float(x) for x in hi],
        'gray_world_gains': [float(x) for x in gains],
        'red_gain': RED_GAIN, 'blue_gain': BLUE_GAIN, 'saturation': SATURATION,
    }


def apply_legacy(neg, recipe, rotate=0):
    """Positive (sRGB-encoded uint16) from a negative and its recipe."""
    neg = _trim(neg, recipe['crop_px'])
    k = recipe['k']
    base = np.array(recipe['film_base'], np.float32)
    lo = np.array(recipe['density_black'], np.float32)
    hi = np.array(recipe['density_white'], np.float32)
    gains = np.array(recipe['gray_world_gains'], np.float32)
    gains = gains * np.array([recipe.get('red_gain', 1.0), 1.0, recipe.get('blue_gain', 1.0)], np.float32)
    sat = recipe.get('saturation', 1.0)
    floor = 10 ** (k * (lo - hi))
    H, W = neg.shape[:2]
    out = np.empty((H, W, 3), np.uint16)
    step = 512
    for y in range(0, H, step):
        blk = neg[y:y + step].astype(np.float32) / 65535.0
        d = -np.log10(np.clip(blk / base, 1e-4, 1.0))
        lin = (10 ** (k * (d - hi)) - floor) / (1 - floor)
        enc = srgb(np.clip(lin * gains, 0, 1)).astype(np.float32)
        if sat != 1.0:
            luma = (0.2126 * enc[..., 0] + 0.7152 * enc[..., 1] + 0.0722 * enc[..., 2])[..., None]
            enc = np.clip(luma + sat * (enc - luma), 0, 1)
        out[y:y + step] = np.rint(enc * 65535).astype(np.uint16)
    if rotate:
        out = np.ascontiguousarray(np.rot90(out, rotate))
    return out


def convert(neg, k=1.7, crop=CROP, rotate=ROTATE, log=print):
    """Return an sRGB-encoded uint16 positive. `neg` is a uint16 (H, W, 3) array."""
    recipe = measure(neg, k=k, crop=crop)
    log('  ' + describe(recipe))
    return apply(neg, recipe, rotate=rotate)


def recipe_to_json_legacy(recipe, path, negative_name=None):
    """Write the recipe with the formula, so any program can redo the conversion."""
    base = np.array(recipe['film_base']); lo = np.array(recipe['density_black']); hi = np.array(recipe['density_white'])
    doc = {
        'format': 'v500-negative-recipe/1',
        'negative': negative_name,
        'input': '16-bit linear RGB TIFF (gamma 1.0), transmittance T = value / 65535',
        'crop_px': recipe['crop_px'],
        'k': recipe['k'],
        'film_base': recipe['film_base'],
        'density_black': recipe['density_black'],
        'density_white': recipe['density_white'],
        'gray_world_gains': recipe['gray_world_gains'],
        'red_gain': recipe.get('red_gain', 1.0),
        'blue_gain': recipe.get('blue_gain', 1.0),
        'saturation': recipe.get('saturation', 1.0),
        'derived': {
            'transmittance_at_black_point': [float(x) for x in base * 10 ** (-lo)],
            'transmittance_at_white_point': [float(x) for x in base * 10 ** (-hi)],
        },
        'formula': [
            'D = -log10(clip(T / film_base, 1e-4, 1))',
            'floor = 10^(k (density_black - density_white))',
            'L = (10^(k (D - density_white)) - floor) / (1 - floor)',
            'G = gray_world_gains * (red_gain, 1, blue_gain)',
            'P = srgb_oetf(clip(L * G, 0, 1))',
            'Y = 0.2126 P_r + 0.7152 P_g + 0.0722 P_b;  P = clip(Y + saturation (P - Y), 0, 1)',
        ],
        'output': 'sRGB, 10-bit HEIC (quality 0.9) or 16-bit TIFF; all steps are per channel',
    }
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2)
        f.write('\n')


CHANNEL_RATIOS = (1.18, 1.0, 0.74)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
V2_FORMAT = 'v500-negative-recipe/2'


def _check_negative(neg, crop):
    if neg.ndim != 3 or neg.shape[2] != 3 or neg.dtype != np.uint16:
        raise ValueError('expected a uint16 RGB negative')
    if int(crop) != crop or crop < 0 or 2 * crop >= min(neg.shape[:2]):
        raise ValueError('crop must leave a nonempty image')


def _sample(neg):
    """Bound the measurement cost independently of scan resolution."""
    step = max(1, int(np.ceil(max(neg.shape[:2]) / 640)))
    return neg[::step, ::step].astype(np.float32) / 65535.0


def _valid_pixels(t):
    h, w = t.shape[:2]
    margin = max(1, int(min(h, w) * 0.04))
    interior = np.zeros((h, w), bool)
    interior[margin:-margin, margin:-margin] = True
    finite = (t.min(2) > 2 / 65535) & (t.max(2) < 0.98)
    valid = interior & finite
    if valid.sum() < 32:
        valid = finite
    if valid.sum() < 8:
        raise ValueError('too few unclipped pixels to measure a negative')
    return valid


def _inverse(t, reference, exponents):
    return np.power(np.asarray(reference, np.float32) / np.maximum(t, 1 / 65535),
                    np.asarray(exponents, np.float32)).astype(np.float32)


def _region_values(image, regions):
    h, w = image.shape[:2]
    values = []
    for rect in regions:
        if len(rect) != 4 or not np.isfinite(rect).all():
            raise ValueError('neutral rectangle needs four finite coordinates')
        x0, y0, x1, y1 = rect
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('neutral rectangle coordinates must lie in [0, 1]')
        left, top = int(x0 * w), int(y0 * h)
        right, bottom = max(left + 1, int(x1 * w)), max(top + 1, int(y1 * h))
        values.append(image[top:bottom, left:right].reshape(-1, 3))
    return values


def calibrate_channel_ratios(neg, neutral_regions, crop=CROP):
    """Fit a roll's exponents from two neutral patches of different brightness.

    Both surfaces must be achromatic and under the same illumination. Scanner clipping, mixed
    light, and a colored patch invalidate this estimate. Green is the reference exponent.
    """
    _check_negative(neg, crop)
    if len(neutral_regions) != 2:
        raise ValueError('calibration needs two neutral rectangles')
    t = _sample(_trim(neg, crop))
    patches = _region_values(t, neutral_regions)
    if any(np.any((p <= 2 / 65535) | (p >= 0.98)) for p in patches):
        raise ValueError('calibration patches contain clipped scanner values')
    a, b = [np.median(np.log(p), axis=0) for p in patches]
    delta = a - b
    if np.any(np.abs(delta) < 0.15) or not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError('calibration patches need more brightness separation in every channel')
    ratios = delta[1] / delta
    if np.any((ratios < 0.3) | (ratios > 3)):
        raise ValueError('patches imply implausible channel ratios')
    return ratios.tolist()


def _auto_white_balance(lin, valid, percentile=None, selection_lin=None):
    """Use a mode of candidate neutral highlights, with a gray-edge disagreement diagnostic.

    This is a statistical estimate, not a semantic recognition of white objects. A colored
    surface or mixed lighting can defeat both estimators. confidence is a heuristic, not a
    calibrated probability. Gray-edge is used for the diagnostic, not to force scene averages.
    """
    from scipy.ndimage import gaussian_filter

    initial = 1 / np.maximum(np.percentile(lin[valid], 99.5, axis=0), 1e-8)
    rgb = lin * initial
    lum = np.sum(rgb * LUMA, axis=2)
    # Broad bright surfaces support a white-patch estimate. A dark-dominated scene needs a
    # wider brightness range, because its brightest pixels may be only flash-lit skin or lamps.
    # This describes the histogram, not a semantic classification of daylight versus flash.
    if selection_lin is None:
        selection_lum = lum
    else:
        selection_rgb = selection_lin / np.maximum(np.percentile(selection_lin[valid], 99.5, axis=0), 1e-8)
        selection_lum = np.sum(selection_rgb * LUMA, axis=2)
    distribution = np.percentile(selection_lum[valid], [50, 99])
    median_ratio = float(distribution[0] / max(distribution[1], 1e-8))
    if percentile is None:
        percentile = float(55 + 30 * np.clip((median_ratio - 0.20) / 0.10, 0, 1))
    mx, mn = rgb.max(2), rgb.min(2)
    candidates = valid & (lum > np.percentile(lum[valid], percentile))
    candidates &= (mx / np.maximum(mn, 1e-8) < 2.5)
    candidates &= lum < np.percentile(lum[valid], 99.8)
    log_rgb = np.log(np.maximum(rgb, 1e-8))
    q = log_rgb[..., (0, 2)] - log_rgb[..., 1, None]
    warnings = []
    if candidates.sum() < 32:
        return initial, dict(method='upper-percentile fallback', confidence=0.0,
                             candidate_percentile=percentile, median_to_highlight_ratio=median_ratio,
                             candidate_fraction=float(candidates.mean()),
                             warnings=['No supported neutral highlight cluster. Choose a neutral patch.'])

    histogram, edges = np.histogramdd(q[candidates], bins=(64, 64), range=((-1, 1), (-1, 1)))
    histogram = gaussian_filter(histogram, 1.5)
    if histogram.max() <= 0:
        return initial, dict(method='upper-percentile fallback', confidence=0.0,
                             candidate_percentile=percentile, median_to_highlight_ratio=median_ratio,
                             candidate_fraction=float(candidates.mean()),
                             warnings=['Neutral candidates are outside the automatic WB search range.'])
    ij = np.unravel_index(histogram.argmax(), histogram.shape)
    center = np.array([(edges[c][ij[c]] + edges[c][ij[c] + 1]) / 2 for c in range(2)])
    selected = candidates & (np.linalg.norm(q - center, axis=2) < 0.16)
    if selected.sum() < 16:
        return initial, dict(method='upper-percentile fallback', confidence=0.0,
                             candidate_percentile=percentile, median_to_highlight_ratio=median_ratio,
                             candidate_fraction=float(candidates.mean()),
                             warnings=['Neutral cluster has too few pixels. Choose a neutral patch.'])
    cast = np.median(q[selected], axis=0)
    gain = initial * np.exp(-np.array([cast[0], 0.0, cast[1]]))
    gain /= gain[1]

    # Edge differences are less sensitive to the area occupied by a single color. They remain
    # susceptible to colored edges, so disagreement prompts review instead of an extra correction.
    smooth = gaussian_filter(rgb, (2, 2, 0))
    dy, dx = np.gradient(smooth, axis=0), np.gradient(smooth, axis=1)
    edge = np.sqrt(dx * dx + dy * dy)[valid]
    edge = np.minimum(edge, np.percentile(edge, 99, axis=0))
    illuminant = np.maximum(np.mean(edge ** 3, axis=0) ** (1 / 3), 1e-8)
    edge_gain = initial / illuminant
    edge_gain /= edge_gain[1]
    disagreement = float(np.max(np.abs(np.log(gain / edge_gain))))
    coverage = selected.sum() / candidates.sum()
    confidence = float(min(1.0, coverage / 0.35) * np.exp(-disagreement / 0.35))
    if confidence < 0.65:
        warnings.append('Automatic WB is ambiguous. Check a neutral surface, especially under mixed light.')
    return gain, dict(method='neutral highlight mode', confidence=confidence,
                      candidate_percentile=percentile, median_to_highlight_ratio=median_ratio,
                      candidate_fraction=float(candidates.mean()),
                      cluster_fraction=float(coverage), gray_edge_disagreement_log=disagreement,
                      warnings=warnings)


def measure(neg, k=1.7, crop=CROP, *, channel_ratios=CHANNEL_RATIOS, neutral_regions=(),
            wb_gains=None, exposure_ev=0.0, saturation=1.25, tone_gamma=1.03,
            black_reference_strength=1.0, wb_percentile=None, method='v2',
            wb_prior=None, wb_prior_limit=0.20, color_model='oklab', vibrance=0.10,
            shadows_ev=0.0, hue_adjustments=(), local_wb=(), tone_contrast=1.12,
            contrast_pivot=0.18, shoulder=0.88, shoulder_model='rational'):
    """Measure a reproducible recipe, optionally using known neutral surfaces for white balance."""
    _check_negative(neg, crop)
    if method == 'v1':
        return measure_legacy(neg, k, crop)
    if method != 'v2':
        raise ValueError('method must be v1 or v2')
    ratios = np.asarray(channel_ratios, np.float32)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError('channel_ratios must contain three positive finite values')
    if not np.isfinite([k, exposure_ev, saturation, tone_gamma, black_reference_strength,
                        wb_prior_limit, vibrance, shadows_ev, tone_contrast,
                        contrast_pivot, shoulder]).all():
        raise ValueError('rendering controls must be finite')
    if (k <= 0 or saturation < 0 or tone_gamma <= 0 or abs(exposure_ev) > 20
            or wb_prior_limit < 0 or not 0 <= vibrance <= 1 or abs(shadows_ev) > 3):
        raise ValueError('invalid rendering controls')
    if not .5 <= tone_contrast <= 2 or not 0 < contrast_pivot < 1 or not 0 < shoulder < 1:
        raise ValueError('invalid tone contrast, pivot, or shoulder')
    if shoulder_model not in ('rational', 'exponential'):
        raise ValueError('unsupported highlight shoulder')
    if color_model not in ('oklab', 'encoded-rgb'):
        raise ValueError('color_model must be oklab or encoded-rgb')
    for adjustment in hue_adjustments:
        if (len(adjustment) != 3 or not np.isfinite(adjustment).all()
                or not 0 < adjustment[1] <= 180 or abs(adjustment[2]) > 60):
            raise ValueError('hue adjustments need center, width, and shift in degrees')
    for adjustment in local_wb:
        rect = adjustment['rect']
        gains = np.asarray(adjustment['gains'])
        if (len(rect) != 4 or not np.isfinite(rect).all()
                or not 0 <= rect[0] < rect[2] <= 1 or not 0 <= rect[1] < rect[3] <= 1
                or gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0)
                or not 0 < adjustment['feather'] <= 1):
            raise ValueError('local WB needs a normalized rectangle, positive gains, and feather')
    if not 0 <= black_reference_strength <= 1:
        raise ValueError('black_reference_strength must lie in [0, 1]')
    if wb_percentile is not None and (not np.isfinite(wb_percentile) or not 0 <= wb_percentile <= 99.5):
        raise ValueError('wb_percentile must lie in [0, 99.5] or be None for automatic selection')
    t = _sample(_trim(neg, crop))
    valid = _valid_pixels(t)
    reference = np.percentile(t[valid], 1, axis=0)
    exponents = k * ratios
    lin = _inverse(t, reference, exponents)
    selection_lin = lin
    clear = np.percentile(t[valid], 99.5, axis=0)
    pedestal = _inverse(clear, reference, exponents) * black_reference_strength
    lin = np.maximum(lin - pedestal, 0)
    if wb_gains is not None and neutral_regions:
        raise ValueError('choose either wb_gains or neutral_regions')
    if neutral_regions:
        values = _region_values(t, neutral_regions)
        if any(np.any((p <= 2 / 65535) | (p >= 0.98)) for p in values):
            raise ValueError('neutral patch contains clipped scanner values')
        patches = _region_values(lin, neutral_regions)
        white = np.mean([np.median(np.log(np.maximum(p, 1e-8)), axis=0) for p in patches], axis=0)
        gain = np.exp(white[1] - white)
        diagnostic = dict(method='neutral rectangles', confidence=None, warnings=[])
    elif wb_gains is not None:
        gain = np.asarray(wb_gains, np.float32)
        if gain.shape != (3,) or not np.isfinite(gain).all() or np.any(gain <= 0):
            raise ValueError('wb_gains must contain three positive finite values')
        gain = gain / gain[1]
        diagnostic = dict(method='manual gains', confidence=None, warnings=[])
    else:
        gain, diagnostic = _auto_white_balance(lin, valid, wb_percentile, selection_lin)
        if wb_prior is not None:
            if not np.allclose(wb_prior['exponents'], exponents, rtol=1e-5):
                raise ValueError('WB prior needs matching film exponents')
            coefficients = np.asarray(wb_prior['coefficients'], np.float64)
            if coefficients.shape != (3,) or np.any(coefficients <= 0) or not np.isfinite(coefficients).all():
                raise ValueError('invalid WB prior coefficients')
            prior_gain = coefficients / reference ** exponents
            prior_gain /= prior_gain[1]
            delta = np.log(gain / prior_gain)[[0, 2]]
            distance = float(np.linalg.norm(delta))
            # A roll reference constrains unsupported changes of illuminant caused by sky,
            # foliage, or a single colored garment. Explicit neutrals bypass this constraint.
            if distance > wb_prior_limit:
                delta *= wb_prior_limit / distance
                gain = prior_gain * np.exp([delta[0], 0, delta[1]])
                diagnostic['method'] += ' with roll prior'
            diagnostic['prior_distance_log'] = distance
            diagnostic['prior_limit_log'] = wb_prior_limit
    lin *= gain
    lum = np.sum(lin * LUMA, axis=2)
    black = float(np.percentile(lum[valid], 0.5))
    # One luminance black point and one exposure leave the WB and film channel ratios intact.
    high = float(np.percentile(np.maximum(lum[valid] - black, 0), 99))
    exposure = float(0.95 / max(high, 1e-8) * 2 ** exposure_ev)
    interior = _trim(t, max(1, int(min(t.shape[:2]) * 0.04)))
    outer_base = np.percentile(t.reshape(-1, 3), 99.7, axis=0)
    inner_base = np.percentile(interior.reshape(-1, 3), 99.7, axis=0)
    extra_clipping = np.mean(t.max(2) >= 0.98) - np.mean(interior.max(2) >= 0.98)
    if np.max(outer_base / np.maximum(inner_base, 1e-6)) > 1.8 or extra_clipping > 0.01:
        diagnostic['warnings'].append('Bright scan border excluded from automatic measurements.')
    return dict(format=V2_FORMAT, process='density-v2', crop_px=int(crop), k=float(k),
                channel_ratios=ratios.tolist(), exponents=exponents.tolist(),
                reference_transmittance=reference.tolist(), scanner_black_rgb=pedestal.tolist(),
                black_transmittance=clear.tolist(), black_reference_strength=float(black_reference_strength),
                wb_gains=gain.tolist(),
                neutral_regions=[list(r) for r in neutral_regions], black_luminance=black,
                exposure=exposure, exposure_ev=float(exposure_ev), tone_gamma=float(tone_gamma),
                shoulder=float(shoulder), shoulder_model=shoulder_model,
                tone_contrast=float(tone_contrast), contrast_pivot=float(contrast_pivot),
                saturation=float(saturation), color_model=color_model,
                vibrance=float(vibrance), shadows_ev=float(shadows_ev), wb=diagnostic,
                hue_adjustments=[list(a) for a in hue_adjustments],
                local_wb=[dict(rect=list(a['rect']), gains=list(a['gains']), feather=float(a['feather'])) for a in local_wb],
                warnings=list(diagnostic['warnings']))


_OK_M1 = np.array([[.4122214708, .5363325363, .0514459929],
                   [.2119034982, .6806995451, .1073969566],
                   [.0883024619, .2817188376, .6299787005]], np.float32)
_OK_M2 = np.array([[.2104542553, .7936177850, -.0040720468],
                   [1.9779984951, -2.4285922050, .4505937099],
                   [.0259040371, .7827717662, -.8086757660]], np.float32)
_OK_I1 = np.linalg.inv(_OK_M1)
_OK_I2 = np.linalg.inv(_OK_M2)


def linear_to_oklab(rgb):
    return np.cbrt(rgb @ _OK_M1.T) @ _OK_M2.T


def oklab_to_linear(lab):
    return ((lab @ _OK_I2.T) ** 3) @ _OK_I1.T


def _perceptual_color(rgb, saturation, vibrance, hue_adjustments=()):
    """Adjust chroma at fixed Oklab hue/lightness, then fit gamut along that hue.

    Binary search is restricted to pixels outside sRGB. Neutral pixels remain neutral;
    vibrance has less effect on colors that already have high chroma.
    """
    lab = linear_to_oklab(rgb)
    chroma = np.linalg.norm(lab[..., 1:], axis=2)
    if hue_adjustments:
        hue = np.arctan2(lab[..., 2], lab[..., 1])
        shift = np.zeros_like(hue)
        for center, width, degrees in hue_adjustments:
            difference = (np.degrees(hue) - center + 180) % 360 - 180
            shift += np.radians(degrees) * np.exp(-.5 * (difference / width) ** 2)
        # Near-neutrals have no useful hue. Fade the adjustment before that singularity.
        hue += shift * (-np.expm1(-(chroma / .025) ** 2))
        lab[..., 1] = chroma * np.cos(hue)
        lab[..., 2] = chroma * np.sin(hue)
    gain = saturation * (1 + vibrance * np.exp(-(chroma / .12) ** 2))
    lab[..., 1:] *= gain[..., None]
    result = oklab_to_linear(lab)
    outside = (result.min(2) < -1e-6) | (result.max(2) > 1)
    if outside.any():
        selected = lab[outside]
        low, high = np.zeros(len(selected), np.float32), np.ones(len(selected), np.float32)
        for _ in range(13):
            mid = (low + high) * .5
            trial = selected.copy()
            trial[:, 1:] *= mid[:, None]
            check = oklab_to_linear(trial)
            valid = (check.min(1) >= 0) & (check.max(1) <= 1)
            low = np.where(valid, mid, low)
            high = np.where(valid, high, mid)
        selected[:, 1:] *= low[:, None]
        result[outside] = oklab_to_linear(selected)
    return np.clip(result, 0, 1)


def _saturate_gamut(enc, amount):
    """Change encoded chroma and fit it within the RGB cube along the same hue direction."""
    y = np.sum(enc * LUMA, axis=2, keepdims=True)
    chroma = (enc - y) * amount
    allowance = np.where(chroma > 0, 1 - y, y)
    bounds = np.min(allowance / np.maximum(np.abs(chroma), 1e-8), axis=2, keepdims=True)
    return np.clip(y + chroma * np.minimum(1, bounds), 0, 1)


def _local_white_balance(rgb, adjustments, row, height, width):
    """Blend optional foreground illuminants in log-gain space with smooth rectangle edges."""
    if not adjustments:
        return rgb
    y = (np.arange(row, row + rgb.shape[0], dtype=np.float32) + .5)[:, None] / height
    x = (np.arange(width, dtype=np.float32) + .5)[None, :] / width
    log_gain = np.zeros_like(rgb)
    for adjustment in adjustments:
        left, top, right, bottom = adjustment['rect']
        feather = adjustment['feather']
        dx = np.clip(np.maximum(left - x, x - right) / feather, 0, 1)
        dy = np.clip(np.maximum(top - y, y - bottom) / feather, 0, 1)
        mask = (1 - dx * dx * (3 - 2 * dx)) * (1 - dy * dy * (3 - 2 * dy))
        log_gain += mask[..., None] * np.log(np.asarray(adjustment['gains'], np.float32))
    return rgb * np.exp(log_gain)


def _highlight_shoulder(peak, knee, model='exponential'):
    above = np.maximum(peak - knee, 0)
    if model == 'rational':
        compressed = knee + (1 - knee) * above / (above + 1 - knee)
    elif model == 'exponential':
        compressed = knee + (1 - knee) * (-np.expm1(-above / (1 - knee)))
    else:
        raise ValueError('unsupported highlight shoulder')
    return np.where(peak <= knee, peak, compressed)


def apply(neg, recipe, rotate=0):
    """Render a v1 or v2 recipe to an sRGB-encoded uint16 RGB array."""
    version = recipe.get('format')
    process = recipe.get('process')
    if version not in (None, 'v500-negative-recipe/1', V2_FORMAT):
        raise ValueError('unsupported negative recipe format')
    if version != V2_FORMAT and process != 'density-v2':
        return apply_legacy(neg, recipe, rotate)
    _check_negative(neg, recipe['crop_px'])
    neg = _trim(neg, recipe['crop_px'])
    h, w = neg.shape[:2]
    out = np.empty((h, w, 3), np.uint16)
    for row in range(0, h, 256):
        t = neg[row:row + 256].astype(np.float32) / 65535.0
        lin = _inverse(t, recipe['reference_transmittance'], recipe['exponents'])
        lin = np.maximum(lin - np.asarray(recipe.get('scanner_black_rgb', [0, 0, 0]), np.float32), 0)
        lin *= np.asarray(recipe['wb_gains'], np.float32)
        # Remove the same achromatic pedestal from all channels. Merely lowering luminance
        # leaves that pedestal in the RGB ratios and desaturates the reconstructed shadows.
        lin = np.maximum(lin - recipe['black_luminance'], 0)
        lin = _local_white_balance(lin, recipe.get('local_wb', ()), row, h, w)
        lum = np.sum(lin * LUMA, axis=2)
        lifted = lum * recipe['exposure']
        if recipe.get('shadows_ev', 0):
            lifted *= np.exp2(recipe['shadows_ev'] * np.exp(-lifted / .20))
        lifted = np.power(lifted, recipe['tone_gamma'])
        if recipe.get('tone_contrast', 1) != 1:
            pivot = recipe.get('contrast_pivot', .18)
            lifted = pivot * np.power(lifted / pivot, recipe['tone_contrast'])
        lin *= (lifted / np.maximum(lum, 1e-8))[..., None]
        peak = lin.max(2)
        knee = recipe['shoulder']
        mapped = _highlight_shoulder(peak, knee, recipe.get('shoulder_model', 'exponential'))
        lin *= (mapped / np.maximum(peak, 1e-8))[..., None]
        if recipe.get('color_model') == 'oklab':
            enc = srgb(_perceptual_color(lin, recipe['saturation'], recipe.get('vibrance', 0),
                                         recipe.get('hue_adjustments', ())))
        else:
            enc = _saturate_gamut(srgb(lin), recipe['saturation'])
        out[row:row + 256] = np.rint(enc * 65535).astype(np.uint16)
    if rotate:
        out = np.ascontiguousarray(np.rot90(out, rotate))
    return out


def describe(recipe):
    if recipe.get('process') != 'density-v2':
        return 'v1 film base %s, gray-world gains %s' % (
            np.round(recipe['film_base'], 4), np.round(recipe['gray_world_gains'], 3))
    mode = recipe['wb']['method']
    if 'candidate_percentile' in recipe['wb']:
        mode += ', percentile cutoff %.1f' % recipe['wb']['candidate_percentile']
    return 'v2 exponents %s, WB %s (%s)' % (
        np.round(recipe['exponents'], 3), np.round(recipe['wb_gains'], 3), mode)


def transfer_white_balance(source_recipe, target_recipe):
    """Transfer one scene's WB between negatives made with the same film and scanner settings.

    The numerical inverse references differ between frames. Transfer the coefficients of
    T^-exponent, not the raw gains, so a change of reference does not change the illuminant.
    A common illuminant is a user/editor assumption, not something this function infers.
    """
    exponent = np.asarray(source_recipe['exponents'], np.float64)
    if not np.allclose(exponent, target_recipe['exponents'], rtol=1e-6):
        raise ValueError('WB transfer needs the same channel exponents')
    source = np.asarray(source_recipe['reference_transmittance'], np.float64)
    target = np.asarray(target_recipe['reference_transmittance'], np.float64)
    gain = np.exp(np.log(source_recipe['wb_gains']) + exponent * np.log(source / target))
    return (gain / gain[1]).tolist()


def recipe_to_json(recipe, path, negative_name=None):
    if recipe.get('process') != 'density-v2':
        return recipe_to_json_legacy(recipe, path, negative_name)
    doc = dict(recipe)
    doc.update(negative=negative_name,
               input='16-bit linear scanner RGB TIFF; T = value / 65535',
               output='sRGB-encoded RGB; no sharpening; no scanner ICC characterization',
               formula=[
                   'R = (reference_transmittance / max(T, 1/65535)) ^ exponents',
                   'U = max(R - scanner_black_rgb, 0) * wb_gains',
                   'A = max(U - black_luminance, 0); Y = dot(A, (0.2126, 0.7152, 0.0722))',
                   'V = (Y * exposure) ^ tone_gamma',
                   'RGB = A * V / max(Y, 1e-8)',
                   'M = max(RGB); S = M if M <= shoulder, otherwise shoulder + (1-shoulder)*(1-exp(-(M-shoulder)/(1-shoulder)))',
                   'P = srgb_oetf(RGB * S / max(M, 1e-8))',
                   'Increase chroma about encoded luma by saturation; reduce chroma along the same direction to fit [0,1]',
               ])
    if recipe.get('color_model') == 'oklab':
        doc['formula'][-1] = ('Convert display-linear RGB to Oklab; multiply a,b by saturation * '
                              '(1 + vibrance * exp(-(chroma/0.12)^2)); reduce a,b together at fixed '
                              'L and hue until sRGB is in gamut; encode sRGB')
        doc['formula'].insert(-1, 'Apply optional Gaussian hue-sector rotations in Oklab, fading to zero at neutral')
        doc['formula'].insert(4, 'Before tone_gamma: lifted *= 2^(shadows_ev * exp(-lifted/0.20))')
    if recipe.get('local_wb'):
        doc['formula'].insert(3, 'Apply optional local WB: A *= exp(sum(smooth_rectangle_mask * log(local RGB gains)))')
    if 'tone_contrast' in recipe:
        index = next(i for i, text in enumerate(doc['formula']) if text.startswith('RGB = A'))
        doc['formula'].insert(index, 'After tone_gamma: V = contrast_pivot * (V / contrast_pivot)^tone_contrast')
    if recipe.get('shoulder_model') == 'rational':
        doc['formula'] = [
            'M = max(RGB); A = max(M - shoulder, 0); S = M if M <= shoulder, otherwise shoulder + (1-shoulder)*A/(A+1-shoulder)'
            if text.startswith('M = max(RGB);') else text for text in doc['formula']]
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.write('\n')
