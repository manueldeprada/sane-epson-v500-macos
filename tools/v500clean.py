#!/usr/bin/env python3
"""Remove dust from a 16-bit RGB film scan using its infrared pass.

    v500clean.py rgb.tif ir.tif cleaned.tif [--mask mask.tif]

The infrared image may have a different pixel grid than the RGB image (the V500 scans
infrared at most at 3200 x 2400 dpi). The infrared dust map is built at its own resolution
and registered to the RGB image by correlating it with dark residuals in the RGB luminance.
Each masked region is then filled in two parts: a biharmonic interpolation of the pixels around
it (edges continue across the hole and no structure is invented), plus film grain taken from a
clean patch with matching surroundings and limited to the grain amplitude measured around the
hole. Needs numpy and scipy.
"""
import argparse
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import splu


# ---------------------------------------------------------------- TIFF input and output
def read_tiff(path):
    """Read an uncompressed 8 or 16-bit TIFF in either byte order. Returns a uint8 or uint16 array."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:2] == b'II':
        e = '<'
    elif data[:2] == b'MM':
        e = '>'
    else:
        raise ValueError('not a TIFF file')
    ifd = struct.unpack_from(e + 'I', data, 4)[0]
    n = struct.unpack_from(e + 'H', data, ifd)[0]
    tags = {}
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8}
    fmts = {1: 'B', 3: 'H', 4: 'I', 2: 'B', 6: 'b', 7: 'B', 8: 'h', 9: 'i', 16: 'Q'}
    for i in range(n):
        tag, typ, cnt = struct.unpack_from(e + 'HHI', data, ifd + 2 + 12 * i)
        size = sizes.get(typ, 1) * cnt
        field = ifd + 2 + 12 * i + 8
        if size <= 4:
            raw = data[field:field + size]
        else:
            off = struct.unpack_from(e + 'I', data, field)[0]
            raw = data[off:off + size]
        if typ in fmts:
            tags[tag] = struct.unpack('%s%d%s' % (e, cnt, fmts[typ]), raw)
    compression = tags.get(259, (1,))[0]
    if compression not in (1, 8, 32946):
        raise ValueError('only uncompressed and Deflate TIFF are supported')
    w, h = tags[256][0], tags[257][0]
    bits = tags[258][0]
    spp = tags.get(277, (1,))[0]
    predictor = tags.get(317, (1,))[0]
    offs, counts = tags[273], tags[279]
    parts = [data[o:o + c] for o, c in zip(offs, counts)]
    if compression != 1:
        parts = [zlib.decompress(p) for p in parts]
    buf = b''.join(parts)
    if bits == 16:
        arr = np.frombuffer(buf, dtype=e + 'u2')[: w * h * spp]
        if e == '>':
            arr = arr.astype('<u2')
        if predictor == 2:
            arr = np.cumsum(arr.reshape(h, w, spp), axis=1, dtype=np.uint16).reshape(-1)
    else:
        arr = np.frombuffer(buf, dtype=np.uint8)[: w * h * spp]
        if predictor == 2:
            arr = np.cumsum(arr.reshape(h, w, spp), axis=1, dtype=np.uint8).reshape(-1)
    return arr.reshape(h, w, spp) if spp > 1 else arr.reshape(h, w)


def write_tiff16(path, arr, compress=True, rows_per_strip=64):
    """Write a 16-bit gray or RGB TIFF. By default the strips are Deflate-compressed with the
    horizontal predictor (lossless, about 30% smaller on film grain)."""
    arr = np.ascontiguousarray(arr.astype('<u2'))
    h, w = arr.shape[:2]
    spp = arr.shape[2] if arr.ndim == 3 else 1
    img = arr.reshape(h, w, spp)
    if not compress:
        rows_per_strip = h
    starts = list(range(0, h, rows_per_strip))

    def encode(y0):
        blk = img[y0:y0 + rows_per_strip]
        if compress:
            d = blk.copy()
            d[:, 1:, :] = blk[:, 1:, :] - blk[:, :-1, :]
            return zlib.compress(d.tobytes(), 6)
        return blk.tobytes()

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        strips = list(pool.map(encode, starts))
    n = len(strips)
    offsets, pos = [], 8
    for st in strips:
        offsets.append(pos)
        pos += len(st) + (len(st) % 2)
    extra = b''
    bits_off = pos
    if spp == 3:
        extra += struct.pack('<3H', 16, 16, 16)
    offs_off = bits_off + len(extra)
    counts_off = offs_off + 4 * n
    ifd_off = counts_off + 4 * n
    extra += struct.pack('<%dI' % n, *offsets) + struct.pack('<%dI' % n, *[len(x) for x in strips])
    entries = []

    def ent(tag, typ, cnt, val):
        entries.append(struct.pack('<HHII', tag, typ, cnt, val))

    ent(256, 4, 1, w)
    ent(257, 4, 1, h)
    ent(258, 3, spp if spp == 3 else 1, bits_off if spp == 3 else 16)
    ent(259, 3, 1, 8 if compress else 1)
    ent(262, 3, 1, 2 if spp == 3 else 1)
    ent(273, 4, n, offs_off if n > 1 else offsets[0])
    ent(277, 3, 1, spp)
    ent(278, 4, 1, rows_per_strip)
    ent(279, 4, n, counts_off if n > 1 else len(strips[0]))
    ent(284, 3, 1, 1)
    if compress:
        ent(317, 3, 1, 2)
    with open(path, 'wb') as f:
        f.write(b'II*\0' + struct.pack('<I', ifd_off))
        for st in strips:
            f.write(st)
            if len(st) % 2:
                f.write(b'\0')
        f.write(extra)
        f.write(struct.pack('<H', len(entries)) + b''.join(entries) + struct.pack('<I', 0))


# ---------------------------------------------------------------- cleaning
def estimate_registration(res, mask_y, mask_x, search_y, search_x):
    """Shift (dy, dx) of the RGB luminance residual that best matches the dark infrared dust map."""
    H, W = res.shape
    best = (9.0, 0, 0)
    for dy in range(-search_y, search_y + 1):
        yy = mask_y - dy
        oky = (yy >= 0) & (yy < H)
        for dx in range(-search_x, search_x + 1):
            xx = mask_x - dx
            ok = oky & (xx >= 0) & (xx < W)
            if ok.sum() < 20:
                continue
            v = float(res[yy[ok], xx[ok]].mean())
            if v < best[0]:
                best = (v, dy, dx)
    return best


def diffusion_base(out, known, ya, yb, xa, xb, order=2, target=None, values=None):
    """Solve the harmonic (order 1) or biharmonic (order 2) equation for every unknown pixel of the
    window [ya:yb, xa:xb] with the known pixels as boundary data. The biharmonic solution continues
    edges across the hole with matching slope but can overshoot, the harmonic one never leaves the
    range of its boundary. Returns (mask_of_unknowns, values) or None when the stencil leaves the
    window."""
    h, w = yb - ya, xb - xa
    unk = ~known[ya:yb, xa:xb]
    if target is not None:
        # unfilled pixels of other regions that sit near the window border keep their current values
        # as boundary data; the stencil would otherwise reach outside the window
        band = np.ones((h, w), bool)
        band[2:-2, 2:-2] = False
        if (target & band).any():
            return None
        unk = unk & ~band
    if not unk.any():
        return None
    idx = -np.ones((h, w), np.int64)
    n = int(unk.sum())
    idx[unk] = np.arange(n)
    uy, ux = np.nonzero(unk)
    if order == 1:
        stencil = [((0, 0), 4.0), ((1, 0), -1.0), ((-1, 0), -1.0), ((0, 1), -1.0), ((0, -1), -1.0)]
    else:
        stencil = [((0, 0), 20.0), ((1, 0), -8.0), ((-1, 0), -8.0), ((0, 1), -8.0), ((0, -1), -8.0),
                   ((1, 1), 2.0), ((1, -1), 2.0), ((-1, 1), 2.0), ((-1, -1), 2.0),
                   ((2, 0), 1.0), ((-2, 0), 1.0), ((0, 2), 1.0), ((0, -2), 1.0)]
    rows, cols, vals = [], [], []
    rhs = np.zeros((n, 3), np.float64)
    for (dy, dx), c in stencil:
        qy, qx = uy + dy, ux + dx
        if (qy < 0).any() or (qy >= h).any() or (qx < 0).any() or (qx >= w).any():
            return None
        j = idx[qy, qx]
        isu = j >= 0
        rows.append(np.nonzero(isu)[0]); cols.append(j[isu]); vals.append(np.full(int(isu.sum()), c))
        kn = ~isu
        rhs[kn] -= c * (values[qy[kn], qx[kn]] if values is not None else out[qy[kn] + ya, qx[kn] + xa])
    A = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return unk, splu(A).solve(rhs)


GRAIN_SIGMA = 3.0
BOUNDARY_SMOOTH = 2.5
BIH_LIMIT = 0.15


def line_response(dm, length=9, orientations=8):
    """Maximum over orientations of the mean of dm along a short line. Thin continuous dust (hairs,
    scratches) keeps its contrast in this response while uncorrelated noise averages down."""
    best = None
    c = length // 2
    for a in range(orientations):
        th = np.pi * a / orientations
        k = np.zeros((length, length), np.float32)
        for t in np.linspace(-c, c, 4 * length):
            k[int(round(c + t * np.sin(th))), int(round(c + t * np.cos(th)))] = 1.0
        k /= k.sum()
        r = ndi.convolve(dm, k, mode='nearest')
        best = r if best is None else np.maximum(best, r)
    return best


def clean(rgb, ir, grow=4, thr=4.0, cell=24, ncand=1500, ring=3, shift=None, log=print,
          strong_sigma=15.0, grow_strong=9, regrow=False, regrow_k=2.5, regrow_dist=12,
          local_reg=False, local_reg_range=4, local_reg_min_px=150, low_sigma=2.5, agree_grow=True, agree_ir=1.2, agree_rgb=4.0, agree_dist=25, thin_grow=False, thin_size=21, thin_k=6.0, thin_dist=60, thin_reach=14, line_filter=False, line_core=4.5, line_low=3.0, visible_min=None, vis_gate=5.0, ir_gate=6.0, dip_size=21, grain=True, final_clip=True):
    H, W = rgb.shape[:2]
    rgb = rgb.astype(np.float32)
    ir = ir.astype(np.float32)
    # the infrared pixel grid is 2 x 8/3 times coarser than the color grid (3200 x 2400 dpi against 6400 dpi).
    # The color width is rounded down to a multiple of 8, so the ratio of the image sizes is slightly off;
    # use the exact factors whenever the sizes agree with them.
    sy, sx = H / ir.shape[0], W / ir.shape[1]
    if abs(sy - 8 / 3) < 0.02 * 8 / 3:
        sy = 8 / 3
    if abs(sx - 2.0) < 0.02 * 2.0:
        sx = 2.0

    # dust map at the infrared resolution
    t0 = time.time()
    dm = ndi.median_filter(ir, size=7) - ir
    sig = 1.4826 * np.median(np.abs(dm - np.median(dm)))
    strong_core = dm > thr * sig
    if low_sigma is not None:
        # hysteresis: weaker pixels count when they are connected to a strong core, which bridges the
        # gaps where a fiber crosses an image edge and its infrared contrast dips
        low = dm > low_sigma * sig
        lab, nlab = ndi.label(low, structure=np.ones((3, 3)))
        keep = np.zeros(nlab + 1, bool)
        keep[np.unique(lab[strong_core])] = True
        keep[0] = False
        strong_core = keep[lab]
        del lab
    if line_filter:
        lr = line_response(dm)
        sig_l = 1.4826 * np.median(np.abs(lr - np.median(lr)))
        lcore = lr > line_core * sig_l
        llow = lr > line_low * sig_l
        lab, nlab = ndi.label(llow | strong_core, structure=np.ones((3, 3)))
        keep = np.zeros(nlab + 1, bool)
        keep[np.unique(lab[lcore | strong_core])] = True
        keep[0] = False
        extra = keep[lab] & llow
        log('line filter adds %d px to the infrared mask' % int((extra & ~strong_core).sum()))
        strong_core = strong_core | extra
        del lab, lr
    m_native = ndi.binary_dilation(strong_core, iterations=1)
    log('infrared mask: %d px at %dx%d (%.1fs)' % (m_native.sum(), ir.shape[1], ir.shape[0], time.time() - t0))
    ys, xs = np.nonzero(m_native)
    # mask coordinates on the RGB grid (centers of the infrared pixels)
    my = np.clip(((ys + 0.5) * sy).astype(np.int64), 0, H - 1)
    mx = np.clip(((xs + 0.5) * sx).astype(np.int64), 0, W - 1)
    # keep a manageable sample for the registration search
    if len(my) > 150000:
        sel = np.random.default_rng(0).choice(len(my), 150000, replace=False)
        sy_s, sx_s = my[sel], mx[sel]
    else:
        sy_s, sx_s = my, mx

    # registration
    t0 = time.time()
    gl = rgb.mean(2)
    res = gl - ndi.gaussian_filter(gl, 8)
    res /= res.std()
    del gl
    if shift is None:
        # estimate the shift on a grid of tiles and fit an affine model, because the offset drifts along the
        # scan and can also change across the width (a scale mismatch between the two passes)
        nby, nbx = 8, 3
        ye = np.linspace(0, H, nby + 1).astype(int)
        xe = np.linspace(0, W, nbx + 1).astype(int)
        pts = []
        for i in range(nby):
            for j in range(nbx):
                sel = (sy_s >= ye[i]) & (sy_s < ye[i + 1]) & (sx_s >= xe[j]) & (sx_s < xe[j + 1])
                if sel.sum() >= 300:
                    sc, dyb, dxb = estimate_registration(res, sy_s[sel], sx_s[sel], 30, 26)
                    if sc < -0.3:
                        pts.append(((ye[i] + ye[i + 1]) / 2, (xe[j] + xe[j + 1]) / 2, dyb, dxb, sc))
        if not pts:
            log('warning: not enough dust to register the infrared pass, frame left uncleaned')
            return np.clip(np.rint(rgb), 0, 65535).astype(np.uint16), np.zeros((H, W), bool), None
        P = np.array(pts)
        wt = np.minimum(-P[:, 4], 1.5)       # a few very strong features must not outvote the rest
        med = np.median(P[:, 2:4], axis=0)
        keep = np.hypot(P[:, 2] - med[0], P[:, 3] - med[1]) < 12.0
        if keep.sum() < 3:
            keep = np.ones(len(P), bool)

        def fit(keep):
            if keep.sum() >= 4:
                A = np.stack([np.ones(keep.sum()), P[keep, 0] / H, P[keep, 1] / W], 1)
            else:
                A = np.ones((keep.sum(), 1))
            sw = np.sqrt(wt[keep])[:, None]
            cy = np.linalg.lstsq(A * sw, P[keep, 2] * sw[:, 0], rcond=None)[0]
            cx = np.linalg.lstsq(A * sw, P[keep, 3] * sw[:, 0], rcond=None)[0]
            return cy, cx

        def predict(c, y, x):
            if len(c) == 3:
                return c[0] + c[1] * y / H + c[2] * x / W
            return np.full_like(np.asarray(y, dtype=float), c[0])

        for _ in range(3):
            cy, cx = fit(keep)
            ry = P[:, 2] - predict(cy, P[:, 0], P[:, 1])
            rx = P[:, 3] - predict(cx, P[:, 0], P[:, 1])
            bad = np.hypot(ry, rx)
            thr_bad = max(3.0, 2.5 * 1.4826 * np.median(bad[keep]))
            new_keep = bad <= thr_bad
            if new_keep.sum() < 4 or (new_keep == keep).all():
                break
            keep = new_keep
        cy, cx = fit(keep)
        log('registration tiles (row, col, dy, dx, score): %s' % [(int(p[0]), int(p[1]), int(p[2]), int(p[3]), round(p[4], 2)) for p in P])
        c6 = (cy[0], cy[1], cy[2], cx[0], cx[1], cx[2]) if len(cy) == 3 else (cy[0], 0.0, 0.0, cx[0], 0.0, 0.0)
        log('registration model: dy = %.1f %+.1f*y/H %+.1f*x/W, dx = %.1f %+.1f*y/H %+.1f*x/W (%d of %d tiles used)'
            % (c6 + (int(keep.sum()), len(P))))
        # local correction: shifts estimated on a finer grid around the affine prediction, smoothed, and
        # interpolated, so that slowly varying local distortion between the two passes is followed too
        tile = 400
        gy, gx = -(-H // tile), -(-W // tile)
        fy = np.zeros((gy, gx)); fx = np.zeros((gy, gx)); fw = np.zeros((gy, gx))
        pdy = predict(cy, sy_s, sx_s); pdx = predict(cx, sy_s, sx_s)
        ty = np.minimum(sy_s // tile, gy - 1); tx = np.minimum(sx_s // tile, gx - 1)
        tid = ty * gx + tx
        order = np.argsort(tid, kind='stable')
        tid_sorted = tid[order]
        bounds = np.searchsorted(tid_sorted, np.arange(gy * gx + 1))
        used = 0
        for t in range(gy * gx):
            idx = order[bounds[t]:bounds[t + 1]]
            if len(idx) < 150:
                continue
            py_, px_ = sy_s[idx] - np.rint(pdy[idx]).astype(np.int64), sx_s[idx] - np.rint(pdx[idx]).astype(np.int64)
            best = (None, 0, 0)
            zero = None
            for dy in range(-8, 9):
                qy = py_ - dy
                oky = (qy >= 0) & (qy < H)
                for dx in range(-8, 9):
                    qx = px_ - dx
                    ok = oky & (qx >= 0) & (qx < W)
                    if ok.sum() < 100:
                        continue
                    v_ = float(res[qy[ok], qx[ok]].mean())
                    if dy == 0 and dx == 0:
                        zero = v_
                    if best[0] is None or v_ < best[0]:
                        best = (v_, dy, dx)
            if best[0] is not None and zero is not None and best[0] < -0.3 and best[0] < zero - 0.1:
                fy[t // gx, t % gx] = best[1]; fx[t // gx, t % gx] = best[2]; fw[t // gx, t % gx] = min(-best[0], 1.5)
                used += 1
        if used >= 3:
            den = ndi.gaussian_filter(fw, 1.0)
            fy_s = np.where(den > 1e-3, ndi.gaussian_filter(fy * fw, 1.0) / np.maximum(den, 1e-3), 0.0)
            fx_s = np.where(den > 1e-3, ndi.gaussian_filter(fx * fw, 1.0) / np.maximum(den, 1e-3), 0.0)
            # tiles far from any measurement fall back to the affine model
            conf = np.clip(den / 0.3, 0, 1)
            fy_s *= conf; fx_s *= conf
            log('local correction from %d of %d tiles, largest %.1f px' % (used, gy * gx, float(np.hypot(fy_s, fx_s).max())))
        else:
            fy_s = np.zeros((gy, gx)); fx_s = np.zeros((gy, gx))

        def reg(y, x):
            c = (np.asarray(y, dtype=float) / tile - 0.5, np.asarray(x, dtype=float) / tile - 0.5)
            ly = ndi.map_coordinates(fy_s, c, order=1, mode='nearest')
            lx = ndi.map_coordinates(fx_s, c, order=1, mode='nearest')
            return (np.rint(predict(cy, y, x) + ly).astype(np.int64), np.rint(predict(cx, y, x) + lx).astype(np.int64))
        log('(%.1fs)' % (time.time() - t0))
    else:
        log('registration forced: dy,dx = %d,%d' % shift)
        reg = lambda y, x: (np.full_like(y, shift[0]), np.full_like(y, shift[1]))

    # masks on the RGB grid, shifted to the RGB features. Strong infrared dust (an opaque fiber or
    # speck) scatters light around itself, so its halo is wider than its infrared footprint and gets
    # a wider mask than weak dust.
    def to_rgb_grid(m_ir, per_component=False):
        if not per_component:
            up = ndi.zoom(m_ir.astype(np.float32), (sy, sx), order=1)[:H, :W] > 0.3
            out_mask = np.zeros((H, W), bool)
            yy, xx = np.nonzero(up)
            dyr, dxr = reg(yy, xx)
            out_mask[np.clip(yy - dyr, 0, H - 1), np.clip(xx - dxr, 0, W - 1)] = True
            return out_mask
        # each large component is re-registered on its own within a few pixels of the fitted model
        lab, n = ndi.label(m_ir, structure=np.ones((3, 3)))
        labup = ndi.zoom(lab.astype(np.int32), (sy, sx), order=0)[:H, :W]
        out_mask = np.zeros((H, W), bool)
        moved = 0
        for k, sl in enumerate(ndi.find_objects(labup), 1):
            if sl is None:
                continue
            yy, xx = np.nonzero(labup[sl] == k)
            yy = yy + sl[0].start
            xx = xx + sl[1].start
            dyr, dxr = reg(yy, xx)
            by, bx = yy - dyr, xx - dxr
            dyk = dxk = 0
            if len(yy) >= local_reg_min_px:
                step = max(1, len(yy) // 2000)
                py, px = by[::step], bx[::step]
                best_v, best, v0 = None, (0, 0), np.inf
                for ddy in range(-local_reg_range, local_reg_range + 1):
                    for ddx in range(-local_reg_range, local_reg_range + 1):
                        qy, qx = py - ddy, px - ddx
                        ok = (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W)
                        if ok.sum() < 20:
                            continue
                        val = float(res[qy[ok], qx[ok]].mean())
                        if ddy == 0 and ddx == 0:
                            v0 = val
                        if best_v is None or val < best_v:
                            best_v, best = val, (ddy, ddx)
                if best_v is not None and best_v < v0 - 0.15:
                    dyk, dxk = best
                    moved += 1
            out_mask[np.clip(by - dyk, 0, H - 1), np.clip(bx - dxk, 0, W - 1)] = True
        log('local re-registration moved %d large components' % moved)
        return out_mask

    mask = to_rgb_grid(m_native, per_component=local_reg)
    if regrow:
        # region growing: from the registered infrared mask into neighbors that are darker in the
        # negative than their surroundings (the halo of a dust particle), at most regrow_dist pixels out
        lum = rgb.mean(2)
        known0 = ~ndi.binary_dilation(mask, iterations=6)
        kf = known0.astype(np.float32)
        ref = ndi.gaussian_filter(lum * kf, 10) / np.maximum(ndi.gaussian_filter(kf, 10), 1e-3)
        rr = ndi.gaussian_filter(lum - ref, 1.0)
        sample = rr[::7, ::7][known0[::7, ::7]]
        sig_n = 1.4826 * np.median(np.abs(sample - np.median(sample)))
        near = ndi.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=regrow_dist)
        cand = (rr < -regrow_k * sig_n) & near
        before = int(mask.sum())
        mask = ndi.binary_propagation(mask, mask=cand | mask, structure=np.ones((3, 3), bool))
        log('region growing: mask %d -> %d px' % (before, int(mask.sum())))
        del lum, ref, rr, cand, near
    del res
    extra_strong = None
    if agree_grow:
        # two-sensor agreement: pixels near a detected region are added when the infrared is at least weakly
        # positive there AND the color image is clearly darker (in the negative) than its surroundings. This
        # recovers the tails of thin hairs whose infrared contrast dips, without adding image structure
        # (which fails the infrared test) or infrared noise (which fails the color test).
        lum = rgb.mean(2)
        known0 = ~ndi.binary_dilation(mask, iterations=6)
        kf = known0.astype(np.float32)
        ref = ndi.gaussian_filter(lum * kf, 10) / np.maximum(ndi.gaussian_filter(kf, 10), 1e-3)
        rr = ndi.gaussian_filter(lum - ref, 1.0)
        sample = rr[::7, ::7][known0[::7, ::7]]
        sig_n = 1.4826 * np.median(np.abs(sample - np.median(sample)))
        # local dip depth: how far a pixel lies below the grey closing of its surroundings. Dust is narrower
        # than dip_size and shows as a dip; image structure wider than that (teeth, eyes, a face against its
        # background) has no dip inside it, so it cannot pass for dust.
        # (measured on log luminance, which is the contrast the positive shows, and normalized by the local
        # noise of that dip so that grainy dense areas do not look like dust)
        lg = np.log(np.maximum(lum, 1.0))
        dip = ndi.grey_closing(lg, size=(dip_size, dip_size)) - lg
        blk = 96
        hb, wb = H // blk, W // blk
        cut = dip[:hb * blk, :wb * blk].reshape(hb, blk, wb, blk)[:, ::3, :, ::3]
        bm = np.median(cut, axis=(1, 3))
        bs = 1.4826 * np.median(np.abs(cut - bm[:, None, :, None]), axis=(1, 3)) + 1e-4
        bm = ndi.zoom(bm, (H / hb, W / wb), order=1, mode='nearest')[:H, :W]
        bs = ndi.zoom(bs, (H / hb, W / wb), order=1, mode='nearest')[:H, :W]
        rrd = -(ndi.gaussian_filter(dip, 1.0) - bm) / bs
        sig_d = 1.0
        del dip, bm, bs, cut, lg
        # alignment self-check: after registration the infrared dust should be dark in the color image
        agree = float((rr[mask] < -2.0 * sig_n).mean()) if mask.any() else 0.0
        log('registration check: %.0f%% of the infrared dust pixels are clearly dark in the color image%s' % (
            100 * agree, '' if agree >= 0.15 else ' (LOW: the alignment may be poor)'))
        if vis_gate is not None and mask.any():
            # visibility gate: a detected region is kept only when the color image shows it (peak darkness of
            # at least vis_gate times the grain) or when the infrared is strong. Weak infrared blips on dense
            # image areas (teeth, bright eyes) are not dust: filling them removes real detail.
            labm, nm = ndi.label(mask, structure=np.ones((3, 3)))
            idxm = np.arange(1, nm + 1)
            dark = -np.asarray(ndi.minimum(rrd, labm, idxm)) / sig_d
            has_strong = np.asarray(ndi.maximum(to_rgb_grid(dm > strong_sigma * sig).astype(np.uint8), labm, idxm)) > 0
            # a region whose infrared never exceeds ir_gate sigma is at the noise level of the dust map: the
            # color image must not be the only evidence, because a dip in dense detail (the gap between two
            # teeth) looks like dust there (frame 22)
            has_ir = np.asarray(ndi.maximum(to_rgb_grid(dm > ir_gate * sig).astype(np.uint8), labm, idxm)) > 0
            keep_c = np.concatenate([[False], ((dark >= vis_gate) & has_ir) | has_strong])
            log('visibility gate: dropped %d of %d regions that the color image does not show' % (int((~keep_c[1:]).sum()), nm))
            mask = keep_c[labm] & mask
            del labm
        weak = to_rgb_grid(ndi.binary_dilation(dm > agree_ir * sig, iterations=1))
        near = ndi.maximum_filter(mask.astype(np.uint8), size=2 * agree_dist + 1).astype(bool)
        cand = weak & near & (rrd < -agree_rgb * sig_d)
        before = int(mask.sum())
        mask0 = mask
        mask = ndi.binary_propagation(ndi.binary_dilation(mask, iterations=2), mask=cand | mask, structure=np.ones((3, 3), bool))
        extra_strong = mask & ~mask0
        log('two-sensor growing: mask %d -> %d px' % (before, int(mask.sum())))
        del lum, ref, rr, rrd, cand, near, weak, kf, known0
    if thin_grow:
        # thin dark structures of the negative (dust, hairs, scratches) that touch a detected region are added
        # to it: where the infrared contrast of a thin hair dips, the color image still shows it clearly
        lum = rgb.mean(2)
        th = ndi.grey_closing(lum, size=(thin_size, thin_size)) - lum
        sample = th[::5, ::5]
        sig_t = 1.4826 * np.median(np.abs(sample - np.median(sample)))
        cand = th > np.median(sample) + thin_k * sig_t
        near = ndi.maximum_filter(mask.astype(np.uint8), size=2 * thin_dist + 1).astype(bool)
        cand &= near
        before = int(mask.sum())
        mask = ndi.binary_propagation(ndi.binary_dilation(mask, iterations=thin_reach), mask=cand | mask, structure=np.ones((3, 3), bool))
        log('thin structure growing: mask %d -> %d px' % (before, int(mask.sum())))
        del lum, th, cand, near, sample
    omega = ndi.binary_dilation(mask, iterations=grow)
    if extra_strong is not None and extra_strong.any():
        # pixels that the color image shows clearly get the wide growth of strong dust (their halo is wide too)
        omega |= ndi.binary_dilation(extra_strong, iterations=grow_strong)
    m_strong = ndi.binary_dilation(dm > strong_sigma * sig, iterations=1)
    if m_strong.any():
        omega |= ndi.binary_dilation(to_rgb_grid(m_strong), iterations=grow_strong)
    omega = ndi.binary_dilation(omega, iterations=1)
    log('unknown region: %.2f%% of the frame' % (100 * omega.mean()))

    out = rgb
    known = ~omega
    rng = np.random.default_rng(7)
    dys = rng.integers(-220, 221, ncand * 3)
    dxs = rng.integers(-220, 221, ncand * 3)
    far = (np.abs(dys) + np.abs(dxs)) > 14
    dys, dxs = dys[far][:ncand], dxs[far][:ncand]

    # grain level of flat areas, measured once per frame: grain is only transferred at this amplitude
    # (near an edge the local high-pass is dominated by the edge, and copying it makes visible blocks)
    rng_g = np.random.default_rng(3)
    blocks = []
    for _ in range(24):
        y0b = int(rng_g.integers(0, max(1, H - 160))); x0b = int(rng_g.integers(0, max(1, W - 160)))
        blk = rgb[y0b:y0b + 160, x0b:x0b + 160]
        hpb = np.stack([blk[..., c] - ndi.gaussian_filter(blk[..., c], GRAIN_SIGMA) for c in range(3)], -1)
        blocks.append(1.4826 * np.median(np.abs(hpb - np.median(hpb, axis=(0, 1))), axis=(0, 1)))
    sig_global = np.median(np.array(blocks), axis=0) + 1.0
    log('grain level (counts): %s' % np.round(sig_global, 0))

    # pass 1: smooth base. Each connected dust region is solved as one biharmonic problem so that no
    # tile seams appear (very large regions are cut into 96 px blocks). Edges continue across the hole
    # and nothing is copied from elsewhere.
    t1 = time.time()
    kb = known.copy()
    skipped = []
    lab, nlab = ndi.label(omega, structure=np.ones((3, 3), bool))
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        comp_full = lab[sl] == k
        regions = []
        if int(comp_full.sum()) <= 40000:
            ya, yb = max(0, sl[0].start - 14), min(H, sl[0].stop + 14)
            xa, xb = max(0, sl[1].start - 14), min(W, sl[1].stop + 14)
            comp = np.zeros((yb - ya, xb - xa), bool)
            comp[sl[0].start - ya:sl[0].stop - ya, sl[1].start - xa:sl[1].stop - xa] = comp_full
            regions.append((ya, yb, xa, xb, comp))
        else:
            for ty in range(sl[0].start, sl[0].stop, 96):
                for tx in range(sl[1].start, sl[1].stop, 96):
                    ya, yb = max(0, ty - 14), min(H, ty + 96 + 14)
                    xa, xb = max(0, tx - 14), min(W, tx + 96 + 14)
                    comp = lab[ya:yb, xa:xb] == k
                    inside = np.zeros_like(comp)
                    inside[ty - ya:ty + 96 - ya, tx - xa:tx + 96 - xa] = True
                    regions.append((ya, yb, xa, xb, comp & inside))
        for ya, yb, xa, xb, comp in regions:
            if not comp.any():
                continue
            wk = kb[ya:yb, xa:xb]
            # boundary data: the known pixels around the region, lightly smoothed so that their grain and
            # chroma noise is not interpolated into smooth colored blobs inside the region
            wvals = out[ya:yb, xa:xb].astype(np.float32).copy()
            kf_w = wk.astype(np.float32)
            den_w = np.maximum(ndi.gaussian_filter(kf_w, BOUNDARY_SMOOTH), 1e-3)
            for c_ in range(3):
                wvals[..., c_] = np.where(wk, ndi.gaussian_filter(wvals[..., c_] * kf_w, BOUNDARY_SMOOTH) / den_w, wvals[..., c_])
            sol2 = diffusion_base(out, kb, ya, yb, xa, xb, order=2, target=comp, values=wvals)
            sol1 = diffusion_base(out, kb, ya, yb, xa, xb, order=1, target=comp, values=wvals)
            tgt = out[ya:yb, xa:xb]
            if sol1 is not None and sol2 is not None:
                unk_mask, v1 = sol1
                _, v2 = sol2
                lut = -np.ones(unk_mask.shape, np.int64)
                lut[unk_mask] = np.arange(int(unk_mask.sum()))
                # the biharmonic solution may differ from the harmonic one by at most a fraction of the
                # contrast around the region, which keeps edges continuing without dark or bright blocks
                ring_c = ndi.binary_dilation(comp, iterations=3) & ~comp & wk
                rv_c = tgt[ring_c]
                if len(rv_c):
                    tau = BIH_LIMIT * (np.percentile(rv_c, 95, axis=0) - np.percentile(rv_c, 5, axis=0)) + 150.0
                else:
                    tau = np.full(3, 150.0)
                i1 = lut[comp]
                orig = tgt[comp].copy()
                new_vals = v1[i1] + np.clip(v2[i1] - v1[i1], -tau, tau)
                if visible_min is not None:
                    # visibility gate: leave the region alone when the color image deviates from what its
                    # surroundings predict by no more than the grain does (the infrared flagged something
                    # that is not visible, for example the edge of an eyelid)
                    core = mask[ya:yb, xa:xb][comp]
                    if core.sum() >= 8:
                        kn_win = tgt[wk]
                        gx = np.abs(np.diff(tgt, axis=1))[wk[:, 1:] & wk[:, :-1]]
                        sig_g = 1.4826 * np.median(gx, axis=0) / np.sqrt(2) + 1.0 if len(gx) else np.full(3, 300.0)
                        score = float((np.abs(orig - new_vals)[core] / sig_g).mean())
                        if score < visible_min:
                            skipped.append((ya, yb, xa, xb, comp))
                            continue
                tgt[comp] = new_vals
            else:
                # window touches the image border: copy the nearest known pixel
                idx_near = ndi.distance_transform_edt(~wk, return_distances=False, return_indices=True)
                tgt[comp] = tgt[idx_near[0][comp], idx_near[1][comp]]
            kb[ya:yb, xa:xb][comp] = True
    del kb, lab
    for ya, yb, xa, xb, comp in skipped:
        omega[ya:yb, xa:xb][comp] = False
        known[ya:yb, xa:xb][comp] = True
    log('smooth base: %.0fs (%d regions left alone because they are not visible in the color image)' % (time.time() - t1, len(skipped)))

    oy_, ox_ = np.nonzero(omega)
    cells = np.unique((oy_ // cell) * (W // cell + 2) + (ox_ // cell))
    cpr = W // cell + 2
    log('%d cells to fill' % len(cells))
    struct4 = ndi.generate_binary_structure(2, 1)
    nfail = 0
    t0 = time.time()
    for ci, c in enumerate(cells):
        cy, cx = divmod(int(c), cpr)
        y0, y1 = cy * cell, min((cy + 1) * cell, H)
        x0, x1 = cx * cell, min((cx + 1) * cell, W)
        pad = ring + 2
        ya, yb = max(0, y0 - pad), min(H, y1 + pad)
        xa, xb = max(0, x0 - pad), min(W, x1 + pad)
        Rloc = np.zeros((yb - ya, xb - xa), bool)
        Rloc[y0 - ya:y1 - ya, x0 - xa:x1 - xa] = ~known[y0:y1, x0:x1]
        if not Rloc.any():
            continue
        kn_loc = known[ya:yb, xa:xb]
        ringm = ndi.binary_dilation(Rloc, structure=struct4, iterations=ring) & ~Rloc & kn_loc
        ry, rx = np.nonzero(ringm)
        ry, rx = ry + ya, rx + xa
        Ry, Rx = np.nonzero(Rloc)
        Ry, Rx = Ry + ya, Rx + xa
        if len(ry) < 8:
            nfail += 1
            continue
        cy_ = ry[None, :] + dys[:, None]
        cx_ = rx[None, :] + dxs[:, None]
        ry_ = Ry[None, :] + dys[:, None]
        rx_ = Rx[None, :] + dxs[:, None]
        inb = ((cy_ >= 0) & (cy_ < H) & (cx_ >= 0) & (cx_ < W)).all(1)
        inb &= ((ry_ >= 0) & (ry_ < H) & (rx_ >= 0) & (rx_ < W)).all(1)
        cyc, cxc = np.clip(cy_, 0, H - 1), np.clip(cx_, 0, W - 1)
        ryc, rxc = np.clip(ry_, 0, H - 1), np.clip(rx_, 0, W - 1)
        valid = inb & known[cyc, cxc].all(1) & known[ryc, rxc].all(1)
        if not valid.any():
            nfail += 1
            continue
        diff = out[cyc, cxc] - out[ry, rx][None, :, :]
        cost = (diff ** 2).sum((1, 2))
        cost[~valid] = np.inf
        h, w = yb - ya, xb - xa
        ly, lx = Ry - ya, Rx - xa
        base = out[Ry, Rx].copy()

        # 2. grain: the high-pass detail of a clean patch whose surroundings match, limited to the grain
        #    amplitude measured around the hole, so no speck or edge can be pasted in
        win = out[ya:yb, xa:xb]
        kf = kn_loc.astype(np.float32)
        den = np.maximum(ndi.gaussian_filter(kf, GRAIN_SIGMA), 1e-3)
        blur_t = np.stack([ndi.gaussian_filter(win[..., c] * kf, GRAIN_SIGMA) for c in range(3)], -1) / den[..., None]
        hp_t = (win - blur_t)[ringm]
        sig = sig_global
        detail = np.zeros_like(base)
        for kk in (np.argsort(cost)[:25] if grain else []):
            if not np.isfinite(cost[kk]):
                break
            dy, dx = int(dys[kk]), int(dxs[kk])
            sya, syb = max(0, ya + dy), min(H, yb + dy)
            sxa, sxb = max(0, xa + dx), min(W, xb + dx)
            if (syb - sya, sxb - sxa) != (h, w):
                continue
            swin = out[sya:syb, sxa:sxb]
            skn = known[sya:syb, sxa:sxb].astype(np.float32)
            sden = np.maximum(ndi.gaussian_filter(skn, GRAIN_SIGMA), 1e-3)
            blur_s = np.stack([ndi.gaussian_filter(swin[..., c] * skn, GRAIN_SIGMA) for c in range(3)], -1) / sden[..., None]
            hp_s = (swin - blur_s)[ly, lx]
            if (np.abs(hp_s) > 4 * sig).mean() <= 0.02:
                detail = np.clip(hp_s, -3 * sig, 3 * sig)
                break
        rv = out[ry, rx]
        m = 0.25 * (rv.max(0) - rv.min(0)) + 300.0
        out[Ry, Rx] = np.clip(base + detail, rv.min(0) - m, rv.max(0) + m) if final_clip else base + detail
        known[Ry, Rx] = True
        if (ci + 1) % 500 == 0:
            log('  filled %d / %d cells (%.0fs)' % (ci + 1, len(cells), time.time() - t0))
    log('cells without a usable source: %d (%.0fs)' % (nfail, time.time() - t0))
    return np.clip(np.rint(out), 0, 65535).astype(np.uint16), mask, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rgb')
    ap.add_argument('ir')
    ap.add_argument('out')
    ap.add_argument('--mask', help='also write the registered dust mask as a 16-bit gray TIFF')
    ap.add_argument('--grow', type=int, default=4, help='mask growth in RGB pixels (default 4)')
    ap.add_argument('--threshold', type=float, default=4.0, help='infrared dust threshold in robust sigmas (default 4)')
    ap.add_argument('--shift', help='force the registration shift as dy,dx in RGB pixels')
    a = ap.parse_args()
    shift = tuple(int(v) for v in a.shift.split(',')) if a.shift else None
    rgb = read_tiff(a.rgb)
    ir = read_tiff(a.ir)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint16:
        sys.exit('the RGB file must be a 16-bit RGB TIFF')
    if ir.ndim != 2:
        sys.exit('the infrared file must be a gray TIFF')
    out, mask, sh = clean(rgb, ir, grow=a.grow, thr=a.threshold, shift=shift)
    write_tiff16(a.out, out)
    if a.mask:
        write_tiff16(a.mask, mask.astype(np.uint16) * 65535)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
