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


def diffusion_base(out, known, ya, yb, xa, xb, order=2, target=None):
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
        rhs[kn] -= c * out[qy[kn] + ya, qx[kn] + xa]
    A = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return unk, splu(A).solve(rhs)


GRAIN_SIGMA = 3.0
BIH_LIMIT = 0.15


def clean(rgb, ir, grow=4, thr=4.0, cell=24, ncand=1500, ring=3, shift=None, log=print,
          strong_sigma=15.0, grow_strong=9, regrow=False, regrow_k=2.5, regrow_dist=12,
          local_reg=False, local_reg_range=4, local_reg_min_px=150, low_sigma=2.5, visible_min=None, grain=True, final_clip=True):
    H, W = rgb.shape[:2]
    rgb = rgb.astype(np.float32)
    ir = ir.astype(np.float32)
    sy, sx = H / ir.shape[0], W / ir.shape[1]

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
        # estimate the shift in vertical bands and fit a line, since the offset drifts along the scan
        nb = 8
        edges = np.linspace(0, H, nb + 1).astype(int)
        pts = []
        for k in range(nb):
            sel = (sy_s >= edges[k]) & (sy_s < edges[k + 1])
            if sel.sum() >= 300:
                sc, dyb, dxb = estimate_registration(res, sy_s[sel], sx_s[sel], 30, 14)
                if sc < -0.3:
                    pts.append(((edges[k] + edges[k + 1]) / 2, dyb, dxb, sc))
        if not pts:
            log('warning: not enough dust to register the infrared pass, frame left uncleaned')
            return np.clip(np.rint(rgb), 0, 65535).astype(np.uint16), np.zeros((H, W), bool), None
        yc = np.array([p[0] for p in pts]); wt = np.array([-p[3] for p in pts])
        if len(pts) == 1:
            fit_dy = np.array([0.0, float(pts[0][1])]); fit_dx = np.array([0.0, float(pts[0][2])])
        else:
            fit_dy = np.polyfit(yc, [p[1] for p in pts], 1, w=wt)
            fit_dx = np.polyfit(yc, [p[2] for p in pts], 1, w=wt)
        log('registration (RGB px) per band: %s' % [(int(p[0]), p[1], p[2], round(p[3], 2)) for p in pts])
        log('registration model: dy = %.2f + %.5f*y, dx = %.2f + %.5f*y' % (
            fit_dy[1], fit_dy[0], fit_dx[1], fit_dx[0]))
        reg = lambda y: (np.rint(np.polyval(fit_dy, y)).astype(np.int64), np.rint(np.polyval(fit_dx, y)).astype(np.int64))
        log('(%.1fs)' % (time.time() - t0))
    else:
        log('registration forced: dy,dx = %d,%d' % shift)
        reg = lambda y: (np.full_like(y, shift[0]), np.full_like(y, shift[1]))

    # masks on the RGB grid, shifted to the RGB features. Strong infrared dust (an opaque fiber or
    # speck) scatters light around itself, so its halo is wider than its infrared footprint and gets
    # a wider mask than weak dust.
    def to_rgb_grid(m_ir, per_component=False):
        if not per_component:
            up = ndi.zoom(m_ir.astype(np.float32), (sy, sx), order=1)[:H, :W] > 0.3
            out_mask = np.zeros((H, W), bool)
            yy, xx = np.nonzero(up)
            dyr, dxr = reg(yy)
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
            dyr, dxr = reg(yy)
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
    omega = ndi.binary_dilation(mask, iterations=grow)
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
            sol2 = diffusion_base(out, kb, ya, yb, xa, xb, order=2, target=comp)
            sol1 = diffusion_base(out, kb, ya, yb, xa, xb, order=1, target=comp)
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
