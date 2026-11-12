#!/usr/bin/env python3
"""Remove dust from a 16-bit RGB film scan using its infrared pass.

    v500clean.py rgb.tif ir.tif cleaned.tif [--mask mask.tif]

The infrared image may have a different pixel grid than the RGB image (the V500 scans
infrared at most at 3200 x 2400 dpi). The infrared dust map is built at its own resolution,
registered to the RGB image by correlating it with dark residuals in the RGB luminance,
then each masked region is filled with the best-matching clean patch from the same image
and Poisson-corrected so that the borders blend. Needs numpy and scipy.
"""
import argparse
import struct
import sys
import time

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
    if tags.get(259, (1,))[0] != 1:
        raise ValueError('compressed TIFF is not supported')
    w, h = tags[256][0], tags[257][0]
    bits = tags[258][0]
    spp = tags.get(277, (1,))[0]
    offs, counts = tags[273], tags[279]
    buf = b''.join(data[o:o + c] for o, c in zip(offs, counts))
    if bits == 16:
        arr = np.frombuffer(buf, dtype=e + 'u2')[: w * h * spp]
        if e == '>':
            arr = arr.astype('<u2')
    else:
        arr = np.frombuffer(buf, dtype=np.uint8)[: w * h * spp]
    return arr.reshape(h, w, spp) if spp > 1 else arr.reshape(h, w)


def write_tiff16(path, arr):
    """Write a 16-bit gray or RGB uncompressed TIFF (single strip)."""
    arr = np.ascontiguousarray(arr.astype('<u2'))
    h, w = arr.shape[:2]
    spp = arr.shape[2] if arr.ndim == 3 else 1
    pixel_bytes = arr.tobytes()
    entries = []
    bits_off = None
    # layout: header(8) | pixel data | extra (bits array) | IFD
    data_off = 8
    extra_off = data_off + len(pixel_bytes)
    extra = b''
    if spp == 3:
        bits_off = extra_off
        extra += struct.pack('<3H', 16, 16, 16)
    ifd_off = extra_off + len(extra)
    if ifd_off % 2:
        extra += b'\0'
        ifd_off += 1

    def ent(tag, typ, cnt, val):
        entries.append(struct.pack('<HHII', tag, typ, cnt, val))

    ent(256, 4, 1, w)
    ent(257, 4, 1, h)
    if spp == 3:
        ent(258, 3, 3, bits_off)
    else:
        ent(258, 3, 1, 16)
    ent(259, 3, 1, 1)
    ent(262, 3, 1, 2 if spp == 3 else 1)
    ent(273, 4, 1, data_off)
    ent(277, 3, 1, spp)
    ent(278, 4, 1, h)
    ent(279, 4, 1, len(pixel_bytes))
    ent(284, 3, 1, 1)
    with open(path, 'wb') as f:
        f.write(b'II*\0' + struct.pack('<I', ifd_off))
        f.write(pixel_bytes)
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


def clean(rgb, ir, grow=4, thr=4.0, cell=24, ncand=1500, ring=3, shift=None, log=print):
    H, W = rgb.shape[:2]
    rgb = rgb.astype(np.float32)
    ir = ir.astype(np.float32)
    sy, sx = H / ir.shape[0], W / ir.shape[1]

    # dust map at the infrared resolution
    t0 = time.time()
    dm = ndi.median_filter(ir, size=7) - ir
    sig = 1.4826 * np.median(np.abs(dm - np.median(dm)))
    m_native = ndi.binary_dilation(dm > thr * sig, iterations=1)
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
    del res

    # mask on the RGB grid, shifted to the RGB features
    mask = np.zeros((H, W), bool)
    up = ndi.zoom(m_native.astype(np.float32), (sy, sx), order=1)
    up = up[:H, :W] > 0.3
    if up.shape != (H, W):
        tmp = np.zeros((H, W), bool)
        tmp[:up.shape[0], :up.shape[1]] = up
        up = tmp
    mask = np.zeros((H, W), bool)
    yy, xx = np.nonzero(up)
    dyr, dxr = reg(yy)
    yy = yy - dyr
    xx = xx - dxr
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    mask[yy[ok], xx[ok]] = True
    del up
    omega = ndi.binary_dilation(ndi.binary_dilation(mask, iterations=grow), iterations=1)
    log('unknown region: %.2f%% of the frame' % (100 * omega.mean()))

    out = rgb
    known = ~omega
    rng = np.random.default_rng(7)
    dys = rng.integers(-220, 221, ncand * 3)
    dxs = rng.integers(-220, 221, ncand * 3)
    far = (np.abs(dys) + np.abs(dxs)) > 14
    dys, dxs = dys[far][:ncand], dxs[far][:ncand]

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
        k = int(np.argmin(cost))
        dy, dx = int(dys[k]), int(dxs[k])

        # Poisson correction on a local window
        h, w = yb - ya, xb - xa
        idx = -np.ones((h, w), np.int64)
        n = len(Ry)
        ly, lx = Ry - ya, Rx - xa
        idx[ly, lx] = np.arange(n)
        rows, cols, vals = [], [], []
        rhs = np.zeros((n, 3), np.float64)
        deg = np.zeros(n)
        for ddy, ddx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            qy, qx = ly + ddy, lx + ddx
            inw = (qy >= 0) & (qy < h) & (qx >= 0) & (qx < w)
            qyc, qxc = np.clip(qy, 0, h - 1), np.clip(qx, 0, w - 1)
            inR = inw & (idx[qyc, qxc] >= 0)
            isb = inw & ~inR & kn_loc[qyc, qxc]
            deg += inR + isb
            rows.append(np.nonzero(inR)[0])
            cols.append(idx[qyc, qxc][inR])
            vals.append(-np.ones(inR.sum()))
            bi = np.nonzero(isb)[0]
            gy, gx = qyc[bi] + ya, qxc[bi] + xa
            sy2, sx2 = gy + dy, gx + dx
            good = (sy2 >= 0) & (sy2 < H) & (sx2 >= 0) & (sx2 < W)
            bgood = bi[good]
            rhs[bgood] += out[gy[good], gx[good]] - out[sy2[good], sx2[good]]
            deg[np.setdiff1d(bi, bgood)] -= 1
        deg = np.maximum(deg, 1)
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(deg)
        A = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        corr = splu(A).solve(rhs)
        out[Ry, Rx] = out[Ry + dy, Rx + dx] + corr
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
