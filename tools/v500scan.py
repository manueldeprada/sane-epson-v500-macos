#!/usr/bin/env python3
"""Scan every frame in the Epson V500 film holder and write cleaned negatives and positives.

    v500scan.py                  scan everything in the tray
    v500scan.py --list           preview and frame detection only
    v500scan.py --only 2,3       scan only these frames

For each frame: a 6400 dpi 16-bit linear RGB pass and a 3200 x 2400 dpi infrared pass, dust
removal from the infrared pass (v500clean.py), then
    frameN_negative.tif   cleaned 16-bit linear negative (input for Negative Lab Pro or similar)
    frameN.heic           10-bit sRGB positive, upright (negpos.py)
Raw passes are deleted unless --keep-raw is given.

Needs the patched SANE build (scripts/build-sane.sh), numpy, scipy, Pillow, and swiftc (to build
tiff2heic once). Set V500_SANE_PREFIX if SANE is not in ~/.local/sane-epson-v500 and V500_OUT to
choose the output folder (default ./scans).
"""
import argparse
import concurrent.futures as cf
import datetime
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import negpos            # noqa: E402
import v500clean         # noqa: E402

# ---- defaults
SANE_PREFIX = os.environ.get('V500_SANE_PREFIX', os.path.expanduser('~/.local/sane-epson-v500'))
OUT_ROOT = os.environ.get('V500_OUT', os.path.join(os.getcwd(), 'scans'))
RGB_DPI = 6400
IR_DPI = 3200
PREVIEW_DPI = 200
TPU_MM = (68.5, 236.9)          # usable transparency area (width, height), slightly inside 68.58 x 236.98
FILM_PITCH_MM = 38.0            # 35 mm film: 8 perforations of 4.75 mm
GAP_MM = 2.2                    # unexposed band between frames, removed from each side
WINDOW_INSET_MM = 0.5
SCAN_TIMEOUT = 1500


def say(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- scanner access
def scan_env():
    env = dict(os.environ)
    env['SANE_CONFIG_DIR'] = os.path.join(SANE_PREFIX, 'etc', 'sane.d')
    return env


def scanimage():
    path = os.path.join(SANE_PREFIX, 'bin', 'scanimage')
    if not os.path.exists(path):
        sys.exit('patched scanimage not found at %s (set V500_SANE_PREFIX)' % path)
    return path


def find_device():
    r = subprocess.run([scanimage(), '-L'], env=scan_env(), capture_output=True, text=True, timeout=120)
    m = re.search(r"device `(epson2:[^']+)'", r.stdout)
    if not m:
        sys.exit('no Epson scanner found. Power it on, close VueScan and Image Capture, and check the USB cable.\n'
                 + r.stdout + r.stderr)
    return m.group(1)


def run_scan(args, out_path):
    """One scanimage run with a single retry that re-detects the device address."""
    for attempt in (1, 2):
        dev = find_device()
        cmd = [scanimage(), '-d', dev, '--source', 'Transparency Unit', '--film-type', FILM_TYPE] + args + [
            '--format=tiff', '-o', out_path]
        t0 = time.time()
        r = subprocess.run(cmd, env=scan_env(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=SCAN_TIMEOUT)
        if r.returncode == 0 and os.path.exists(out_path):
            return time.time() - t0
        say('  scan failed (attempt %d): %s' % (attempt, r.stderr.strip().splitlines()[-1:] or r.returncode))
        time.sleep(5)
    sys.exit('scanner did not respond. If it is wedged, power-cycle it (or reset it over USB) and rerun.')


# ---------------------------------------------------------------- frame detection
def detect_frames(prev, dpi=PREVIEW_DPI):
    """Find film frames in a preview (uint8 H x W x 3). Returns a list of dicts in mm."""
    from scipy import ndimage as ndi
    px = dpi / 25.4
    lum = prev.astype(np.float32).mean(2)
    H, W = lum.shape
    lit = ndi.uniform_filter1d((lum > 40).mean(0), 5)
    # windows: runs of columns where most rows are lit, at least 12 mm wide
    runs, start = [], None
    for x in range(W + 1):
        on = x < W and lit[x] > 0.35
        if on and start is None:
            start = x
        if not on and start is not None:
            if x - start >= 12 * px:
                runs.append((start, x))
            start = None
    frames = []
    for (c0, c1) in runs:
        inset = int(round(WINDOW_INSET_MM * px))
        cc0, cc1 = c0 + int(0.08 * (c1 - c0)), c1 - int(0.08 * (c1 - c0))
        win = lum[:, cc0:cc1]
        film = ((win > 25) & (win < 238)).mean(1) > 0.5
        film = ndi.binary_closing(film, structure=np.ones(int(6 * px)))
        lab, n = ndi.label(film)
        if n == 0:
            continue
        sizes = ndi.sum(film, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        rows = np.nonzero(lab == k)[0]
        r0, r1 = int(rows[0]), int(rows[-1])
        if (r1 - r0) < 20 * px:
            continue
        # gaps between frames are uniform across the width: low standard deviation along the row
        sd = win[r0:r1 + 1].std(1)
        u = np.clip((np.median(sd) - sd) / (np.median(sd) + 1e-6), 0, 1)
        pitch = FILM_PITCH_MM * px
        best = (-1.0, 0)
        for ph in np.arange(0, pitch, 1.0):
            pos = ph + np.arange(0, (r1 - r0) / pitch + 1) * pitch
            pos = pos[pos < len(u) - 2]
            if len(pos) == 0:
                continue
            sc = np.mean([u[max(0, int(p) - 3):int(p) + 4].mean() for p in pos])
            if sc > best[0]:
                best = (sc, ph)
        score, phase = best
        gap = GAP_MM * px
        k0 = -1
        while True:
            top = r0 + phase + k0 * pitch + gap / 2
            bot = r0 + phase + (k0 + 1) * pitch - gap / 2
            if bot > r1 + 4:
                break
            if top >= r0 - 4:
                frames.append({
                    'x_mm': (cc0 - (0.08 * (c1 - c0)) + inset) / px,
                    'y_mm': top / px,
                    'w_mm': ((c1 - c0) - 2 * inset) / px,
                    'h_mm': (bot - top) / px,
                    'phase_score': float(score),
                    'window': (c0, c1),
                })
            k0 += 1
    # number frames: windows left to right, top to bottom inside each window
    frames.sort(key=lambda f: (f['window'][0], f['y_mm']))
    for i, f in enumerate(frames, 1):
        f['n'] = i
    return frames


def save_preview(prev, frames, path):
    from PIL import Image, ImageDraw
    im = Image.fromarray(prev)
    d = ImageDraw.Draw(im)
    px = PREVIEW_DPI / 25.4
    for f in frames:
        x0, y0 = f['x_mm'] * px, f['y_mm'] * px
        d.rectangle([x0, y0, x0 + f['w_mm'] * px, y0 + f['h_mm'] * px], outline=(255, 40, 40), width=2)
        d.text((x0 + 6, y0 + 6), '#%d' % f['n'], fill=(255, 255, 0))
    im.save(path)


# ---------------------------------------------------------------- per-frame processing
def process_frame(n, rgb_path, ir_path, out_neg, out_heic, keep_raw):
    """Runs in a worker process while the scanner moves on to the next frame."""
    t0 = time.time()
    lines = []
    log = lambda m: lines.append(m)
    rgb = v500clean.read_tiff(rgb_path)
    ir = v500clean.read_tiff(ir_path)
    cleaned, mask, _ = v500clean.clean(rgb, ir, log=log)
    del rgb
    v500clean.write_tiff16(out_neg, cleaned)
    pos = negpos.convert(cleaned, log=log)
    del cleaned
    tmp = out_heic + '.tmp.tif'
    v500clean.write_tiff16(tmp, pos)
    del pos
    subprocess.run([heic_tool(), tmp, out_heic, '0.9'], check=True, capture_output=True)
    os.remove(tmp)
    if not keep_raw:
        os.remove(rgb_path)
        os.remove(ir_path)
    lines.append('done in %.0fs' % (time.time() - t0))
    return n, lines


def heic_tool():
    exe = os.path.join(HERE, 'tiff2heic')
    src = os.path.join(HERE, 'tiff2heic.swift')
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        say('building tiff2heic ...')
        subprocess.run(['swiftc', '-O', src, '-o', exe], check=True)
    return exe


# ---------------------------------------------------------------- main
FILM_TYPE = 'Negative Film'


def main():
    global FILM_TYPE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', help='output folder (default: scans/<date>_<time>)')
    ap.add_argument('--list', action='store_true', help='preview and detect frames, then stop')
    ap.add_argument('--only', help='comma-separated frame numbers to scan, for example 2,3')
    ap.add_argument('--keep-raw', action='store_true', help='keep the raw RGB and infrared TIFFs')
    ap.add_argument('--film-type', default=FILM_TYPE)
    ap.add_argument('--frames', help='manual areas in mm as x,y,w,h;x,y,w,h (skips detection)')
    a = ap.parse_args()
    FILM_TYPE = a.film_type

    out = a.out or os.path.join(OUT_ROOT, datetime.datetime.now().strftime('%Y-%m-%d_%H%M'))
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    heic_tool()

    # preview
    prev_tif = os.path.join(out, 'preview.tif')
    say('preview at %d dpi ...' % PREVIEW_DPI)
    dt = run_scan(['--mode', 'Color', '--depth', '8', '--resolution', str(PREVIEW_DPI),
                   '-l', '0', '-t', '0', '-x', str(TPU_MM[0]), '-y', str(TPU_MM[1])], prev_tif)
    prev = v500clean.read_tiff(prev_tif)
    say('  %.0fs' % dt)

    if a.frames:
        frames = []
        for i, part in enumerate(a.frames.split(';'), 1):
            x, y, w, h = (float(v) for v in part.split(','))
            frames.append({'n': i, 'x_mm': x, 'y_mm': y, 'w_mm': w, 'h_mm': h, 'phase_score': 1.0})
    else:
        frames = detect_frames(prev)
    if not frames:
        sys.exit('no film found in the preview (%s). Check that the holder is seated and the film is in the window.'
                 % prev_tif)
    save_preview(prev, frames, os.path.join(out, 'preview_frames.png'))
    os.remove(prev_tif)
    say('%d frame(s) detected:' % len(frames))
    for f in frames:
        say('  #%d  x %.1f  y %.1f  %.1f x %.1f mm' % (f['n'], f['x_mm'], f['y_mm'], f['w_mm'], f['h_mm']))
    if min(f['phase_score'] for f in frames) < 0.15:
        say('warning: the frame spacing was hard to find; check preview_frames.png before trusting the borders')
    if a.list:
        say('preview with frame boxes: %s' % os.path.join(out, 'preview_frames.png'))
        return
    if a.only:
        wanted = {int(v) for v in a.only.split(',')}
        frames = [f for f in frames if f['n'] in wanted]

    # scan each frame; clean the previous one in a worker while the next is scanning
    pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=__import__('multiprocessing').get_context('spawn'))
    futures = []
    for f in frames:
        n = f['n']
        say('frame %d: RGB %d dpi ...' % (n, RGB_DPI))
        area = ['-l', '%.2f' % f['x_mm'], '-t', '%.2f' % f['y_mm'], '-x', '%.2f' % f['w_mm'], '-y', '%.2f' % f['h_mm']]
        rgb_path = os.path.join(out, 'frame%d_rgb_raw.tif' % n)
        ir_path = os.path.join(out, 'frame%d_ir_raw.tif' % n)
        dt = run_scan(['--mode', 'Color', '--depth', '16', '--gamma-correction', 'User defined (Gamma=1.0)',
                       '--resolution', str(RGB_DPI)] + area, rgb_path)
        say('  RGB %.0fs; infrared %d dpi ...' % (dt, IR_DPI))
        dt = run_scan(['--mode', 'Infrared', '--depth', '16', '--resolution', str(IR_DPI)] + area, ir_path)
        say('  infrared %.0fs; cleaning in the background' % dt)
        futures.append(pool.submit(
            process_frame, n, rgb_path, ir_path,
            os.path.join(out, 'frame%d_negative.tif' % n), os.path.join(out, 'frame%d.heic' % n), a.keep_raw))
        for fut in [x for x in futures if x.done()]:
            futures.remove(fut)
            report(fut)
    for fut in futures:
        report(fut)
    pool.shutdown()
    say('done: %s' % out)


def report(fut):
    n, lines = fut.result()
    say('frame %d:' % n)
    for l in lines:
        say('  ' + l)


if __name__ == '__main__':
    main()
