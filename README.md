# sane-epson-v500-macos

A patch that lets SANE scan film with the Epson Perfection V500 Photo on macOS, with a separate
infrared pass for dust detection.

Stock SANE cannot open this scanner (USB `04b8:0130`, reported as `GT-X770`), because it does not
speak plain ESC/I on the USB bus. Epson's macOS ICA driver contains a library that does. The patch
loads that library from your own Epson install and sends the scanner's traffic through it. Nothing
from Epson is included here.

Tested on a V500 Photo on macOS 27, Apple Silicon.

## Requirements

- Epson's ICA driver for the V500 (installed from Epson's support site)
- Xcode command line tools, and `brew install libusb pkgconf`

## Build

```bash
scripts/build-sane.sh [prefix]    # default prefix: ~/.local/sane-epson-v500
```

This downloads SANE 1.4.0, applies the patch, and builds only the `epson2` backend. It leaves any
existing SANE install alone. To check the driver library and scanner without SANE, build
`tools/interpreter-probe.c` (instructions in the file).

## Use

```bash
export SANE_CONFIG_DIR=~/.local/sane-epson-v500/etc/sane.d
scanimage=~/.local/sane-epson-v500/bin/scanimage

$scanimage -L        # "Epson GT-X770"; the device name changes after a USB reset

# one 35 mm negative frame, 16-bit, 800 dpi
$scanimage -d epson2:libusb:000:007 --source "Transparency Unit" --film-type "Negative Film" \
  --mode Color --depth 16 --gamma-correction "User defined (Gamma=1.0)" --resolution 800 \
  -l 38.1 -t 45.72 -x 25.4 -y 38.1 --format=tiff -o frame.tiff

# infrared pass over the same area: --mode Infrared --depth 16 (800, 1600 or 3200 dpi)
```

## Scan a film strip

```bash
python3 tools/v500scan.py          # preview, find the frames, scan, clean, convert
python3 tools/v500scan.py --list   # preview and frame boxes only
```

For each frame the tool scans a 6400 dpi color pass and a 3200 x 2400 dpi infrared pass (about
3 minutes), removes dust, and writes `frameN_negative.tif` (16-bit linear, Deflate) and
`frameN.heic` (10-bit sRGB, upright) to `scans/<date>_<time>/`. Cleaning runs at 6400 dpi. The
stored files are then averaged 2 x 2 in linear light to 3200 dpi (about 57 MB per negative and 5 MB
per HEIC; `--store-dpi 6400` keeps the full size). Frames are cleaned in worker processes while the
scanner moves on to the next one. `--start N` continues the numbering for the next tray of a roll,
`--raw-only` keeps the raw passes without cleaning, and `--from-raw DIR` cleans saved raw passes.
It needs numpy, scipy, Pillow, and `swiftc` (builds `tiff2heic` on first use). The positive is a
plain per-channel inversion with gray-world balance. Use the negative TIFF for a better conversion.

**Dust removal.** The infrared pass is not registered to the color pass, and the offset differs
between scans: 13 rows on one 8 mm crop, -3 to +1 rows along one full frame, and 4 to 9 columns
(6400 dpi pixels). A fixed correction in SANE or in a scanning app is therefore wrong.
`tools/v500clean.py` measures the shift for each scan by correlating infrared dust with dark specks
in the color image and fits a linear drift along the frame. It grows the mask by 4 pixels and fills
each masked cell in two parts: a biharmonic interpolation of the surrounding pixels (edges continue
across the hole, nothing is copied from elsewhere) plus film grain from a clean patch, limited to
the grain amplitude measured around the hole. An earlier version copied whole patches and pasted
dark pieces of the glasses frame into a face. It can run alone:
`python3 tools/v500clean.py rgb.tif ir.tif cleaned.tif`.

## Limits

- The transparency area is 68.58 x 236.98 mm. Requests outside it fail with `Invalid argument`.
- Infrared works at 800, 1600 and 3200 dpi (3200 x 2400 pixels per inch) and returns one gray frame.
  Other color resolutions need the infrared frame resampled by the caller.
- Negative Film compensates the orange mask but does not invert.

Details of the library interface are in [docs/INTERFACE.md](docs/INTERFACE.md).

## License

GPL-2.0-or-later (a derivative of SANE 1.4.0). See [LICENSE](LICENSE).
