# Interpreter interface

Observed on one V500 Photo with Epson's macOS ICA driver (files dated 2023-09-07), by running the
library and reading its exported symbols and disassembly. Not from Epson documentation.

The library is `EPSON Scanner.app/Contents/PlugIns/Interpreter 7C.bundle/Contents/MacOS/Interpreter 7C`
(universal, `7C` is the GT-X770). The same directory holds `Interpreter XX` libraries for other
models; they were not tried.

## Calling convention

```c
unsigned char INTInit(void *read_cb, void *write_cb, void *refcon);
unsigned char INTClose(void);
unsigned char INTWrite(void *buf, unsigned int size);   /* ESC/I bytes to the scanner */
unsigned char INTRead(void *buf, unsigned int size);    /* ESC/I bytes from the scanner */

unsigned char cb(void *buf, unsigned int size, void *refcon, void *aux);
```

- The library does not open the USB device. Each callback does one bulk transfer: IN `0x81` for
  `read_cb`, OUT `0x02` for `write_cb`. There is no control transfer.
- Callbacks return 1 or 0. Only the low byte counts, so returning a length fails for multiples of 256.
- `INTInit` initializes the scanner through the callbacks and takes a few seconds.
- `INTRead` can write past the requested size. Read into a padded buffer and copy.
- Read an info-block header (`STX`, status, 16-bit length) with one `INTRead(buf, 4)`, then the data
  with a second call.
- Commands with a parameter: write the 2-byte command, read ACK `06`, write the parameter, read ACK.
  NAK is `15`.

## Scanner behavior

- It reports command level `D2`. `FS I` is not supported.
- The transparency area in `ESC f` is 8640 x 29856 in 1/3200 inch (68.58 x 236.98 mm). Dividing by
  the 1600 dpi maximum gives twice the real width.
- 200, 400, 800, 1600 and 6400 dpi. The reported maximum depth is 8 bit, but 16 bit is accepted.
- Each scan is `FS W` plus a 64-byte block (the layout `epson2` uses): resolution at bytes 0-7,
  offset 8-15, size 16-23, color 24 (`0x13` RGB, `0` gray), depth 25, option control 26, lines per
  block 28, gamma 29, color correction 31, film type 37.
- Option control 1 is the transparency unit, 3 is infrared, 4 returns RGB like 1.
- **Infrared** accepts main resolution 800, 1600 or 3200 and sub resolution 800, 1600 or 2400. Below
  3200 they are equal; at 3200 the sub resolution is 2400. Everything else (6400, a square 3200) is
  refused. It returns one gray frame. Gamma 3 with color correction 0 is accepted. No `ESC #` unlock
  is needed.
- Same area and resolution: the infrared frame lines up with RGB within 1 pixel (800 and 1600 dpi).
- Film types 0 to 3 are accepted. Negative film compensates the orange mask and does not invert.

## A stuck scanner

An aborted scan can make the next `INTInit` wait about 60 seconds and fail. `libusb_reset_device`
recovers it (the device then re-enumerates at a new address), as does a power cycle.
