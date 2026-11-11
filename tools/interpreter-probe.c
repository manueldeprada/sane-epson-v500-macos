/*
 * interpreter-probe: talk to an Epson Perfection V500 Photo through the
 * interpreter library of Epson's macOS ICA driver, without SANE.
 *
 * It loads the library, hands it two libusb callbacks, then sends the ESC/I
 * identity (ESC I) and extended status (ESC f) commands and prints the
 * answers. Use it to check that the interpreter loads on your machine and
 * that the scanner answers.
 *
 * Build:
 *   clang -arch arm64 interpreter-probe.c -o interpreter-probe \
 *     $(pkg-config --cflags --libs libusb-1.0)
 *
 * Run (no other program may hold the scanner):
 *   ./interpreter-probe [path to Interpreter 7C]
 *
 * Copyright (C) 2026 the sane-epson-v500-macos contributors
 * License: GPL-2.0-or-later
 */

#include <dlfcn.h>
#include <libusb.h>
#include <stdio.h>
#include <string.h>

#define EPSON_VENDOR 0x04b8
#define V500_PRODUCT 0x0130
#define EP_IN 0x81
#define EP_OUT 0x02
#define TIMEOUT_MS 30000

#define DEFAULT_LIBRARY                                                       \
	"/Library/Image Capture/Devices/EPSON Scanner.app/Contents/PlugIns/" \
	"Interpreter 7C.bundle/Contents/MacOS/Interpreter 7C"

/* INTRead can write past the requested size, so reads go through padding. */
#define READ_SLACK 64

static libusb_device_handle *usb;

/*
 * Callbacks have the shape bool cb(buf, size, refcon, aux). Only the low byte
 * of the return value counts: return 1 or 0, never a length.
 */
static unsigned char
usb_read(void *buf, unsigned int size, void *refcon, void *aux)
{
	int done = 0;

	(void) refcon;
	(void) aux;
	return libusb_bulk_transfer(usb, EP_IN, buf, (int) size, &done,
				    TIMEOUT_MS) == 0 && done == (int) size;
}

static unsigned char
usb_write(void *buf, unsigned int size, void *refcon, void *aux)
{
	int done = 0;

	(void) refcon;
	(void) aux;
	return libusb_bulk_transfer(usb, EP_OUT, buf, (int) size, &done,
				    TIMEOUT_MS) == 0 && done == (int) size;
}

typedef unsigned char (*init_fn)(void *, void *, void *);
typedef unsigned char (*close_fn)(void);
typedef unsigned char (*io_fn)(void *, unsigned int);

static io_fn int_read, int_write;

static int
read_exact(void *dst, unsigned int size)
{
	unsigned char scratch[4096 + READ_SLACK] = { 0 };

	if (size > 4096 || !int_read(scratch, size))
		return 0;
	memcpy(dst, scratch, size);
	return 1;
}

/* Send a two-byte command and read the info block it returns. */
static int
info_block(unsigned char cmd, unsigned char prefix, unsigned char *data,
	   unsigned int max, unsigned int *len)
{
	unsigned char command[2] = { prefix, cmd };
	unsigned char header[4];

	if (!int_write(command, 2) || !read_exact(header, 4) || header[0] != 0x02)
		return 0;
	*len = header[2] | (header[3] << 8);
	if (*len > max || !read_exact(data, *len))
		return 0;
	return 1;
}

int
main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : DEFAULT_LIBRARY;
	unsigned char data[256];
	unsigned int len;
	init_fn init;
	close_fn close_interpreter;
	void *lib;

	if (libusb_init(NULL) != 0)
		return 1;
	usb = libusb_open_device_with_vid_pid(NULL, EPSON_VENDOR, V500_PRODUCT);
	if (usb == NULL) {
		fprintf(stderr, "no V500 found (or another program holds it)\n");
		return 1;
	}
	if (libusb_claim_interface(usb, 0) != 0) {
		fprintf(stderr, "cannot claim the USB interface\n");
		return 1;
	}

	lib = dlopen(path, RTLD_NOW);
	if (lib == NULL) {
		fprintf(stderr, "cannot load %s: %s\n", path, dlerror());
		return 1;
	}
	init = (init_fn) dlsym(lib, "INTInit");
	close_interpreter = (close_fn) dlsym(lib, "INTClose");
	int_read = (io_fn) dlsym(lib, "INTRead");
	int_write = (io_fn) dlsym(lib, "INTWrite");
	if (!init || !close_interpreter || !int_read || !int_write) {
		fprintf(stderr, "%s does not export the interpreter interface\n", path);
		return 1;
	}

	if (!init(usb_read, usb_write, NULL)) {
		fprintf(stderr, "INTInit failed; power-cycle the scanner and retry\n");
		return 1;
	}

	if (info_block('I', 0x1b, data, sizeof(data), &len) && len >= 2)
		printf("ESC I: command level %c%c, %u bytes\n", data[0], data[1], len);
	else
		fprintf(stderr, "ESC I failed\n");

	if (info_block('f', 0x1b, data, sizeof(data), &len) && len >= 0x2a) {
		printf("ESC f: model %.16s\n", (const char *) data + 0x1a);
		printf("ESC f: transparency unit %s\n",
		       (data[6] & 0x80) ? "installed" : "not installed");
		printf("ESC f: transparency area %u x %u (1/3200 inch)\n",
		       data[7] | (data[8] << 8), data[9] | (data[10] << 8));
	} else {
		fprintf(stderr, "ESC f failed\n");
	}

	close_interpreter();
	libusb_release_interface(usb, 0);
	libusb_close(usb);
	libusb_exit(NULL);
	return 0;
}
