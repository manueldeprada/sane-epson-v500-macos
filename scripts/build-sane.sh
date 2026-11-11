#!/usr/bin/env bash
# Build SANE 1.4.0 with the epson2 backend only, patched to drive the Epson
# Perfection V500 Photo through Epson's interpreter.
#
# Usage: scripts/build-sane.sh [prefix]
#
# The default prefix is $HOME/.local/sane-epson-v500. Requires Xcode command
# line tools and Homebrew packages: libusb, pkgconf.

set -euo pipefail

SANE_VERSION="1.4.0"
SANE_URL="https://gitlab.com/-/project/429008/uploads/843c156420e211859e974f78f64c3ea3/sane-backends-1.4.0.tar.gz"
SANE_SHA256="f99205c903dfe2fb8990f0c531232c9a00ec9c2c66ac7cb0ce50b4af9f407a72"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PREFIX="${1:-$HOME/.local/sane-epson-v500}"
WORK="${TMPDIR:-/tmp}/sane-epson-v500-build.$$"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this patch supports macOS only" >&2
  exit 1
fi

if command -v brew >/dev/null 2>&1; then
  BREW_PREFIX="$(brew --prefix)"
elif [[ -d /opt/homebrew ]]; then
  BREW_PREFIX=/opt/homebrew
else
  BREW_PREFIX=/usr/local
fi
export PKG_CONFIG_PATH="$BREW_PREFIX/lib/pkgconfig:$BREW_PREFIX/opt/libusb/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

for tool in curl patch make pkg-config; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool is required (brew install pkgconf)" >&2
    exit 1
  }
done
pkg-config --exists libusb-1.0 || {
  echo "error: libusb is required (brew install libusb)" >&2
  exit 1
}

trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"

echo "==> downloading SANE $SANE_VERSION"
curl -fsSL -o "$WORK/sane.tar.gz" "$SANE_URL"
echo "$SANE_SHA256  $WORK/sane.tar.gz" | shasum -a 256 -c - >/dev/null || {
  echo "error: checksum mismatch for the SANE tarball" >&2
  exit 1
}
tar -xzf "$WORK/sane.tar.gz" -C "$WORK"
SRC="$WORK/sane-backends-$SANE_VERSION"

echo "==> applying patches"
for patch_file in "$ROOT"/patches/*.patch; do
  patch -d "$SRC" -p1 < "$patch_file"
done

echo "==> configuring (epson2 only) into $PREFIX"
cd "$SRC"
./configure \
  --prefix="$PREFIX" \
  --sysconfdir="$PREFIX/etc" \
  --localstatedir="$PREFIX/var" \
  --enable-local-backends \
  BACKENDS="epson2" \
  --without-gphoto2 \
  --with-usb=yes \
  --disable-avahi \
  --disable-locking \
  CPPFLAGS="-I$BREW_PREFIX/include" \
  LDFLAGS="-L$BREW_PREFIX/lib" >"$WORK/configure.log" 2>&1 || {
  tail -20 "$WORK/configure.log" >&2
  exit 1
}

echo "==> building"
make -j"$(sysctl -n hw.ncpu)" >"$WORK/make.log" 2>&1 || {
  tail -20 "$WORK/make.log" >&2
  exit 1
}
make install >"$WORK/install.log" 2>&1 || {
  tail -20 "$WORK/install.log" >&2
  exit 1
}

# Only the epson2 backend is built, so list only it and keep the stock
# "usb" line of epson2.conf (the patch adds the V500 USB ID to the backend).
mkdir -p "$PREFIX/etc/sane.d"
printf 'epson2\n' > "$PREFIX/etc/sane.d/dll.conf"
printf 'usb\n' > "$PREFIX/etc/sane.d/epson2.conf"

cat <<EOF

Installed to $PREFIX

Try it (the scanner must be on and no other program may hold it):

  export SANE_CONFIG_DIR="$PREFIX/etc/sane.d"
  "$PREFIX/bin/scanimage" -L

Epson's ICA driver for the V500 must be installed. See README.md.
EOF
