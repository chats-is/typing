#!/bin/bash
# Builds a standard macOS installer package: Scripts/release/make_pkg.sh <version> [out dir]
# Installs Zhiyin.app into /Library/Input Methods (all users) and registers it for the person
# running the installer, so it appears in the input menu right away.
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION="${1:?usage: make_pkg.sh <version> [out dir]}"
OUT="${2:-dist}"
[ -d build/Zhiyin.app ] || Scripts/build.sh
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/root/Library/Input Methods" "$STAGE/scripts" "$OUT"
cp -R build/Zhiyin.app "$STAGE/root/Library/Input Methods/"
mkdir -p build/tools && swiftc -O Tools/register.swift -o build/tools/register
cp build/tools/register "$STAGE/scripts/register"
cat > "$STAGE/scripts/postinstall" <<'POST'
#!/bin/bash
# Register and enable the input method for the person who is logged in (the installer runs as root).
APP="/Library/Input Methods/Zhiyin.app"
USER_NAME="$(stat -f%Su /dev/console)"
pkill -x Zhiyin 2>/dev/null || true
if [ -n "$USER_NAME" ] && [ "$USER_NAME" != "root" ]; then
    sudo -u "$USER_NAME" "$(dirname "$0")/register" "$APP" || true
fi
exit 0
POST
chmod +x "$STAGE/scripts/postinstall" "$STAGE/scripts/register"
pkgbuild --root "$STAGE/root" --scripts "$STAGE/scripts" \
    --identifier is.chats.zhiyin.pkg --version "$VERSION" --install-location / \
    "$OUT/智音输入法-$VERSION.pkg" >/dev/null
rm -rf "$STAGE"
(cd "$OUT" && shasum -a 256 "智音输入法-$VERSION.pkg" > "智音输入法-$VERSION.pkg.sha256")
ls -la "$OUT/智音输入法-$VERSION.pkg"
