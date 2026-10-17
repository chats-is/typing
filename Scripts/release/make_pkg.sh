#!/bin/bash
# Builds a standard macOS installer package: Scripts/release/make_pkg.sh <version> [out dir]
# Installs Typing.app into /Library/Input Methods (all users) and registers it for the person
# running the installer, so it appears in the input menu right away.
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION="${1:?usage: make_pkg.sh <version> [out dir]}"
OUT="${2:-dist}"
[ -d build/Typing.app ] || Scripts/build.sh
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/root/Library/Input Methods" "$STAGE/scripts" "$OUT"
cp -R build/Typing.app "$STAGE/root/Library/Input Methods/"
mkdir -p build/tools && swiftc -O Tools/register.swift -o build/tools/register
cp build/tools/register "$STAGE/scripts/register"
cat > "$STAGE/scripts/postinstall" <<'POST'
#!/bin/bash
# Register and enable the input method for the person who is logged in (the installer runs as root).
APP="/Library/Input Methods/Typing.app"
USER_NAME="$(stat -f%Su /dev/console)"
pkill -x Typing 2>/dev/null || true
if [ -n "$USER_NAME" ] && [ "$USER_NAME" != "root" ]; then
    # A copy from an old per-user install would show up as a second "智能输入".
    sudo -u "$USER_NAME" rm -rf "/Users/$USER_NAME/Library/Input Methods/Typing.app" 2>/dev/null || true
    sudo -u "$USER_NAME" "$(dirname "$0")/register" "$APP" || true
fi
exit 0
POST
chmod +x "$STAGE/scripts/postinstall" "$STAGE/scripts/register"
pkgbuild --root "$STAGE/root" --scripts "$STAGE/scripts" \
    --identifier is.chats.inputmethod.Typing.pkg --version "$VERSION" --install-location / \
    "$OUT/Typing-$VERSION.pkg" >/dev/null
rm -rf "$STAGE"
(cd "$OUT" && shasum -a 256 "Typing-$VERSION.pkg" > "Typing-$VERSION.pkg.sha256")
ls -la "$OUT/Typing-$VERSION.pkg"
