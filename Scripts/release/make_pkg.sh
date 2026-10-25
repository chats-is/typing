#!/bin/bash
# Builds a standard macOS installer package: Scripts/release/make_pkg.sh <version> [out dir]
# Installs Typing.app into /Library/Input Methods (all users) and registers it for the account
# running the installer, so it appears in the input menu right away. Apple silicon, macOS 15 or later.
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION="${1:?usage: make_pkg.sh <version> [out dir]}"
OUT="${2:-dist}"
# Always build: a stale build would ship an older Info.plist (version) or binary.
Scripts/build.sh
source Scripts/app_path.sh
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/root/Library/Input Methods" "$STAGE/scripts" "$OUT"
cp -R "$APP" "$STAGE/root/Library/Input Methods/"
mkdir -p build/tools && swiftc -O Tools/register.swift -o build/tools/register
cp build/tools/register "$STAGE/scripts/register"
# Stop a running copy before the payload replaces its files (an input method keeps its lexicon
# open; swapping the files under it breaks typing until it is relaunched).
cat > "$STAGE/scripts/preinstall" <<'PRE'
#!/bin/bash
pkill -x Typing 2>/dev/null || true
exit 0
PRE
cat > "$STAGE/scripts/postinstall" <<'POST'
#!/bin/bash
# Register and enable the input method for the logged-in account (the installer runs as root).
APP="/Library/Input Methods/Typing.app"
USER_NAME="$(stat -f%Su /dev/console)"
# A copy the system relaunched while the payload was being written still holds the old files:
# stop it (only that one; the launch agent counts every exit, so no needless kills).
INSTALLED_AT="$(stat -f%m "$APP/Contents/MacOS/Typing" 2>/dev/null || echo 0)"
for PID in $(pgrep -x Typing); do
    STARTED="$(LC_ALL=C ps -o lstart= -p "$PID" | xargs -I{} env LC_ALL=C date -j -f '%a %b %d %T %Y' '{}' +%s 2>/dev/null || echo 0)"
    [ "$STARTED" -lt "$INSTALLED_AT" ] && kill "$PID" 2>/dev/null || true
done
if [ -n "$USER_NAME" ] && [ "$USER_NAME" != "root" ]; then
    USER_ID="$(id -u "$USER_NAME")"
    USER_HOME="$(dscl . -read "/Users/$USER_NAME" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
    # A copy from an old per-user install would show up as a second "智能输入".
    [ -n "$USER_HOME" ] && rm -rf "$USER_HOME/Library/Input Methods/Typing.app" 2>/dev/null || true
    # Registering and enabling talk to the logged-in account's Text Input Sources session (and may open System
    # Settings), so run inside that login session, not in the installer's root session.
    launchctl asuser "$USER_ID" sudo -u "$USER_NAME" "$(dirname "$0")/register" "$APP" || true
fi
exit 0
POST
chmod +x "$STAGE/scripts/preinstall" "$STAGE/scripts/postinstall" "$STAGE/scripts/register"
pkgbuild --root "$STAGE/root" --scripts "$STAGE/scripts" \
    --identifier is.chats.inputmethod.Typing.pkg --version "$VERSION" --install-location / \
    "$STAGE/Typing.pkg" >/dev/null
# The installer refuses Intel Macs and systems older than the app's minimum, with the system's own
# "can't be installed on this computer" message, instead of installing an input method that cannot run.
MIN_OS="$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' Bundle/Info.plist)"
cat > "$STAGE/distribution.xml" <<DIST
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>智能输入法</title>
    <options hostArchitectures="arm64" customize="never" require-scripts="false"/>
    <allowed-os-versions>
        <os-version min="$MIN_OS"/>
    </allowed-os-versions>
    <domains enable_localSystem="true"/>
    <choices-outline>
        <line choice="is.chats.inputmethod.Typing.pkg"/>
    </choices-outline>
    <choice id="is.chats.inputmethod.Typing.pkg" visible="false">
        <pkg-ref id="is.chats.inputmethod.Typing.pkg"/>
    </choice>
    <pkg-ref id="is.chats.inputmethod.Typing.pkg" version="$VERSION" onConclusion="none">Typing.pkg</pkg-ref>
</installer-gui-script>
DIST
productbuild --distribution "$STAGE/distribution.xml" --package-path "$STAGE" "$OUT/Typing-$VERSION.pkg" >/dev/null
rm -rf "$STAGE"
(cd "$OUT" && shasum -a 256 "Typing-$VERSION.pkg" > "Typing-$VERSION.pkg.sha256")
ls -la "$OUT/Typing-$VERSION.pkg"
