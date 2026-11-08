#!/bin/bash
# Removes the installed input method and the data it wrote.
# Run this file; it asks for the administrator password once.
# Copied out of Typing.app first: the script lives in the bundle it deletes.
set -eu

if [ "${TYPING_UNINSTALL_COPY:-}" != 1 ]; then
    app=$(CDPATH= cd "$(dirname "$0")/../.." && pwd)
    copy=$(mktemp /tmp/typing-uninstall.XXXXXX)
    cp "$0" "$copy"
    chmod +x "$copy"
    TYPING_UNINSTALL_APP="$app" TYPING_UNINSTALL_COPY=1 exec "$copy" "$@"
fi

helper="${TYPING_UNINSTALL_APP:-}/Contents/MacOS/disable-input-source"

# One password. Cancel use, then delete, inside that call, so no second dialog can start the input method again.
# Canceling use can fail. The files are still deleted.
user=$(id -un)
user_id=$(id -u "$user")
root=$(mktemp /tmp/typing-uninstall-root.XXXXXX)
cat > "$root" <<EOF
#!/bin/bash
set -eu
if [ -x "$helper" ]; then
    launchctl asuser "$user_id" sudo -u "$user" "$helper" || true
fi
pkill -x Typing 2>/dev/null || true
rm -rf "/Library/Input Methods/Typing.app"
pkgutil --forget is.chats.inputmethod.Typing.pkg || true
EOF
chmod +x "$root"
osascript -e "do shell script \"/bin/bash $root\" with administrator privileges"
rm -f "$root"
rm -rf "$HOME/Library/Input Methods/Typing.app"
# After the files are gone. Unregistering while the bundle is still on disk leaves a "not found" row.
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
for app in "/Library/Input Methods/Typing.app" "$HOME/Library/Input Methods/Typing.app"; do
    "$LSREGISTER" -u "$app" || true
done
# The container directory and the system metadata file inside it cannot be removed.
# Delete the files the program wrote.
container="$HOME/Library/Containers/is.chats.inputmethod.Typing"
if [ -d "$container" ]; then
    find "$container" -mindepth 1 -maxdepth 1 ! -name '.com.apple.containermanagerd.metadata.plist' -exec rm -rf {} +
fi
