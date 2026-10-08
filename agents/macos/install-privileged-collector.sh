#!/bin/sh
# Opt-in install of the root Mac collector. An ordinary user runs this; sudo
# prompts for the copy and launchctl bootstrap. The user LaunchAgent is not
# read or written.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
LABEL=ai.onyx.sparkdash-mac-agent.privileged
TEMPLATE="$SCRIPT_DIR/ai.onyx.sparkdash-mac-agent.privileged.plist"
DEST="/Library/LaunchDaemons/${LABEL}.plist"
USER_AGENT="${HOME:-}/Library/LaunchAgents/ai.onyx.sparkdash-mac-agent.plist"

if [ ! -f "$TEMPLATE" ]; then
  echo "missing plist template: $TEMPLATE" >&2
  exit 1
fi

echo "Installing $LABEL from $REPO_ROOT"
echo "This asks for sudo to copy a LaunchDaemon and bootstrap it."
echo "It does not modify the user LaunchAgent${USER_AGENT:+ at $USER_AGENT}."
if [ -n "${HOME:-}" ] && [ -f "$USER_AGENT" ]; then
  echo "User agent is installed and will be left running on its own port."
fi

TMP=$(mktemp "${TMPDIR:-/tmp}/sparkdash-mac-agent-privileged.XXXXXX.plist")
trap 'rm -f "$TMP"' EXIT
python3 - "$TEMPLATE" "$TMP" "$REPO_ROOT" <<'PY'
import pathlib, sys
template, dest, root = sys.argv[1:]
text = pathlib.Path(template).read_text(encoding="utf-8")
pathlib.Path(dest).write_text(text.replace("REPO_ROOT", root), encoding="utf-8")
PY

sudo cp "$TMP" "$DEST"
sudo chown root:wheel "$DEST"
sudo chmod 644 "$DEST"
if sudo launchctl print "system/${LABEL}" >/dev/null 2>&1; then
  sudo launchctl bootout "system/${LABEL}" || true
fi
sudo launchctl bootstrap system "$DEST"
sudo launchctl enable "system/${LABEL}" || true
sudo launchctl kickstart -k "system/${LABEL}" || true
echo "Privileged collector is on http://127.0.0.1:8791/metrics"
echo "Point the sparkDash Mac node agent port at 8791 to read temperatures."
echo "Uninstall with: agents/macos/uninstall-privileged-collector.sh"
