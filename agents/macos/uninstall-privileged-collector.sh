#!/bin/sh
# Undo install-privileged-collector.sh. An ordinary user runs this; sudo
# prompts to bootout and delete the LaunchDaemon. The user LaunchAgent is
# not stopped or removed.
set -eu

LABEL=ai.onyx.sparkdash-mac-agent.privileged
DEST="/Library/LaunchDaemons/${LABEL}.plist"
USER_AGENT="${HOME:-}/Library/LaunchAgents/ai.onyx.sparkdash-mac-agent.plist"

echo "Removing $LABEL. The user LaunchAgent is not touched."
if [ -n "${HOME:-}" ] && [ -f "$USER_AGENT" ]; then
  echo "Leaving $USER_AGENT in place."
fi

if sudo launchctl print "system/${LABEL}" >/dev/null 2>&1; then
  sudo launchctl bootout "system/${LABEL}" || true
fi
sudo rm -f "$DEST"
echo "Privileged collector removed. The user agent, if installed, is unchanged."
