"""Keep the installed user-agent listener private by default."""

import plistlib
from pathlib import Path
import unittest


class LaunchAgentTemplateTests(unittest.TestCase):
    def test_template_is_valid_and_binds_only_loopback(self):
        template = Path(__file__).resolve().parents[1] / "ai.onyx.sparkdash-mac-agent.plist"
        with template.open("rb") as source:
            args = plistlib.load(source)["ProgramArguments"]
        self.assertEqual(args.count("--host"), 1)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertEqual(args[args.index("--port") + 1], "8790")
