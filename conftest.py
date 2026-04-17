"""
Workspace-level pytest configuration.

This conftest.py is at the workspace root and ensures the panoseti_ctapipe_plugin
package is importable during test discovery and execution.
"""

import sys
from pathlib import Path

# Add the plugin package to sys.path so imports work
plugin_path = Path(__file__).parent / "panoseti_ctapipe_plugin"
if str(plugin_path) not in sys.path:
    sys.path.insert(0, str(plugin_path))
