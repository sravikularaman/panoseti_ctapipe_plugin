#!/bin/bash
# Run tests with correct PYTHONPATH
cd "$(dirname "$0")"
PYTHONPATH="$(pwd)/panoseti_ctapipe_plugin" python -m pytest tests/ "$@"
