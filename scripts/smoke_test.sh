#!/usr/bin/env bash
set -euo pipefail

python -m compileall src
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
