#!/usr/bin/env bash
# Internetli mashinada bajariladi. Target: Windows x64 + Python 3.11.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
pip download -r "$HERE/../../backend/requirements.txt" \
    -d "$HERE/wheels" \
    --platform win_amd64 \
    --python-version 3.11 \
    --only-binary=:all:
echo "Wheels yuklandi: $HERE/wheels"
echo "Endi butun 'offline-wheels' papkasini air-gapped mashinaga ko'chiring."
