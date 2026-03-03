#!/bin/bash
# Helper script to apply patches to the BAT source directory.
# Usage: patch_bat.sh [source directory] [patch directory]

set -e

SOURCE_DIR=$1
PATCH_DIR=$2

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist."
    exit 1
fi

if [ ! -d "$PATCH_DIR" ]; then
    echo "Error: Patch directory $PATCH_DIR does not exist."
    exit 1
fi

cd "$SOURCE_DIR"

for p in "$PATCH_DIR"/*.patch; do
    if [ -f "$p" ]; then
        echo "Applying patch: $p"
        patch -p1 < "$p"
    fi
done
