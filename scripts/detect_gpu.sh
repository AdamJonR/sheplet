#!/usr/bin/env bash
# Detect GPU features for cargo build.
# Source this file; it sets FEATURES and CARGO_FEATURES.

FEATURES=""
if [[ "$(uname -s)" == "Darwin" ]]; then
    FEATURES="metal,accelerate"
elif command -v nvidia-smi &>/dev/null; then
    FEATURES="cuda"
fi

if [ -n "$FEATURES" ]; then
    echo "GPU features: $FEATURES"
    CARGO_FEATURES="--features $FEATURES"
else
    echo "GPU features: none (CPU only)"
    CARGO_FEATURES=""
fi
