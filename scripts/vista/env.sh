#!/usr/bin/env bash

if ! command -v module >/dev/null 2>&1; then
    echo "Error: scripts/vista/env.sh requires the TACC module environment." >&2
    return 1 2>/dev/null || exit 1
fi
if ! module load nvidia/25.3 cuda/12.9; then
    echo "Error: scripts/vista/env.sh could not load the Vista NVIDIA/CUDA modules." >&2
    return 1 2>/dev/null || exit 1
fi

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$PATH:$HOME/.local/bin" ;;
esac
export TB_ROOT="$STOCKYARD/tensorboard"
export TB_SYSTEM=vista
