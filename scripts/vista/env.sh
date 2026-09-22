#!/usr/bin/env bash

module load nvidia/25.3 cuda/12.9
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH="$PATH:$HOME/.local/bin"
export TB_ROOT="$STOCKYARD/tensorboard"
export TB_SYSTEM=vista
