#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# CLI-driven Rust stub generation for the IntPair example.
#
# Steps:
#   1. build the C++ shared library with CMake,
#   2. run `tvm-ffi-stubgen --target rust` against it to (re)generate the
#      `rust/src/generated` module tree,
#   3. run the demo (`cargo run --example demo`) to prove the bindings work.
#
# Usage:
#   bash scripts/run_stubgen.sh

set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"

# --- shared library suffix for the current platform --------------------------
case "$(uname -s)" in
  Darwin) LIB_SUFFIX="dylib" ;;
  MINGW* | MSYS* | CYGWIN*) LIB_SUFFIX="dll" ;;
  *) LIB_SUFFIX="so" ;;
esac

# --- activate the project virtualenv -----------------------------------------
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
else
  echo "No .venv found at $REPO_ROOT/.venv; install the package first" \
    "(uv pip install -e . from the repo root)." >&2
  exit 1
fi

BUILD_DIR="$EXAMPLE_DIR/build"
LIB_PATH="$BUILD_DIR/rust_stubgen.$LIB_SUFFIX"
GENERATED_DIR="$EXAMPLE_DIR/rust/src/generated"

echo "==> [1/3] build the C++ shared library"
cmake -S "$EXAMPLE_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --parallel

echo "==> [2/3] generate Rust bindings with tvm-ffi-stubgen (CLI)"
# Wipe the generated tree so we exercise from-scratch init mode. Without the
# directory, stubgen must run with --init-* to recreate the scaffolding; a plain
# (non-init) run only refreshes existing directive blocks. The --init-* values
# mirror the CMake STUB_* options in CMakeLists.txt.
rm -rf "$GENERATED_DIR"
tvm-ffi-stubgen "$GENERATED_DIR" \
  --target rust \
  --dlls "$LIB_PATH" \
  --init-lib rust_stubgen \
  --init-pypkg rust_stubgen \
  --init-prefix "rust_stubgen." \
  --verbose

echo "==> [3/3] run the demo against the generated bindings"
( cd "$EXAMPLE_DIR/rust" && cargo run --example demo )

echo "Done."
