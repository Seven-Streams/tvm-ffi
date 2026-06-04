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
# One-shot driver for the Rust stubgen E2E suite. For every `test_*` module it:
#   1. builds the C++ shared library with CMake,
#   2. regenerates the Rust bindings via `tvm-ffi-stubgen --target rust`,
#   3. runs the module's `cargo test`.
#
# Usage:
#   bash run_all.sh                 # all modules
#   bash run_all.sh test_scalar_types test_object_hierarchy
#   SKIP_STUBGEN=1 bash run_all.sh  # build + test, but do not regenerate stubs
#   SKIP_BUILD=1   bash run_all.sh  # skip the C++ CMake build step

set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SUITE_DIR/.." && pwd)"

# --- colors (disabled when stdout is not a TTY) ------------------------------
if [[ -t 1 ]]; then
  C_BOLD=$'\033[1m'
  C_GREEN=$'\033[32m'
  C_RED=$'\033[31m'
  C_CYAN=$'\033[36m'
  C_RESET=$'\033[0m'
else
  C_BOLD="" C_GREEN="" C_RED="" C_CYAN="" C_RESET=""
fi

log() { echo "${C_CYAN}${C_BOLD}==> $*${C_RESET}"; }

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
  echo "${C_RED}No .venv found at $REPO_ROOT/.venv; install the package first" \
    "(uv pip install -e . from the repo root).${C_RESET}" >&2
  exit 1
fi

# --- module selection --------------------------------------------------------
modules=()
if [[ $# -gt 0 ]]; then
  modules=("$@")
else
  for dir in "$SUITE_DIR"/test_*/; do
    [[ -d "$dir" ]] || continue
    modules+=("$(basename "$dir")")
  done
fi

if [[ ${#modules[@]} -eq 0 ]]; then
  echo "${C_RED}No test_* modules found under $SUITE_DIR${C_RESET}" >&2
  exit 1
fi

passed=()
failed=()

run_module() {
  local name="$1"
  local mod_dir="$SUITE_DIR/$name"
  local build_dir="$mod_dir/build"
  local lib_path="$build_dir/$name.$LIB_SUFFIX"
  local generated_dir="$mod_dir/rust/src/generated"

  if [[ ! -d "$mod_dir" ]]; then
    echo "${C_RED}Skipping unknown module: $name${C_RESET}" >&2
    return 1
  fi

  # NOTE: `set -e` is suppressed inside a function used as an `if` condition,
  # so every step is guarded explicitly to stop the module on first failure.
  if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    log "[$name] cmake configure + build"
    cmake -S "$mod_dir" -B "$build_dir" || return 1
    cmake --build "$build_dir" --parallel || return 1
  fi

  if [[ "${SKIP_STUBGEN:-0}" != "1" ]]; then
    log "[$name] regenerate Rust bindings"
    tvm-ffi-stubgen "$generated_dir" --target rust --dlls "$lib_path" || return 1
  fi

  log "[$name] cargo test"
  ( cd "$mod_dir/rust" && cargo test ) || return 1
}

for name in "${modules[@]}"; do
  if run_module "$name"; then
    passed+=("$name")
  else
    failed+=("$name")
  fi
done

# --- summary -----------------------------------------------------------------
echo
log "Summary"
for name in "${passed[@]:-}"; do
  [[ -n "$name" ]] && echo "  ${C_GREEN}PASS${C_RESET} $name"
done
for name in "${failed[@]:-}"; do
  [[ -n "$name" ]] && echo "  ${C_RED}FAIL${C_RESET} $name"
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "${C_RED}${C_BOLD}${#failed[@]} module(s) failed.${C_RESET}"
  exit 1
fi
echo "${C_GREEN}${C_BOLD}All ${#passed[@]} module(s) passed.${C_RESET}"
