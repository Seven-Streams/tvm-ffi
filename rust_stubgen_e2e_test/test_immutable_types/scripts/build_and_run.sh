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

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TEST_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

# Build C++ shared library
cmake -S "$TEST_DIR" -B "$TEST_DIR/build"
cmake --build "$TEST_DIR/build" --parallel

# Generate Rust bindings (optional - for testing stubgen)
uv run tvm-ffi-stubgen "$TEST_DIR/rust/src/generated" --target rust \
  --dlls "$TEST_DIR/build/test_immutable_types.so" \
  --verbose

echo "Build completed successfully!"
