#!/usr/bin/env zsh
set -euo pipefail

source ~/tvm-ffi/.venv/bin/activate
cd ~/tvm-ffi
cmake -S cpp_rust_test -B cpp_rust_test/build
cmake --build cpp_rust_test/build --parallel
cd cpp_rust_test/rust && cargo run --release
cd ../..
