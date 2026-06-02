<!---
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# cpp_rust_test

Minimal C++/Rust interop demo for tvm-ffi:

- C++ defines `ExprObj` with one field `value: int64_t`.
- C++ exports `cpp_rust_test.make_expr`.
- Rust loads the C++ shared library, receives `Expr`, and mutates `value` directly.

## Build C++ shared library

```bash
cmake -S cpp_rust_test -B cpp_rust_test/build
cmake --build cpp_rust_test/build
```

## Run Rust program

```bash
cd cpp_rust_test/rust
cargo run
```

Expected output includes:

```text
created Expr.value = 42
after Rust mutation Expr.value = 50
cpp_rust_test demo OK
```
# cpp_rust_test

Minimal C++ / Rust demo: C++ defines `Expr` with one `int64_t value` field; Rust loads the
shared library, creates an `Expr` via a global function, and mutates `value` in place.

## Prerequisites

From the tvm-ffi repo root:

```bash
uv pip install -e .
export LD_LIBRARY_PATH="$(tvm-ffi-config --libdir):${LD_LIBRARY_PATH:-}"
```

## Build and run

```bash
./cpp_rust_test/scripts/build_and_run.sh
```

Or step by step:

```bash
cmake -S cpp_rust_test -B cpp_rust_test/build
cmake --build cpp_rust_test/build --parallel
cd cpp_rust_test/rust && cargo run --release
```

Override the Expr library path:

```bash
export CPP_RUST_TEST_EXPR_LIB=/path/to/cpp_rust_test_expr.so
```

## Layout

| Path | Role |
|------|------|
| `cpp/expr_lib.cc` | C++ `ExprObj` / `Expr` + reflection + `cpp_rust_test.make_expr` |
| `rust/src/main.rs` | Matching `#[repr(C)]` type + field mutation |
| `CMakeLists.txt` | Builds `cpp_rust_test_expr.so` into `build/` |
