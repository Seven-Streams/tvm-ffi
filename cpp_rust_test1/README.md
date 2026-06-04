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

- `ExprObj`: `value: int64_t`
- `AddObj`: `a: Expr`, `b: Expr`, `value: int64_t`, method `update()` (C++ recomputes `value`)
- Construction is via reflected constructors (`__ffi_init__`), not global factory funcs
- Rust mirrors both with `#[repr(C)]` and mutates the same C++ heap objects

## Prerequisites

From the tvm-ffi repo root:

```bash
uv pip install -e .
export LD_LIBRARY_PATH="$(tvm-ffi-config --libdir):${LD_LIBRARY_PATH:-}"
```

## Build and run

```bash
cmake -S cpp_rust_test -B cpp_rust_test/build
cmake --build cpp_rust_test/build --parallel
cd cpp_rust_test/rust && cargo run --release
```

Override the shared library path:

```bash
export CPP_RUST_TEST_EXPR_LIB=/path/to/cpp_rust_test_expr.so
```

## Layout

| Path | Role |
| ------ | ------ |
| `cpp/expr_lib.cc` | C++ `Expr` / `Add` + reflection + factories |
| `rust/src/main.rs` | `Expr` / `Add` bindings + demo |
| `CMakeLists.txt` | Builds `cpp_rust_test_expr.so` into `build/` |
