<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!--- -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Rust Stubgen Example — `IntPair`

A minimal, end-to-end walk-through of the Rust stubgen target
(`tvm-ffi-stubgen --target rust`), documented in
[`docs/packaging/stubgen.rst`](../../docs/packaging/stubgen.rst).

It registers a single C++ object, `IntPair` (two `i64` fields `a`/`b`, plus
methods `sum`, `scale`, `swapped`, `checked_div`), generates Rust bindings for
it via the CLI, and runs a demo that calls those bindings.

## Layout

```text
cpp/int_pair.cc          C++ extension registering rust_stubgen.IntPair
CMakeLists.txt           builds the shared library; also wires stubgen via CMake
scripts/run_stubgen.sh   one-shot: build -> stubgen (CLI) -> run demo
rust/                    Rust crate that consumes the generated bindings
  build.rs               locates the tvm-ffi runtime + this example's library
  src/lib.rs             mounts `generated`, dlopens the C++ library once
  src/generated/         produced by stubgen (do not edit by hand)
  examples/demo.rs       runnable proof that the bindings work
```

## Run it

Install the Python package first (from the repo root), which provides
`tvm-ffi-stubgen` and the runtime:

```bash
uv pip install -e .
```

Then run the one-shot driver:

```bash
bash examples/rust_stubgen/scripts/run_stubgen.sh
```

It performs three steps:

1. **Build** the C++ shared library with CMake.
2. **Generate** the Rust bindings by invoking the CLI directly:

   ```bash
   tvm-ffi-stubgen rust/src/generated \
     --target rust \
     --dlls build/rust_stubgen.so \
     --init-lib rust_stubgen \
     --init-pypkg rust_stubgen \
     --init-prefix "rust_stubgen." \
     --verbose
   ```

3. **Run the demo** (`cargo run --example demo`), which constructs an `IntPair`,
   calls its methods, mutates a field through `DerefMut`, exercises an
   object-returning method, and checks that a C++ throw surfaces as `Err`.

Expected output ends with:

```text
All IntPair binding checks passed.
```

## CMake alternative

The same generation runs as a post-build step from `CMakeLists.txt` via
`tvm_ffi_configure_target(... STUB_TARGET rust STUB_INIT ON)`, so a plain
`cmake --build` regenerates `rust/src/generated` too. The CLI script exists to
show the standalone `tvm-ffi-stubgen` invocation that the CMake helper wraps.
