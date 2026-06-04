<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Quick Start Guide

## Build All Tests

From the repository root:

```bash
source .venv/bin/activate
cd rust_stubgen_e2e_test

# Build all tests at once
for test in test_*; do
  echo "Building $test..."
  cmake -S "$test" -B "$test/build"
  cmake --build "$test/build" --parallel
done
```

Or build individually:

```bash
# Build one test
cmake -S rust_stubgen_e2e_test/test_scalar_types -B rust_stubgen_e2e_test/test_scalar_types/build
cmake --build rust_stubgen_e2e_test/test_scalar_types/build --parallel
```

## Generate Rust Bindings

After building any test, generate Rust stub code:

```bash
# For a single test
uv run tvm-ffi-stubgen --target rust \
  --dlls rust_stubgen_e2e_test/test_scalar_types/build/test_scalar_types.so \
  --verbose

# For all tests
for test in rust_stubgen_e2e_test/test_*/build/*.so; do
  echo "Generating stubs for $test..."
  uv run tvm-ffi-stubgen --target rust --dlls "$test" --verbose
done
```

## Test Modules Overview

| Module | Focus | Key Types | Mutability |
| -------- | ------- | ----------- | ----------- |
| **test_scalar_types** | Basic types | int, float, bool, string, None | Mutable |
| **test_container_types** | Containers | Array\<T\>, Optional\<T\> | Mutable |
| **test_object_hierarchy** | Inheritance | Shape, Circle, Rectangle | Mutable |
| **test_immutable_types** | Immutability | All read-only fields | Mixed/Immutable |

## What Gets Generated

For each test, the stubgen tool will generate:

1. **Struct bindings** — Rust `#[repr(C)]` structs mirroring C++ object layouts
2. **Impl blocks** — Methods and constructors
3. **Deref/DerefMut** — For field access
4. **ObjectCore trait** — Type information and headers
5. **Import sections** — Necessary use statements
6. **Helper functions** — Type resolution and method lookup

## Inspecting the Output

After running stubgen, check the stdout for:

```text
[Removed] stale object blocks ...
[Skipped] object ... : unsupported type ...
[Warning] ... mixed read-only/read-write fields
```

These messages indicate how the generator handled special cases.

## Next Steps

Once bindings are generated, the next phase would be:

1. Create Rust test projects for each module
2. Import the generated bindings
3. Call into the C++ objects from Rust
4. Verify:
   - Constructor calls work
   - Field access works
   - Method invocations work
   - Static methods work
   - Destructors are called on drop

## Build Status

All C++ libraries build successfully:

- ✅ test_scalar_types.so (585 KB)
- ✅ test_container_types.so (928 KB)
- ✅ test_object_hierarchy.so (860 KB)
- ✅ test_immutable_types.so (940 KB)

Total: 4 comprehensive test suites covering all major Rust stubgen features.
