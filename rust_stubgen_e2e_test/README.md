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

# Rust Stubgen E2E Test Suite

This directory contains comprehensive test cases for the Rust backend of `tvm-ffi-stubgen`, covering all supported features and data types.

## Overview

The test suite consists of 4 independent test modules, each focusing on specific aspects of the Rust code generation:

### 1. `test_scalar_types/` - Basic Scalar Types and String

**Coverage:**

- Scalar types: `int` (i64), `float` (f64), `bool`, `None`, `str` (String)
- Type mapping for all primitive FFI types
- Static methods returning various scalar types
- Instance methods with scalar parameters and return values
- Mutable class (can be modified after construction)

**Key Classes:**

- `ScalarHolder`: Holds all scalar types as fields
  - Static methods: `get_int_constant()`, `get_float_constant()`, `get_bool_constant()`, `get_string_constant()`, `format_scalars()`
  - Instance methods: `set_values()`, `get_description()`
  - All fields are read-write

### 2. `test_container_types/` - Container Types

**Coverage:**

- Container types: `Array<T>`, `Optional<T>`, nested containers
- Type recursion and composition
- Operations on arrays and optionals
- Mixed container scenarios

**Key Classes:**

- `ArrayHolder`: Works with arrays of different element types
  - Holds: `Array<int64_t>`, `Array<double>`, `Array<String>`
  - Static methods: `sum_array()`, `avg_array()`
  - Instance methods: `get_int_array_length()`, `join_string_array()`, `set_arrays()`

- `OptionalHolder`: Demonstrates optional types
  - Holds: `Optional<int64_t>`, `Optional<String>`
  - Static methods: `create_opt_int()`, `create_none_int()`
  - Instance methods: `describe_optionals()`

### 3. `test_object_hierarchy/` - Object Inheritance and Methods

**Coverage:**

- Object inheritance hierarchy
- Constructor generation (`__ffi_init__`)
- Instance methods on both base and derived classes
- Static methods
- Field access in inherited structures
- Multiple inheritance scenarios (through different derived types)

**Key Classes:**

- `Shape` (root object): Base class with width/height
  - Static methods: `get_default_width()`, `get_default_height()`
  - Instance methods: `get_area()`, `get_perimeter()`, `resize()`, `get_description()`
  - All fields are read-write

- `Circle` (derived from Shape): Adds radius field
  - Introduces new fields while inheriting from Shape
  - Instance methods: `get_circle_area()`, `set_radius()`

- `Rectangle` (derived from Shape): Adds is_square flag
  - Mixed boolean and numeric fields
  - Instance method: `update_square_flag()`

### 4. `test_immutable_types/` - Read-only Fields and Immutable Types

**Coverage:**

- Immutable classes (all fields read-only via `def_ro`)
- Mixed mutability (some fields read-only, some read-write)
- Behavior difference in code generation (`Deref` only vs `Deref` + `DerefMut`)
- Static methods on immutable types
- Read-only collections

**Key Classes:**

- `ImmutableVersion`: Fully immutable
  - Fields: major, minor, patch, label (all read-only)
  - Static methods: `get_current_major()`, `get_current_minor()`, `get_current_patch()`
  - Instance methods: `get_version_string()`, `is_greater_than()`

- `ImmutableMetadata`: Immutable with array field
  - Fields: name, author, license, keywords (all read-only)
  - Static method: `get_default_license()`
  - Instance methods: `to_json()`, `get_keyword_count()`

- `MixedMutability`: Demonstrates mixed field mutability
  - Will generate a warning about mixed mutability
  - Illustrates how the generator handles edge cases

## Building Individual Tests

Each test module has its own CMakeLists.txt and build script:

```bash
# Build test_scalar_types
cd rust_stubgen_e2e_test/test_scalar_types
bash scripts/build_and_run.sh

# Or build directly
cmake -S . -B build
cmake --build build --parallel
```

## Generating Rust Bindings

After building a test, generate Rust bindings using the stubgen tool:

```bash
cd /path/to/repo
source .venv/bin/activate
uv run tvm-ffi-stubgen --target rust \
  --dlls rust_stubgen_e2e_test/test_scalar_types/build/test_scalar_types.so \
  --verbose
```

## Features Tested

### Type System Coverage

| Feature | Test Module | Status |
| --------- | ------------ | -------- |
| Scalar types (int, float, bool, string, None) | test_scalar_types | ✓ |
| Array\<T\> | test_container_types | ✓ |
| Optional\<T\> | test_container_types | ✓ |
| Tuples | — | Pending* |
| Any/AnyView | — | Pending* |
| Object references | test_object_hierarchy | ✓ |
| Immutable types | test_immutable_types | ✓ |
| Mixed mutability | test_immutable_types | ✓ |

*To be added in future iterations

### Method Patterns

| Pattern | Test Module | Status |
| --------- | ------------ | -------- |
| Constructor (`__ffi_init__`) | test_object_hierarchy | ✓ |
| Static methods | All | ✓ |
| Instance methods | All | ✓ |
| Const methods | test_immutable_types | ✓ |

### Code Generation Features

| Feature | Test Module | Status |
| --------- | ------------ | -------- |
| Struct and impl generation | All | ✓ |
| Deref implementation | All | ✓ |
| DerefMut (mutable types) | test_scalar_types, test_container_types, test_object_hierarchy | ✓ |
| DerefMut (immutable types) | test_immutable_types | ✓ (negative test) |
| ObjectCore trait | All | ✓ |
| Import section generation | All | ✓ |
| Helpers generation | All | ✓ |
| Type conversions | All | ✓ |

## Future Extensions

Planned additions to increase coverage:

- FFI core types: Tensor, Shape, Device, DataType (for test_container_types extension)
- Function type (Callable)
- Tuple types
- Any/AnyView position rules
- Unsupported types (Map, Dict, List) - to test skip behavior
- Multiple inheritance patterns
- Generic/template scenarios

## Notes

- All test libraries are built as shared objects (.so/.dylib/.dll)
- The reflection registry is populated via TVM_FFI_STATIC_INIT_BLOCK()
- Each test is self-contained and can be built independently
- No Rust code is needed for this phase; Rust tests will be added in the next iteration
