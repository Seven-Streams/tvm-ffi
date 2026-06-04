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

**Object-as-value, error, and lifetime coverage (P0):**

- `Shape`: `checked_div()` (error propagation — `TVM_FFI_THROW` → Rust `Err`),
  `same_size_as(Shape)` / `combined_area(Shape, Shape)` (object params),
  `scaled(factor) -> Shape` (non-constructor object return).
- `ShapeBatch`: `total_area(Array<Shape>)` (array-of-objects param),
  `non_empty_or_none(Shape) -> Optional<Shape>` (nullable object return),
  `split(Shape) -> Array<Shape>` (array-of-objects return).
- `Group { Shape primary; Array<Shape> members; }`: nested object fields —
  construct-with-object, read via `Deref`, write object field via `DerefMut`.
- `Tracked`: process-global live-instance counter + `live_count()` — asserts the
  C++ destructor runs exactly once on the last drop and that `clone()` shares one
  underlying object.

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

**Compile-time read-only guarantee (I3):** `tests/ui/*.rs` + `tests/compile_fail.rs`
use [`trybuild`](https://docs.rs/trybuild) to assert that assigning a `def_ro`
field or taking `&mut` through a read-only type **fails to compile** (no `DerefMut`
is generated). Regenerate the expected `.stderr` with `TRYBUILD=overwrite cargo test`
after a toolchain bump.

### 5. `test_any_types/` - `Any` / `AnyView` in param / return / field positions

**Coverage:**

- A top-level `Any` **argument** renders as the non-owning `AnyView`; an `Any`
  **return**/**field** stays the owning `Any` (per `docs/concepts/any.rst`).
- Because `into_typed_fn!` can't carry `Any`/`AnyView` (AnyView isn't
  `AnyCompatible`; an `Any` return hits the reflexive `TryFrom<Any>` whose error
  is `Infallible`), the Rust backend uses **one uniform calling convention for
  every method**: pack args into `&[AnyView]` and call
  `Function::call_packed(&[AnyView]) -> Result<Any>`. No `into_typed_fn!` is emitted.

**Key Class:**

- `AnyHolder`: holds an `Any` field (`stored`)
  - Static method: `echo(Any) -> Any` — returns its input verbatim; tests assert
    transparent round-trip for int/float/bool/String and for an object, so the
    payload comes back unchanged without C++ inspecting it.
  - Instance methods: `set_any(Any)` / `get_any() -> Any` (field write/read)

### 6. `test_ffi_types/` - Core FFI value types (Shape / DataType / Device / Function)

**Coverage:**

- `ffi::Shape` (F2), `DLDataType` (F3), `DLDevice` (F3) as param / return / field.
- `Function` as a parameter (G1 — Rust passes a closure that C++ invokes) and as a
  return value (G2 — C++ returns a closure that Rust calls).

**Key Class:**

- `FfiTypesHolder`: fields `shape` / `dtype` / `device`
  - Static methods: `shape_product`, `make_shape`, `echo_dtype`, `dtype_bits`,
    `echo_device`, `device_id`, `apply_fn(fn, x)`, `make_adder(n) -> Function`
  - Instance method: `shape_ndim()`

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
| Nested containers (Array<Array\<T\>>, Array<Optional\<T\>>, Optional<Array\<T\>>) | test_container_types | ✓ |
| Shape / DataType / Device | test_ffi_types | ✓ |
| Function (callback param + return) | test_ffi_types | ✓ |
| Tensor (DLPack), param + return | test_ffi_types | ✓ |
| Map / Dict / Variant (unsupported → graceful `[Skipped]`) | test_container_types | ✓ (skip verified) |
| Tuple | — | Pending* (no Rust tuple type / not AnyCompatible) |
| Any/AnyView (param→AnyView, return/field→Any) | test_any_types | ✓ |
| Object references | test_object_hierarchy | ✓ |
| Objects as params / returns / fields | test_object_hierarchy | ✓ |
| Error propagation (`Result::Err`) | test_object_hierarchy | ✓ |
| Destructor / refcount on drop | test_object_hierarchy | ✓ |
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
| DerefMut (immutable types) | test_immutable_types | ✓ (runtime + `trybuild` compile-fail) |
| Reserved-word names raw-escaped (`r#type`) | test_scalar_types | ✓ |
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
