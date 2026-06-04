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

# Rust Stubgen E2E Test — Coverage Gaps & TODO

This checklist tracks test cases not yet covered by the existing modules
(`test_scalar_types`, `test_container_types`, `test_object_hierarchy`,
`test_immutable_types`, `test_any_types`).

Each item notes **what to test + how to set up the C++ side + the Rust-side
assertion focus**.

## Current coverage summary

- ✅ Static methods (no-arg and with scalar/container args) — verified in all 4 modules.
- ✅ Instance methods (scalar + container params, scalar/String/Array/Optional/void returns).
- ✅ Constructors (`__ffi_init__`), field read via Deref, field write via DerefMut.
- ✅ 2-level inheritance (`Object → Shape → Circle/Rectangle`), inherited field access.
- ✅ Mutable vs immutable (`def_rw` / `def_ro`), mixed-mutability treated as read-only.
- ✅ **Registered objects as method/static-method PARAMETERS** — A1/A2/A4 in `test_object_hierarchy`.
- ✅ **Registered objects as RETURN values** (non-constructor) — B1/B3/B4 in `test_object_hierarchy`.
- ✅ **Registered objects as FIELD types (nested objects)** — C1 (`Group`) in `test_object_hierarchy`.
- ✅ **Error propagation (`Result::Err` branch)** — D1 (`checked_div`) in `test_object_hierarchy`.
- ✅ **Destructor / refcount** — J1/J2 (`Tracked`, clone-shares) in `test_object_hierarchy`.

---

## P0 — Core blind spots (most likely to expose ABI / refcount bugs)

### A. Registered object as a method / static-method PARAMETER

- [x] **A1** Instance method takes object param: `bool same_size_as(Shape other)` on `Shape`. → `object_as_instance_method_param`
- [x] **A2** Static factory takes object param: `static int64_t combined_area(Shape a, Shape b)`. → `object_as_static_method_params`
- [ ] **A3** Derived object passed to base-typed param: signature `Shape`, pass a `Circle` (upcast / polymorphism). *Blocked: generated `Circle` is a distinct Rust type with no upcast `From<Circle> for Shape`; needs a stubgen upcast story first.*
- [x] **A4** Container of objects: `static int64_t total_area(Array<Shape>)` on `ShapeBatch`. → `array_of_objects_as_param`

### B. Registered object as a RETURN value

- [x] **B1** Instance method returns object: `Shape::scaled(factor) -> Shape` (new object). → `object_returned_from_instance_method`
- [ ] **B2** Static factory returns object: `static ScalarHolder make_default()`; non-constructor object production, Rust takes over refcount. *(B1 already covers non-constructor object production via an instance method; a pure-static variant is still nice-to-have.)*
- [x] **B3** Returns nullable object: `ShapeBatch::non_empty_or_none(Shape) -> Optional<Shape>`; both `Some`/`None`. → `nullable_object_return`
- [x] **B4** Returns container-of-objects: `ShapeBatch::split(Shape) -> Array<Shape>`. → `array_of_objects_returned`

### C. Registered object as a FIELD type (nested objects)

- [x] **C1** `Group { Shape primary; Array<Shape> members; }`; construct-with-object, read object field via Deref, write object field via DerefMut. → `nested_object_fields`

### D. Error propagation (`Result::Err` branch currently 0%)

- [x] **D1** `Shape::checked_div(int64_t)` `TVM_FFI_THROW(ValueError)` on divide-by-zero; Rust asserts `Err`, checks kind=`ValueError` + message. → `checked_div_ok_and_err`
- [ ] **D2** Constructor throws on invalid args; assert `new(...)` returns `Err`.
- [ ] **D3** Static method throws.

---

## P1 — Type-system coverage (README already marks these Pending)

### E. Container extensions

- [ ] **E1** Nested containers: `Array<Array<i64>>`, `Array<Optional<i64>>`, `Optional<Array<String>>`.
- [ ] **E2** `Optional<String>` as return/param (currently only `Optional<i64>` returns; `Optional<String>` only as a field).
- [ ] **E3** `Map<K,V>` / `Dict`: at minimum verify stubgen skips unsupported types and prints `[Skipped]`; if supported, add positive test.
- [ ] **E4** `Variant<...>`.
- [ ] **E5** `Tuple` / multiple return values.

### F. Core FFI types

- [ ] **F1** `Tensor` (DLPack) as field/param/return.
- [ ] **F2** `ffi::Shape`.
- [ ] **F3** `Device` / `DataType`.

### G. Function / callback

- [ ] **G1** Receive `Function` param: C++ method accepts a callback and invokes it; Rust passes a closure.
- [ ] **G2** Return `Function`.

### H. `Any` / `AnyView`

- [x] **H1** `Any` as param (renders as `AnyView`); push int/float/bool/String through `describe_any`. → `test_any_types::describe_any_dispatches_on_runtime_type`
- [x] **H2** `Any` as return value (`echo`, `get_any`) + `Any` field round-trip. → `test_any_types::{echo_returns_owning_any, any_field_roundtrip}`

> **Stubgen resolution:** `into_typed_fn!` can't carry `Any`/`AnyView` (AnyView isn't
> `AnyCompatible`; an `Any` return hits the reflexive `TryFrom<Any>` whose error is
> `Infallible`). Rather than change the crate, the Rust backend now drops methods whose
> signature involves a top-level `Any`/`AnyView` to the raw `Function::call_packed(&[AnyView])`
> path (`_needs_packed_call` / `_packed_args_expr` in `rust_backend/codegen.py`). All other
> methods are unchanged. *Note: nested `Any` (e.g. `Array<Any>`, `Optional<Any>`) is still
> unsupported — those container types aren't `AnyCompatible` over `Any` either.*

---

## P2 — Behavior / semantic edges

### I. Mutability & field write-back

- [ ] **I1** Directly write a `String` field: scalar test's `mutate_fields_from_rust` skips `string_val`; add `holder.string_val = FFIString::from(...)`.
- [ ] **I2** Directly write a container field: `holder.int_array = Array::new(...)` (currently only via C++ `set_arrays`).
- [ ] **I3** Real negative compile test for read-only: use `trybuild` / compile-fail to assert "assigning a `def_ro` field fails to compile" and "taking `&mut` on a read-only type fails to compile". The current "negative test" is only runtime read-only handling, not a compile-time guarantee.

### J. Destructor / refcount (QUICKSTART claims to verify, but no assertion exists)

- [x] **J1** Drop triggers destructor: `Tracked` has a process-global live counter + `live_count()`; assert C++ dtor runs exactly once on last drop. → `drop_runs_destructor_exactly_once`
- [x] **J2** Clone / shared ownership: clone a `Shape`, mutate via one handle + observe via the other, drop one, confirm the other still usable. → `clone_shares_underlying_object`
- [ ] **J3** Object survives FFI round-trip without leak / premature free (pair with A, B). *(partially exercised by A4/B4 round-trips.)*

### K. Inheritance

- [ ] **K1** 3+ level inheritance: `Object → A → B → C`; verify multi-level Deref reaches top-most base fields.
- [ ] **K2** Calling inherited base methods / base static methods on a derived type (currently `Circle` never tests calling `get_area()` / `get_default_width()`).

---

## P3 — Misc / robustness

- [ ] **L1** Global free functions: `register_global_func` not attached to any class; verify generated free function is callable (all current cases are class methods; zero free-function samples).
- [ ] **L2** Rust reserved-word field/method names: fields named `type` / `match` / `fn` / `move`; verify generated code escapes or renames.
- [ ] **L3** Multiple `init` overloads / constructor default arguments.
- [ ] **L4** Boundary values: `i64` extremes & negatives, negative `format_scalars`, empty string, Unicode string, very long string.
- [ ] **L5** Namespace / module-name collision: two libraries registering the same short type name.

---

## Suggested implementation order

**D (error propagation) → A/B (object as param & return) → C (nested object) → J (dtor/refcount).**
These four are the ones that actually surface ABI and lifetime bugs; the rest are
nice-to-have and can be added on demand.
