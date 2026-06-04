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

This checklist tracks test cases for the Rust stubgen across the six modules
(`test_scalar_types`, `test_container_types`, `test_object_hierarchy`,
`test_immutable_types`, `test_any_types`, `test_ffi_types`).

**Status: all planned items are done.** Every remaining unchecked box is tagged
**Not planned** with a reason — they are blocked codegen limitations (A3, K2),
crate-level gaps (E5 Tuple), by-design no-ops (L1), or low-value vs scaffolding
cost (J3, L3, L5). Run the whole suite with `bash run_all.sh`.

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
- ✅ **Error propagation** — D1/D2/D3 (method/ctor/static throw → `Err`).
- ✅ **Static factory returning object** — B2 (`ShapeBatch::unit_shape`).
- ✅ **Nested containers + `Optional<String>`** — E1/E2 (`NestedHolder`).
- ✅ **3-level inheritance** — K1 (`Box3D → ColoredBox`), multi-level Deref.
- ✅ **Core FFI types** — F1 Tensor, F2 Shape, F3 DataType/Device (`test_ffi_types`).
- ✅ **Function callbacks** — G1/G2 (closure param + returned closure).
- ✅ **`Any`/`AnyView`** — H1/H2 (`test_any_types`).
- ✅ **Unsupported types skip gracefully** — E3/E4 (`Map`/`Variant` → `[Skipped]`).
- ✅ **Reserved-word names + boundary values + direct field writes** — L2 / L4 / I1 / I2.
- ✅ **Compile-time read-only guarantee** — I3 (`trybuild`).

---

## P0 — Core blind spots (most likely to expose ABI / refcount bugs)

### A. Registered object as a method / static-method PARAMETER

- [x] **A1** Instance method takes object param: `bool same_size_as(Shape other)` on `Shape`. → `object_as_instance_method_param`
- [x] **A2** Static factory takes object param: `static int64_t combined_area(Shape a, Shape b)`. → `object_as_static_method_params`
- [ ] **A3** — **Not planned** (blocked). Derived object passed to a base-typed param needs a stubgen upcast (`From<Circle> for Shape`); the generated `Circle` is a distinct Rust type. Requires a codegen change, out of scope for the test suite.
- [x] **A4** Container of objects: `static int64_t total_area(Array<Shape>)` on `ShapeBatch`. → `array_of_objects_as_param`

### B. Registered object as a RETURN value

- [x] **B1** Instance method returns object: `Shape::scaled(factor) -> Shape` (new object). → `object_returned_from_instance_method`
- [x] **B2** Static factory returns object: `ShapeBatch::unit_shape() -> Shape`. → `test_object_hierarchy::static_factory_returns_object`
- [x] **B3** Returns nullable object: `ShapeBatch::non_empty_or_none(Shape) -> Optional<Shape>`; both `Some`/`None`. → `nullable_object_return`
- [x] **B4** Returns container-of-objects: `ShapeBatch::split(Shape) -> Array<Shape>`. → `array_of_objects_returned`

### C. Registered object as a FIELD type (nested objects)

- [x] **C1** `Group { Shape primary; Array<Shape> members; }`; construct-with-object, read object field via Deref, write object field via DerefMut. → `nested_object_fields`

### D. Error propagation (`Result::Err` branch currently 0%)

- [x] **D1** `Shape::checked_div(int64_t)` `TVM_FFI_THROW(ValueError)` on divide-by-zero; Rust asserts `Err`, checks kind=`ValueError` + message. → `checked_div_ok_and_err`
- [x] **D2** Constructor throws on invalid args (`Validated::new(-1)`); asserts `Err` + kind. → `test_object_hierarchy::constructor_throws_returns_err`
- [x] **D3** Static method throws (`ShapeBatch::safe_divide(1, 0)`). → `test_object_hierarchy::static_method_throws_returns_err`

---

## P1 — Type-system coverage (README already marks these Pending)

### E. Container extensions

- [x] **E1** Nested containers: `Array<Array<i64>>`, `Array<Optional<i64>>`, `Optional<Array<String>>` (param/return/field) on `NestedHolder`. → `test_container_types::{nested_array_param_and_return, array_of_optionals_param, nested_container_fields, optional_array_field_some_and_none}`
- [x] **E2** `Optional<String>` as param + return, both Some/None. → `test_container_types::echo_optional_string`
- [x] **E3** `Map<K,V>`: verified graceful skip — `MapHolder` (Map field) → `[Skipped] ... unsupported type 'Map'`, empty block, crate still builds. → `test_container_types` (`MapHolder`, no Rust binding by design)
- [x] **E4** `Variant<...>`: verified graceful skip — `VariantHolder` (Variant method) → `[Skipped] ... unsupported type 'Union'`. → `test_container_types` (`VariantHolder`)
- [ ] **E5** `Tuple` / multiple return values — **Not planned**. The crate has no `Tuple` type and tuples aren't `AnyCompatible`; `ffi::Tuple` reflects as origin `Tuple` (not in `RUST_UNSUPPORTED_ORIGINS`), so it would render a bare undefined `Tuple` rather than skip. Fixing needs either crate tuple support or adding `Tuple`/`tuple` to the unsupported set (one-line, but a codegen change, not a test).

### F. Core FFI types (new module `test_ffi_types`, class `FfiTypesHolder`)

- [x] **F1** `Tensor` (DLPack) as param + return (echo round-trip; Rust builds via `Tensor::from_nd_alloc`). → `test_ffi_types::tensor_param_and_return` *(field position not exercised — a default-constructed `Tensor` field is awkward; param/return covers the codec.)*
- [x] **F2** `ffi::Shape` as param/return/field. → `test_ffi_types::shape_param_and_return`, `ffi_type_fields`
- [x] **F3** `Device` / `DataType` (`DLDevice`/`DLDataType`) as param/return/field. → `test_ffi_types::{datatype_param_and_return, device_param_and_return, ffi_type_fields}`

### G. Function / callback (`test_ffi_types`)

- [x] **G1** Receive `Function` param: `apply_fn(fn, x)` invokes the callback; Rust passes a closure via `Function::from_typed`. → `test_ffi_types::function_as_param`
- [x] **G2** Return `Function`: `make_adder(n)` returns a closure; Rust calls it via `call_packed`. → `test_ffi_types::function_as_return`

### H. `Any` / `AnyView`

- [x] **H1** `Any` as param (renders as `AnyView`); push int/float/bool/String + an object through `echo` and assert transparent round-trip. → `test_any_types::{echo_roundtrips_primitive_types, echo_roundtrips_an_object}`
- [x] **H2** `Any` as return value (`echo`, `get_any`) + `Any` field round-trip. → `test_any_types::{echo_roundtrips_primitive_types, any_field_roundtrip}`

> **Stubgen resolution:** `into_typed_fn!` can't carry `Any`/`AnyView` (AnyView isn't
> `AnyCompatible`; an `Any` return hits the reflexive `TryFrom<Any>` whose error is
> `Infallible`). Rather than change the crate, the Rust backend generates **one uniform
> calling convention for every method/constructor**: pack args into `&[AnyView]` and call
> `Function::call_packed` (`_packed_args_expr` / `_packed_call_lines` in
> `rust_backend/codegen.py`). Each non-`AnyView` arg becomes `AnyView::from(&x)`, an
> `AnyView` arg passes through, a member call prepends `AnyView::from(&*self)`; an `Any`
> return is forwarded as-is, everything else is `?.try_into()?`. No `into_typed_fn!` is
> emitted anywhere. *Note: nested `Any` (e.g. `Array<Any>`, `Optional<Any>`) is still
> unsupported — those container types aren't `AnyCompatible` over `Any` either.*

---

## P2 — Behavior / semantic edges

### I. Mutability & field write-back

- [x] **I1** Directly write a `String` field via DerefMut. → `test_scalar_types::write_string_field_directly`
- [x] **I2** Directly write a container field via DerefMut. → `test_container_types::write_container_field_directly`
- [x] **I3** Compile-time read-only guarantee via `trybuild`: `tests/ui/assign_readonly_field.rs` (field assign) and `tests/ui/mut_borrow_readonly.rs` (`&mut`) both fail to compile with "trait `DerefMut` ... not implemented for `ImmutableVersion`". → `test_immutable_types::readonly_types_reject_mutation` (regenerate `.stderr` with `TRYBUILD=overwrite` on a toolchain bump)

### J. Destructor / refcount (QUICKSTART claims to verify, but no assertion exists)

- [x] **J1** Drop triggers destructor: `Tracked` has a process-global live counter + `live_count()`; assert C++ dtor runs exactly once on last drop. → `drop_runs_destructor_exactly_once`
- [x] **J2** Clone / shared ownership: clone a `Shape`, mutate via one handle + observe via the other, drop one, confirm the other still usable. → `clone_shares_underlying_object`
- [ ] **J3** Object survives FFI round-trip without leak / premature free — **Not planned** (already covered in substance by A4/B4 object round-trips and the J1 refcount assertion; a dedicated leak test would need a memory profiler).

### K. Inheritance

- [x] **K1** 3+ level inheritance: `Object → Shape → Box3D → ColoredBox`; multi-level Deref reaches top-most `Shape` fields. → `test_object_hierarchy::{three_level_inheritance_field_access, mid_level_type_has_own_method_and_inherited_fields}`
- [ ] **K2** Calling inherited base methods on a derived type — **Not planned** (blocked codegen limitation, same class as A3): each generated ref type's `impl` only carries its OWN registered methods; `Deref` reaches the `*Obj` structs (fields), not the parent **ref** type's methods, so `colored_box.volume()` / `circle.get_area()` don't compile. Would need codegen to re-emit inherited methods on derived ref types (or `Deref` a derived ref to its parent ref).

---

## P3 — Misc / robustness

- [ ] **L1** Global free functions — **Not planned** (N/A by design): the Rust backend does not generate bindings for `register_global_func` globals (decision 5 — Rust calls them dynamically via `Function::get_global(name)`); a `global/<prefix>` block is left untouched, so there is no generated output to e2e-test.
- [x] **L2** Rust reserved-word field/method names: fields `type`/`match`/`move` + method registered as `fn` → raw-escaped `r#type`/`r#fn`/... → `test_scalar_types::reserved_word_names_are_raw_escaped`
- [ ] **L3** Multiple `init` overloads / constructor default arguments — **Not planned**: reflection exposes a single `__ffi_init__`, and C++ default args don't translate to Rust optionals (the generated `new` always takes all params positionally). No distinct behavior to assert beyond the existing ctor tests.
- [x] **L4** Boundary values: `i64::MAX`/`MIN`, negative `format_scalars`, empty + Unicode string. → `test_scalar_types::boundary_values`
- [ ] **L5** Namespace / module-name collision (two libraries registering the same short type name) — **Not planned**: would require a second shared library + crate; each module already namespaces under its own `type_key` prefix, so the realistic risk is low relative to the scaffolding cost.

---

## Remaining (all Not planned)

These are intentionally not implemented — see each item above for the reason:

- **A3** / **K2** — blocked codegen limitations (no derived→base upcast; inherited
  methods not re-emitted on derived ref types).
- **E5** — `Tuple` unsupported by the crate (would render a bare undefined `Tuple`).
- **L1** — Rust backend generates no free-function bindings by design.
- **J3** — leak/round-trip already covered in substance; a true leak test needs a profiler.
- **L3** / **L5** — limited / low value relative to scaffolding cost.

Picking any of these up means changing the Rust backend codegen or the crate, not
just adding a test.
