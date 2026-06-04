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

# `tvm-ffi-stubgen` — Rust Backend User Manual

This manual documents the **Rust backend** of `tvm-ffi-stubgen`
(`--target rust`). It generates Rust binding source — `struct` / `impl` / `fn`
/ `use` — for object types registered in the TVM FFI reflection registry, so
that Rust code can construct and call into the same C++ heap objects through the
stable FFI ABI.

The backend lives in `python/tvm_ffi/stub/rust_backend/` and plugs into the same
language-agnostic pipeline (`cli.py`, `file_utils.py`, `lib_state.py`) that
drives the Python backend.

> **Scope note (read this first).** The Rust backend generates bindings for
> **object types only**. It does **not** generate anything for **global
> functions** — see [§7 Global functions are not generated](#7-global-functions-are-not-generated).

---

## 1. Quick start

The generator works in two modes, exactly like the Python backend.

### 1.1 In-place mode (fill existing marker blocks)

Add directive markers to a `.rs` file and let the tool fill them:

```rust
// tvm-ffi-stubgen(begin): helpers
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): import-section
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/my_pkg.Expr
// tvm-ffi-stubgen(end)
```

Then run:

```bash
uv run tvm-ffi-stubgen --target rust \
  --dlls /path/to/libmy_ext.so \
  path/to/bindings.rs
```

The tool loads the shared library (so the reflection registry is populated),
reads each `object/<type_key>` block, and rewrites the content **between** the
`begin` / `end` markers. Everything outside the markers is left untouched.

### 1.2 Init / scaffolding mode (generate a module tree)

Generate the file layout and all object blocks from scratch under a directory:

```bash
uv run tvm-ffi-stubgen --target rust \
  --dlls /path/to/libmy_ext.so \
  --init-pypkg my-ext \
  --init-lib my_ext_shared \
  --init-prefix my_pkg. \
  src/generated
```

This creates one `mod.rs` per registered prefix under `src/generated/`,
scaffolds the `helpers` + `import-section` + per-type `object/<key>` markers,
fills them, and stitches the module tree together with `pub mod <child>;`
declarations (see [§6 File layout](#6-file-layout-and-module-tree)).

### 1.3 Relevant CLI flags

| Flag | Meaning |
| --- | --- |
| `--target rust` | Select the Rust backend (default is `python`). |
| `--dlls "a.so;b.so"` | Shared libraries to preload so the reflection registry is populated. `;`-separated. |
| `--imports "pkgA;pkgB"` | Python modules to import first (to trigger registration). `;`-separated. |
| `--init-pypkg` / `--init-lib` / `--init-prefix` | Enable init mode. All three are **required together**. `--init-prefix` filters which type-key prefix is generated (e.g. `my_pkg.`). |
| `--dry-run` | Compute changes but do not write files. |
| `--verbose` | Print a unified diff per file. |
| `--indent N` | Extra spaces added inside each generated block (default 4). |

---

## 2. What the Rust backend generates per object

For each `object/<type_key>` block (where `<type_key>` is a registered type such
as `my_pkg.Add`), the backend emits a self-contained Rust binding consisting of:

1. **A `#[repr(C)]` data struct `<Leaf>Obj`** mirroring the C++ object layout.
   The parent is embedded as the **first field, always named `base`**:
   - for a root type (parent is `ffi.Object` or none): `base: Object`;
   - otherwise: `base: <Parent>Obj`.

   Each reflected field follows, as `pub <name>: <RustType>`.

2. **`unsafe impl ObjectCore for <Leaf>Obj`** — wires up the `TYPE_KEY`,
   `type_index()` (via the shared `lookup_type_index` helper), and
   `object_header_mut`.

3. **A `#[repr(C)] #[derive(ObjectRef, Clone)]` reference struct `<Leaf>`**
   holding `data: ObjectArc<<Leaf>Obj>`.

4. **`Deref` (and `DerefMut` for mutable classes)** for the ref → its `Obj`, and
   for a derived `Obj` → its embedded `base`.

5. **`impl <Leaf> { ... }`** containing:
   - `pub fn new(...) -> Result<Self>` — only when the type has a constructor
     (`__ffi_init__`); it calls the reflected constructor.
   - one `pub fn` per reflected method (instance or static).

The struct and ref types and their methods are `pub`. The internal `base` and
`data` fields stay private (reached through the generated `Deref` impls).

### 2.1 Worked example

Given the C++ types `cpp_rust_test.Expr` (root, mutable, one `i64` field, one
static method `test`, a constructor) and `cpp_rust_test.Add` (extends `Expr`,
two `Expr` fields, an instance method `update`), the backend produces bindings
equivalent to:

```rust
#[repr(C)]
pub struct ExprObj {
    base: Object,
    pub value: i64,
}

unsafe impl ObjectCore for ExprObj {
    const TYPE_KEY: &'static str = "cpp_rust_test.Expr";
    fn type_index() -> i32 { lookup_type_index(Self::TYPE_KEY) }
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.base)
    }
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
pub struct Expr {
    data: ObjectArc<ExprObj>,
}

impl Deref for Expr { /* ... */ }
impl DerefMut for Expr { /* ... */ }   // mutable class -> DerefMut too

impl Expr {
    pub fn new(_0: i64) -> Result<Self> {
        let ctor = get_type_method(ExprObj::TYPE_KEY, "__ffi_init__")?;
        let call = into_typed_fn!(ctor, Fn(i64) -> Result<Expr>);
        call(_0)
    }

    pub fn test() -> Result<i64> {                  // static method -> no self
        let f = get_type_method(ExprObj::TYPE_KEY, "test")?;
        let call = into_typed_fn!(f, Fn() -> Result<i64>);
        call()
    }
}
```

(The hand-written reference these mirror lives in
`cpp_rust_test1/rust/src/main.rs`.)

---

## 3. Supported types

The Rust backend renders each FFI `TypeSchema` into a Rust type expression. The
mapping is grounded in what the `rust/tvm-ffi` crate **actually provides** — the
`AnyCompatible` impls in `rust/tvm-ffi/src/type_traits.rs` and the crate-root
re-exports in `rust/tvm-ffi/src/lib.rs`.

### 3.1 Scalars and primitives

| FFI origin | Rust type | Import needed? |
| --- | --- | --- |
| `int` | `i64` | no (all integer widths collapse to FFI `int`; default `i64`) |
| `float` | `f64` | no (f32/f64 collapse to FFI `float`; default `f64`) |
| `bool` | `bool` | no |
| `None` | `()` | no (the crate represents void/None as the unit type) |
| `str` | `tvm_ffi::String` | `use tvm_ffi::String;` |

### 3.2 Core, container, and object types

| FFI origin / type key | Rust type | Notes |
| --- | --- | --- |
| `Optional` | `Option<T>` | std prelude; no import. Inner type recursively rendered. |
| `Any` | `tvm_ffi::Any` (or `AnyView`) | position-dependent — see [§5](#5-any-vs-anyview-the-position-rule). |
| `Callable` | `tvm_ffi::Function` | type-erased; **no** generic parameters are emitted even if the FFI schema carries them. |
| `Array` | `tvm_ffi::Array<T>` | the crate's own `Array` — **not** `Vec`. Element type recursively rendered; defaults to `Array<Any>` when no element type is given. |
| `tuple` | `()` or `(T1, T2, ...)` | empty tuple → `()`; otherwise a Rust tuple of rendered element types. |
| `Object` / `ffi.Object` | `tvm_ffi::Object` | |
| `Tensor` / `ffi.Tensor` | `tvm_ffi::Tensor` | |
| `Shape` / `ffi.Shape` | `tvm_ffi::Shape` | |
| `Device` | `tvm_ffi::DLDevice` | dlpack `DLDevice`, re-exported at crate root. |
| `dtype` / `DataType` | `tvm_ffi::DLDataType` | dlpack `DLDataType`. |
| `ffi.String` | `tvm_ffi::String` | |
| `ffi.Bytes` | `tvm_ffi::Bytes` | |
| `ffi.Module` | `tvm_ffi::Module` | |
| `ffi.Error` | `tvm_ffi::Error` | |
| `ffi.Function` | `tvm_ffi::Function` | |
| **User object types** (e.g. `my_pkg.Add`) | the ref type, leaf of the type key (e.g. `Add`) | resolved via the import collector; gets a `use` unless defined in the same file. |

The default mapping table is `RUST_TY_MAP_DEFAULTS` in
`rust_backend/consts.py`. You can override or extend it per file with a
`ty-map` directive (see [§8](#8-directives-reference)).

### 3.3 Nested and composed types

Type rendering is recursive, so the supported types compose freely:

- `Optional[Array[Expr]]` → `Option<tvm_ffi::Array<Expr>>`
- `tuple[int, Optional[str]]` → `(i64, Option<tvm_ffi::String>)`
- `Array[Any]` → `tvm_ffi::Array<tvm_ffi::Any>`

---

## 4. Unsupported types (warn-and-skip)

The `rust/tvm-ffi` crate has **no equivalent** for the following FFI origins:

| FFI origin | Why unsupported |
| --- | --- |
| `Map` | crate has no `Map` type |
| `Dict` | crate has no `Dict` type |
| `List` | crate has no `List` type (it has the immutable `Array<T>`, not a mutable list) |
| `Union` | crate has no tagged-union FFI type |

These are listed in `RUST_UNSUPPORTED_ORIGINS` (`rust_backend/consts.py`).

**Behavior:** when a field, method parameter, or return type uses one of these
origins (at any nesting depth), the type renderer raises an internal
`UnsupportedTypeError`. The object-level generator **catches it and skips the
entire object**, printing a yellow warning:

```text
[Skipped] object my_pkg.SomeType: unsupported type 'Map'
```

The skip is **per-object**: the run continues, other objects in the same file
are still generated, and the file is **not** aborted. The skipped block is left
with only its `begin` / `end` markers (no body), and no `use` imports leak from
the abandoned attempt.

> Do **not** try to "fix" this by mapping `Map` → `HashMap` or `List` → `Vec` in
> a `ty-map` directive: those Rust types are not FFI-ABI-compatible value
> containers and the generated bindings would not link correctly.

---

## 5. `Any` vs `AnyView` (the position rule)

`Any` is rendered position-dependently (decision backed by
`docs/concepts/any.rst`):

- **Field type** → `tvm_ffi::Any` (owning).
- **Method/constructor parameter, top-level** → `tvm_ffi::AnyView` (non-owning
  view). A borrow is cheap and correct for an argument.
- **Method return type, top-level** → `tvm_ffi::Any` (owning). A value coming
  back out of an FFI call has no borrow source to tie a view's lifetime to.
- **Nested `Any`** (e.g. inside `Array<Any>` or `Option<Any>`) stays
  `tvm_ffi::Any` in every position.

---

## 6. File layout and module tree

The Rust backend uses a **single `mod.rs` per module prefix** (the chosen
layout). Consequences:

- Both the "API file" and the "init file" are `mod.rs` (there is no separate
  entry file as in Python's `_ffi_api.py` / `__init__.py`).
- In init mode, each generated `mod.rs` is scaffolded with:
  - a `#![allow(dead_code, unused_imports)]` header + a `//!` doc line (fresh
    files only);
  - a `helpers` marker block;
  - an `import-section` marker block;
  - one `object/<type_key>` marker block per registered type under that prefix.
- **Module-tree stitching is automatic.** After all files are written,
  `finalize_init` writes idempotent `pub mod <child>;` declarations into each
  ancestor `mod.rs`, creating intermediate (type-less) directories and a root
  `mod.rs` as needed.
- **One manual step remains:** you must add a single `mod <mount>;` (or
  `pub mod <mount>;`) line at your crate root (`lib.rs` / `main.rs`) to mount
  the generated tree. The tool cannot safely edit your crate root.

### 6.1 The `helpers` block

Every generated file carries a `helpers` block with two shared support
functions (written fully-qualified, with **zero `use`s**, so they never clash
with the `import-section`):

- `lookup_type_index(type_key)` — resolves and caches a type's runtime index via
  `TVMFFITypeKeyToIndex`.
- `get_type_method(type_key, method_name)` — pulls a reflected `Function` off the
  type's method table (`TVMFFIGetTypeInfo(idx).methods`), used by every generated
  `new` / method.

The `helpers` block is refreshed on every run, even into a pre-existing file.

### 6.2 The `import-section` block

The `import-section` block collects every `use` the generated objects need
(boilerplate like `std::ops::Deref`, `tvm_ffi::object::ObjectArc`, plus all the
type imports). Imports targeting a type **defined in the same file** are dropped.
Leaf-name clashes between two different paths are auto-aliased
(`use b::Foo as Foo2;`).

### 6.3 Stale block pruning

Because a Rust object block is fully self-contained, on regeneration an
`object/<key>` block whose type is **no longer registered** is deleted wholesale
(printing `[Removed] stale object block <key>`), guarded so an unloaded library
can never wipe every block.

---

## 7. Global functions are NOT generated

**The Rust backend never generates bindings for global functions.** This is a
deliberate design decision, not a limitation pending work.

- `RustBackend.generate_global_funcs_block` is a **no-op**: a `global/<prefix>`
  marker block is left untouched.
- Init-mode scaffolding emits **no** `global/` markers.
- The `__all__` / `export` blocks are likewise no-ops (no global-function export
  surface exists).

**Why:** the Rust runtime already calls C++ global functions **dynamically** at
runtime via `Function::get_global("<name>")` (returning a `tvm_ffi::Function`
you invoke with `into_typed_fn!` or `call_packed`). A static stub would add no
value, so none is emitted. If you need a global, call it directly:

```rust
let f = tvm_ffi::Function::get_global("my_pkg.do_thing")?;
let call = into_typed_fn!(f, Fn(i64) -> Result<i64>);
let out = call(21)?;
```

(See `print_cpp_rust_test_global_registry` in `cpp_rust_test1/rust/src/main.rs`
for a fuller example of enumerating/calling globals at runtime.)

In short: **objects → generated; global functions → call at runtime, never
generated.**

---

## 8. Directives reference

Directives are single-line `//` comments embedded in the `.rs` file. The grammar
is identical to the Python backend's, only the comment token differs (`//` vs
`#`).

| Directive | Form | Effect (Rust) |
| --- | --- | --- |
| Block begin | `// tvm-ffi-stubgen(begin): <kind>/<param>` | Opens a generated block. |
| Block end | `// tvm-ffi-stubgen(end)` | Closes a generated block. |
| `object` | `begin): object/<type_key>` | Generates the struct/impl binding for `<type_key>`. |
| `helpers` | `begin): helpers` | Fills the shared `lookup_type_index` / `get_type_method` helpers. |
| `import-section` | `begin): import-section` | Renders all collected `use` statements. |
| `import-object` | `// tvm-ffi-stubgen(import-object): <path> <flag> <alias>` | Seeds a `use <path> [as <alias>];` into the import collector. The "type-checking-only" flag is ignored (Rust has no `TYPE_CHECKING` split). |
| `ty-map` | `// tvm-ffi-stubgen(ty-map): A.B -> tvm_ffi::C` | Overrides the FFI-origin → Rust-type mapping for this run. |
| `global` | `begin): global/<prefix>` | **No-op** for Rust (see §7). |
| `export` / `__all__` | — | **No-op** for Rust. |
| `skip-file` | `// tvm-ffi-stubgen(skip-file)` | Skips the whole file. |

### Identifier safety

FFI field/method names that collide with Rust keywords are emitted as raw
identifiers (`r#match`). The four keywords that cannot be raw identifiers
(`self`, `Self`, `super`, `crate`) are left as-is (they are extremely unlikely
as FFI names).

---

## 9. Mutability rule

Whether a class is treated as mutable is derived from its fields' read-only flags
(`frozen`, from `def_ro` vs `def_rw` on the C++ side):

| Field flags | Treated as | Emits |
| --- | --- | --- |
| **All** fields writable | mutable | `Deref` **and** `DerefMut`; method receivers are `&mut self` |
| **All** fields read-only | immutable | `Deref` only; method receivers are `&self` |
| **Mixed** | immutable (with a warning) | `Deref` only; prints `[Warning] ... mixed read-only/read-write fields; treating the whole type as read-only` |

> Note: even for a `&mut self` method, the `into_typed_fn!` self type is always a
> **shared** borrow (`&T`) — the FFI call borrows `self` as a view, and a
> `&mut self` receiver reborrows to `&self` at the call site.

---

## 10. Constructors and methods

- **`new`** is generated only when the type has an `__ffi_init__` (i.e.
  `has_init`). Its parameters are derived from the **`__ffi_init__` method
  schema** (`args[1:]`, the authoritative order), with positional names
  `_0, _1, …` (the schema carries no parameter names). It constructs by calling
  the reflected `__ffi_init__` — there is no global factory and no naming
  convention. If a type has a column-only auto-init (no `__ffi_init__` method),
  the params fall back to the reflected init fields (named), which may not match
  the true constructor order.
- **Instance methods** (`is_member = True`) take a `self` receiver and drop the
  leading `self` schema arg.
- **Static methods** (`is_member = False`) become associated functions with no
  `self`.
- The `__ffi_init__` method is **never** emitted as a regular method (it is
  consumed by `new`).

---

## 11. Limitations of the current layout

The single-`mod.rs`-per-prefix layout works cleanly for a single-prefix tree
(like `cpp_rust_test`). Known edges for multi-prefix trees:

- **Cross-prefix inheritance:** a `base: <Parent>Obj` whose parent is defined in
  a *different* file gets no `use` (only same-file inheritance is wired). Such
  references assume the generated tree is mounted at the crate root.
- **Dropping object blocks into a pre-existing non-empty file without a
  `helpers` block** won't compile — helpers are only scaffolded into a fresh file
  (the in-place `helpers` directive fills them if you add the marker yourself).
- **Helper duplication:** the helper functions are duplicated once per prefix
  file (cosmetic).

---

## 12. End-to-end verification

The generated bindings have been verified end-to-end against the real C++
library in `cpp_rust_test1/`: a static method call (`Expr::test() == 42`),
construction via the reflected `__ffi_init__` (`Expr::new` / `Add::new`), an
instance method that mutates C++ state (`Add::update()`), shared field access
across the FFI boundary, and correct C++ destructor firing on drop.
