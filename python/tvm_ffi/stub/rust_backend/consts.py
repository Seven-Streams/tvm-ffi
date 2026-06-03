# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Rust-specific constants for the ``tvm-ffi-stubgen`` Rust backend.

Every mapping here is grounded in what the ``rust/tvm-ffi`` crate **actually
supports**, not in assumptions:

* scalar/primitive mappings come from the ``AnyCompatible`` impls in
  ``rust/tvm-ffi/src/type_traits.rs`` (all integers map to FFI ``int``, all
  floats to FFI ``float``, ``()`` is ``None``);
* object/ref type names come from each type's ``#[type_key = "..."]`` attribute
  and the crate-root re-exports in ``rust/tvm-ffi/src/lib.rs``;
* the crate has **no** ``Map`` / ``Dict`` / ``List`` equivalent and uses its own
  ``Array<T>`` (not ``Vec``) — see :data:`RUST_UNSUPPORTED_ORIGINS`.
"""

from __future__ import annotations

#: Default FFI-origin -> Rust-type map used to seed a render.
#:
#: Mirrors the Python ``TY_MAP_DEFAULTS`` convention: the value is the **fully
#: qualified path** (``tvm_ffi::Array``), so the Rust ``TyRenderer`` can split on
#: ``::`` to recover the leaf name (``Array``) *and* derive the ``use`` import
#: (``use tvm_ffi::Array;``) -- exactly as the Python side derives ``from … import``
#: from a dotted value.
#:
#: Primitives and prelude types (``i64``/``f64``/``bool``/``()``/``Option``) carry
#: **no** ``::`` and therefore need no import. Generic types are seeded by their
#: base path only; ``render_type`` composes the ``<T>`` arguments.
RUST_TY_MAP_DEFAULTS = {
    # --- scalars / primitives (type_traits.rs); no import needed ---
    "int": "i64",  # all integer widths map to FFI `int`; default back to i64
    "float": "f64",  # f32/f64 map to FFI `float`; default back to f64
    "bool": "bool",
    "None": "()",  # the crate represents None/void as the unit type
    "str": "tvm_ffi::String",  # tvm_ffi::String (NOT std::string::String -- needs `use`)
    # --- core / containers ---
    "Optional": "Option",  # std prelude; Option<T>, no import
    "Any": "tvm_ffi::Any",  # also AnyView (non-owning); position-dependent, refined later
    "Callable": "tvm_ffi::Function",
    "Array": "tvm_ffi::Array",  # tvm_ffi::Array<T> -- NOT Vec
    "Object": "tvm_ffi::Object",
    "Tensor": "tvm_ffi::Tensor",
    "Shape": "tvm_ffi::Shape",
    "Device": "tvm_ffi::DLDevice",  # dlpack DLDevice (re-exported at crate root)
    "dtype": "tvm_ffi::DLDataType",  # dlpack DLDataType (+ DLDataTypeExt methods)
    "DataType": "tvm_ffi::DLDataType",
    # --- builtin object type keys (ffi.*) ---
    "ffi.String": "tvm_ffi::String",
    "ffi.Bytes": "tvm_ffi::Bytes",
    "ffi.Module": "tvm_ffi::Module",
    "ffi.Error": "tvm_ffi::Error",
    "ffi.Object": "tvm_ffi::Object",
    "ffi.Tensor": "tvm_ffi::Tensor",
    "ffi.Shape": "tvm_ffi::Shape",
    "ffi.Function": "tvm_ffi::Function",
}

#: FFI origins the Rust crate cannot represent. ``render_type`` raises a sentinel
#: ``UnsupportedTypeError`` on these (defined in the Rust codegen module); the
#: object-level generator then warns and skips that object. Do NOT map these to
#: ``HashMap`` / ``Vec`` -- the crate has no such FFI types.
RUST_UNSUPPORTED_ORIGINS = frozenset({"Map", "Dict", "List", "Union"})

#: Module-prefix rewrites applied when constructing a Rust ``use`` path. The
#: builtin ``ffi.*`` type keys live at the crate root ``tvm_ffi::`` (see the
#: ``pub use`` re-exports in ``rust/tvm-ffi/src/lib.rs``). Path-construction
#: details (``::`` joining, aliasing) are handled by the import collector.
RUST_MOD_MAP = {
    "ffi": "tvm_ffi",
}
