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
"""Rust-specific constants for the ``tvm-ffi-stubgen`` Rust backend."""

from __future__ import annotations

#: Default FFI-origin -> Rust-type map. Values are fully qualified paths so
#: ``RustUse``/``RustImports`` can derive both the leaf name and the ``use``
#: import; values without ``::`` (primitives) need no import.
RUST_TY_MAP_DEFAULTS = {
    "int": "i64",
    "float": "f64",
    "bool": "bool",
    "None": "()",
    "str": "tvm_ffi::String",
    "bytes": "tvm_ffi::Bytes",
    "Any": "tvm_ffi::Any",
    "Callable": "tvm_ffi::Function",
    "Array": "tvm_ffi::Array",  # the crate's own Array<T>, NOT Vec
    "Map": "tvm_ffi::Map",  # the crate's read-only Map<K, V>, NOT HashMap
    "Object": "tvm_ffi::Object",
    "Tensor": "tvm_ffi::Tensor",
    "Shape": "tvm_ffi::Shape",
    "Device": "tvm_ffi::DLDevice",
    "dtype": "tvm_ffi::DLDataType",
    "DataType": "tvm_ffi::DLDataType",
    # --- builtin object type keys (ffi.*) ---
    "ffi.String": "tvm_ffi::String",
    "ffi.Bytes": "tvm_ffi::Bytes",
    "ffi.Map": "tvm_ffi::Map",
    "ffi.Module": "tvm_ffi::Module",
    "ffi.Error": "tvm_ffi::Error",
    "ffi.Object": "tvm_ffi::Object",
    "ffi.Tensor": "tvm_ffi::Tensor",
    "ffi.Shape": "tvm_ffi::Shape",
    "ffi.Function": "tvm_ffi::Function",
}

#: Width-correct scalar for a ``#[repr(C)]`` struct field, keyed by
#: ``(ffi origin, sizeof(T))``: the type schema erases scalar widths, but the
#: generated structs read fields at their real offsets, so the width must be
#: recovered from the reflected field size. Signedness is not recorded;
#: unsigned C++ fields render as the same-width signed type.
RUST_SCALAR_BY_SIZE = {
    ("int", 1): "i8",
    ("int", 2): "i16",
    ("int", 4): "i32",
    ("int", 8): "i64",
    ("float", 4): "f32",
    ("float", 8): "f64",
}

#: Origins the crate has no FFI type for: ``Dict``/``List``/``Union`` have no
#: Rust counterpart at all (do NOT map to ``HashMap``/``Vec``), and ``tuple``
#: only has a std rendering (Rust tuples) that is not layout-compatible with C++
#: ``ffi::Tuple``. ``render_rust_type`` raises ``UnsupportedTypeError`` wherever
#: one appears (field, argument, return, or nested) and the enclosing object is
#: skipped. (``Optional`` is supported: native ``Option<T>`` at boundaries,
#: layout-mirror as a direct struct field. ``Map`` is supported via the crate's
#: read-only ``Map<K, V>`` -- see ``render_rust_type``.)
RUST_UNSUPPORTED_ORIGINS = frozenset({"Dict", "List", "Union", "tuple"})

#: Alignment value -> zero-sized marker (re-exported at the crate root) used to
#: give ``tvm_ffi::Optional<T, A, N>`` its alignment (Rust has no const-generic
#: ``align``).
RUST_ALIGN_MARKER = {
    1: "tvm_ffi::Align1",
    2: "tvm_ffi::Align2",
    4: "tvm_ffi::Align4",
    8: "tvm_ffi::Align8",
    16: "tvm_ffi::Align16",
}

#: The crate's layout-mirror type for a C++ ``ffi::Optional<T>`` struct field.
RUST_OPTIONAL_TYPE = "tvm_ffi::Optional"

#: Origins whose Rust rendering does NOT implement ``AnyCompatible``: ``Any``
#: (``tvm_ffi::Any``) and the bare base object (``tvm_ffi::Object``). A type is
#: AnyCompatible iff none of these appear in it (every other renderable leaf --
#: scalars, ``String``/``Bytes``, ``Array<T>``, generated/builtin ObjectRefs --
#: is AnyCompatible). Used to decide whether ``Optional<T>`` can marshal through
#: a native ``Option<V>`` (accessor / argument / return) or must be skipped.
RUST_NOT_ANY_COMPATIBLE_ORIGINS = frozenset({"Any", "Object", "ffi.Object"})

#: Module-prefix rewrites for ``use`` paths: builtin ``ffi.*`` type keys live at
#: the crate root.
RUST_MOD_MAP = {
    "ffi": "tvm_ffi",
}

#: Rust keywords (strict + reserved, across the 2015/2018/2021 editions) that are
#: a parse error when used as a bare identifier. A C++ field/method name landing
#: on one is escaped as a raw identifier ``r#<name>`` (see ``RUST_RAW_IDENT_*``).
RUST_KEYWORDS = frozenset(
    {
        "as", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop",
        "match", "mod", "move", "mut", "pub", "ref", "return", "self", "Self",
        "static", "struct", "super", "trait", "true", "type", "unsafe", "use",
        "where", "while",
        # reserved for future use (also rejected as bare identifiers)
        "abstract", "async", "await", "become", "box", "do", "final", "gen",
        "macro", "override", "priv", "try", "typeof", "unsized", "virtual",
        "yield",
    }
)  # fmt: skip

#: The few keywords a raw identifier (``r#``) cannot spell; a C++ name landing on
#: one is suffix-renamed instead (``self -> self_``). Rare in practice.
RUST_RAW_IDENT_FORBIDDEN = frozenset({"crate", "self", "super", "Self"})
