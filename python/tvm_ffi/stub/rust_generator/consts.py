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
    "str": "::tvm_ffi::String",
    "bytes": "::tvm_ffi::Bytes",
    "Any": "::tvm_ffi::Any",
    "Callable": "::tvm_ffi::Function",
    "Array": "::tvm_ffi::Array",  # the crate's own Array<T>, NOT Vec
    "Map": "::tvm_ffi::Map",  # the crate's own Map<K, V>, NOT HashMap
    # A generic/opaque object VALUE is the single-pointer `ObjectRef` handle
    # (AnyCompatible, niche-optimizable), NOT the 24-byte `Object` data struct
    # (which is only ever the embedded struct `base`, spelled literally by codegen).
    "Object": "::tvm_ffi::object::ObjectRef",
    "Tensor": "::tvm_ffi::Tensor",
    "Shape": "::tvm_ffi::Shape",
    "Device": "::tvm_ffi::DLDevice",
    "dtype": "::tvm_ffi::DLDataType",
    "DataType": "::tvm_ffi::DLDataType",
    # --- builtin object type keys (ffi.*) ---
    "ffi.String": "::tvm_ffi::String",
    "ffi.Bytes": "::tvm_ffi::Bytes",
    "ffi.Module": "::tvm_ffi::Module",
    "ffi.Error": "::tvm_ffi::Error",
    "ffi.Object": "::tvm_ffi::object::ObjectRef",
    "ffi.Tensor": "::tvm_ffi::Tensor",
    "ffi.Shape": "::tvm_ffi::Shape",
    "ffi.Function": "::tvm_ffi::Function",
}

#: Origins the crate has no FFI type for (do NOT map to ``HashMap``/``Vec``;
#: Rust tuples don't match ``ffi::Tuple``'s layout). ``render_rust_type``
#: raises so the object renderer can use its lossless dynamic fallback.
RUST_UNSUPPORTED_ORIGINS = frozenset({"Dict", "List", "Union", "tuple"})

#: Module-prefix rewrites for ``use`` paths: builtin ``ffi.*`` type keys live at
#: the crate root.
RUST_MOD_MAP = {
    "ffi": "tvm_ffi",
}
