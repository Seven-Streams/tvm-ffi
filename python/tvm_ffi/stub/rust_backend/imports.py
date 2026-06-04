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
"""Rust ``use`` modelling for the ``tvm-ffi-stubgen`` Rust backend.

Import representation is language-specific (see the Python counterpart in
:mod:`tvm_ffi.stub.python_backend.imports`). Rust uses ``use a::b::c [as d];``
with ``::`` path separators; the language-agnostic pipeline only handles the
opaque collector (:class:`RustImports`).
"""

from __future__ import annotations

import dataclasses

from . import consts as C


@dataclasses.dataclass(frozen=True, eq=True)
class RustUse:
    """A single Rust ``use`` item: ``use <path> [as <alias>];``.

    Construction normalizes its input into a ``::``-separated path:

    * a value already containing ``::`` (e.g. a :data:`~.consts.RUST_TY_MAP_DEFAULTS`
      entry like ``tvm_ffi::Array``) is kept as-is;
    * a dotted FFI name (e.g. ``ffi.String`` or ``my_pkg.Foo``) has its leading
      segment rewritten via :data:`~.consts.RUST_MOD_MAP` (``ffi -> tvm_ffi``)
      and its dots turned into ``::`` (``ffi.String -> tvm_ffi::String``);
    * a bare leaf with no separator (e.g. ``i64`` / ``Option``) stays bare and
      needs **no** ``use`` (see :meth:`as_use_line`).
    """

    path: str
    alias: str | None = None

    def __init__(self, name: str, alias: str | None = None) -> None:
        """Normalize ``name`` into a Rust ``use`` path and store it."""
        if "::" not in name and "." in name:
            head, _, tail = name.partition(".")
            head = C.RUST_MOD_MAP.get(head, head)
            name = f"{head}.{tail}"
        path = name.replace(".", "::")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "alias", alias)

    @property
    def leaf(self) -> str:
        """The final path segment, e.g. ``Array`` for ``tvm_ffi::Array``."""
        return self.path.rsplit("::", 1)[-1]

    @property
    def full_name(self) -> str:
        """The full ``::`` path, used to dedup against locally-defined types."""
        return self.path

    @property
    def name_in_scope(self) -> str:
        """The identifier this ``use`` brings into scope (alias if any, else leaf)."""
        return self.alias if self.alias else self.leaf

    def as_use_line(self) -> str:
        """Render the ``use`` statement, or ``""`` for a bare prelude/primitive type.

        Bare types (no ``::``) such as ``i64`` / ``bool`` / ``()`` / ``Option``
        are in the prelude or builtin and require no import.
        """
        if "::" not in self.path:
            return ""
        if self.alias:
            return f"use {self.path} as {self.alias};"
        return f"use {self.path};"


@dataclasses.dataclass
class RustImports:
    """Opaque import collector threaded through the Rust generation pipeline.

    The language-agnostic ``cli`` treats this as an opaque token: it asks the
    backend to create one, seed it from ``import-object`` directives, and later
    render it. Only the Rust backend reaches inside.
    """

    items: list[RustUse] = dataclasses.field(default_factory=list)
