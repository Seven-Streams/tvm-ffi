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
"""Rust code generation for the ``tvm-ffi-stubgen`` tool.

Currently implements the type renderer (:func:`render_rust_type`). It turns a
language-agnostic :class:`~tvm_ffi.core.TypeSchema` into a Rust type expression,
recording the ``use`` imports it needs via a ``ty_render`` callback. FFI types
the ``rust/tvm-ffi`` crate cannot represent raise :class:`UnsupportedTypeError`,
which the (future) object/function generators catch to warn-and-skip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .consts import RUST_UNSUPPORTED_ORIGINS
from .imports import RustUse

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema

    from .imports import RustImports


class UnsupportedTypeError(Exception):
    """Raised when an FFI type has no representation in the ``rust/tvm-ffi`` crate.

    Carries the offending FFI ``origin`` (e.g. ``"Map"``) so callers can produce
    a precise ``[Skipped]`` message before skipping the enclosing object/function.
    """

    def __init__(self, origin: str) -> None:
        """Record the unsupported ``origin`` and build a clear message."""
        super().__init__(f"Rust backend does not support FFI type {origin!r}")
        self.origin = origin


def build_ty_render(ty_map: dict[str, str], imports: RustImports) -> Callable[[str], str]:
    """Build a leaf-origin -> Rust-leaf-name mapper that records ``use`` imports.

    Given an FFI origin, it looks up the (fully-qualified) Rust path in ``ty_map``,
    records the ``use`` it needs, and returns the name to use in scope:

    * bare prelude/primitive types (``i64`` / ``bool`` / ``()`` / ``Option``) carry
      no ``::`` -> no ``use`` is recorded and the bare name is returned;
    * a qualified path is recorded once (repeat references to the *same* path reuse
      the binding rather than emitting a duplicate ``use``);
    * if a different path wants a leaf name already taken (e.g. ``a::Foo`` and
      ``b::Foo``), the later one is aliased -- ``use b::Foo as Foo2;`` -- and the
      alias is returned. This type-vs-type clash is the only ``use`` collision that
      arises in Rust (function/method names live in a separate namespace, and
      methods are scoped inside ``impl`` blocks, so they never shadow a ``use``).
    """
    # Seed from anything already recorded (e.g. import-object directives) so we
    # don't re-import or collide with pre-existing uses.
    binding_of: dict[str, str] = {u.full_name: u.name_in_scope for u in imports.items}
    used_names: set[str] = {u.name_in_scope for u in imports.items}

    def _run(origin: str) -> str:
        probe = RustUse(ty_map.get(origin, origin))
        if not probe.as_use_line():
            # bare prelude/primitive: no import, no collision tracking.
            return probe.leaf
        full = probe.full_name
        if full in binding_of:
            return binding_of[full]  # same path already imported (maybe aliased)
        leaf = probe.leaf
        if leaf not in used_names:
            name, use = leaf, probe
        else:
            n = 2
            while f"{leaf}{n}" in used_names:
                n += 1
            name = f"{leaf}{n}"
            use = RustUse(full, alias=name)
        imports.items.append(use)
        used_names.add(name)
        binding_of[full] = name
        return name

    return _run


def render_rust_type(schema: TypeSchema, ty_render: Callable[[str], str]) -> str:
    """Render a :class:`TypeSchema` into a Rust type expression.

    ``ty_render`` maps a leaf origin name to its Rust leaf name and records the
    ``use`` import it needs (see :func:`build_ty_render`).

    Raises
    ------
    UnsupportedTypeError
        If ``schema`` (or any nested arg) uses an FFI origin the crate cannot
        represent (``Union`` / ``Map`` / ``Dict`` / ``List``).

    """
    origin = schema.origin
    args = schema.args or ()

    if origin in RUST_UNSUPPORTED_ORIGINS:
        raise UnsupportedTypeError(origin)

    if origin == "Optional":
        # post_init guarantees exactly one arg.
        inner_schema = next(iter(args), None)
        if inner_schema is None:
            raise ValueError("Optional type requires exactly one argument")
        inner = render_rust_type(inner_schema, ty_render)
        return f"{ty_render('Optional')}<{inner}>"

    if origin == "Array":
        # post_init guarantees (Any,) when no element type is given.
        elem = render_rust_type(args[0], ty_render) if args else ty_render("Any")
        return f"{ty_render('Array')}<{elem}>"

    if origin == "Callable":
        # The crate's Function is type-erased / concrete: no generic params.
        return ty_render("Callable")

    if origin == "tuple":
        if not args:
            return "()"
        inner = ", ".join(render_rust_type(a, ty_render) for a in args)
        return f"({inner})"

    # leaf / object type -> resolve via ty_render (records its `use`).
    return ty_render(origin)
