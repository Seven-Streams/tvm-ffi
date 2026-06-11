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
"""Rust generator helpers for ``tvm-ffi-stubgen`` generation.

This module groups two Rust-specific concerns:

- import/use modelling (:class:`RustUse`, :class:`RustImports`)
- stateless type/object rendering helpers used by Rust codegen

The stateful, per-object rendering orchestration (which threads imports and the
type-render callbacks through one object) lives in ``rust_generator.codegen`` as
:class:`~.codegen._ObjectRenderer`; this module keeps only the pure leaf helpers
it builds on.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

from ..utils import UnsupportedTypeError
from . import consts as C
from .consts import RUST_NO_IMPORT_FULLPATH, RUST_UNSUPPORTED_ORIGINS


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
    """Import collector / ``use`` registrar for Rust codegen.

    The language-agnostic ``cli`` treats this as an opaque token: it asks the
    backend to create one, seed it from ``import-object`` directives, and later
    render it. Only the Rust backend reaches inside.

    All ``use`` recording -- boilerplate *and* type-render callbacks -- goes
    through the single :meth:`record` method, which dedups by full path. Two
    *different* paths wanting the same in-scope name raise
    :class:`UnsupportedTypeError` (the enclosing object is skipped with a
    warning): such collisions only arise from pathological type names, which
    the Rust backend declares unsupported rather than auto-aliasing.
    """

    items: list[RustUse] = dataclasses.field(default_factory=list)

    def record(self, name: str, alias: str | None = None) -> str:
        """Record a ``use`` (deduped by path) and return the in-scope name.

        * bare prelude/primitive types (``i64`` / ``bool`` / ``()`` / ``Option``)
          carry no ``::`` -> no ``use`` is recorded and the bare name is returned;
        * a path in :data:`~.consts.RUST_NO_IMPORT_FULLPATH` is rendered
          fully-qualified inline with no ``use`` (avoids shadowing a prelude name
          -- e.g. ``String`` vs ``std::string::String``);
        * a path already recorded reuses its binding rather than emitting a
          duplicate ``use``;
        * a *different* path wanting an in-scope name already taken raises
          :class:`UnsupportedTypeError` -> the enclosing object is skipped.
          Only pathological type names hit this (e.g. a type key whose leaf is
          ``Object``); rename the type or hand-write the binding outside the
          markers.
        """
        probe = RustUse(name, alias=alias)
        if not probe.as_use_line():
            return probe.leaf  # bare prelude/primitive: no import, no tracking.
        if probe.full_name in RUST_NO_IMPORT_FULLPATH:
            return probe.full_name  # rendered fully-qualified inline; no `use`.
        # `items` stays small (a handful of `use`s per file): linear scans.
        for item in self.items:
            if item.full_name == probe.full_name:
                return item.name_in_scope  # same path already imported.
        bound = probe.name_in_scope
        if any(item.name_in_scope == bound for item in self.items):
            raise UnsupportedTypeError(
                name, f"`use` name {bound!r} collides with an existing import"
            )
        self.items.append(probe)
        return bound


if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema

    from ..utils import ObjectInfo


def render_rust_type(schema: TypeSchema, ty_render: Callable[[str], str]) -> str:
    """Render a :class:`TypeSchema` into a Rust type expression.

    ``ty_render`` maps a leaf origin name to its Rust leaf name and records the
    ``use`` import it needs through :meth:`RustImports.record`.

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
        assert args  # TypeSchema's post_init guarantees exactly one arg.
        inner = render_rust_type(args[0], ty_render)
        return f"{ty_render('Optional')}<{inner}>"

    if origin == "Array":
        assert args  # TypeSchema's post_init fills a missing element type with (Any,).
        elem = render_rust_type(args[0], ty_render)
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


def _class_is_mutable(info: ObjectInfo) -> tuple[bool, bool]:
    """Return (mutable, mixed) from the fields' read-only flags.

    mutable = all fields writable; immutable = all read-only; mixed (some of
    each) -> not mutable, mixed=True so the caller can warn.
    """
    writable = [not f.frozen for f in info.fields]
    if all(writable):
        return True, False
    if not any(writable):
        return False, False
    return False, True


def _deref_impl(ref: str, target: str, field: str, mutable: bool) -> list[str]:
    """Emit ``Deref`` (+ ``DerefMut`` when ``mutable``) for ``ref`` -> ``target``."""
    out = [
        f"impl Deref for {ref} {{",
        f"    type Target = {target};",
        f"    fn deref(&self) -> &{target} {{",
        f"        &self.{field}",
        "    }",
        "}",
        "",
    ]
    if mutable:
        out += [
            f"impl DerefMut for {ref} {{",
            f"    fn deref_mut(&mut self) -> &mut {target} {{",
            f"        &mut self.{field}",
            "    }",
            "}",
            "",
        ]
    return out


def _packed_args_expr(params: list[tuple[str, str]], is_member: bool) -> str:
    """Build the ``&[AnyView]`` element list for a packed call.

    A param whose type already rendered as ``AnyView`` (a top-level ``Any``
    argument) is passed through as-is.
    """
    parts = ["AnyView::from(&*self)"] if is_member else []
    for name, ty in params:
        parts.append(name if ty == "AnyView" else f"AnyView::from(&{name})")
    return ", ".join(parts)


def _packed_call_lines(fvar: str, getter: list[str], packed: str, ret: str) -> list[str]:
    """Build the body lines for a reflected call via ``Function::call_packed``.

    ``getter`` is the (multi-line) binding of ``fvar`` to the reflected method.
    """
    if ret == "Any":
        return [*getter, f"    {fvar}.call_packed(&[{packed}])"]
    return [*getter, f"    Ok({fvar}.call_packed(&[{packed}])?.try_into()?)"]
