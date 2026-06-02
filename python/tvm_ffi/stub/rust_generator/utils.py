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
    """Import collector + alias-aware ``use`` registrar for Rust codegen.

    The language-agnostic ``cli`` treats this as an opaque token: it asks the
    backend to create one, seed it from ``import-object`` directives, and later
    render it. Only the Rust backend reaches inside.

    All ``use`` recording -- boilerplate *and* type-render
    callbacks -- goes through the single :meth:`record` method,
    so there is exactly one view of which leaf names are taken. The collision
    trackers are rebuilt from :attr:`items` in ``__post_init__`` so that a seeded
    copy (``RustImports(items=list(other.items))``) stays consistent: whoever
    records a leaf name first keeps it, and a later, differently-pathed ``use``
    of the same leaf is aliased (``use b::Foo as Foo2;``) rather than emitting a
    duplicate ``use`` that fails to compile.
    """

    items: list[RustUse] = dataclasses.field(default_factory=list)
    #: in-scope name keyed by full ``::`` path -- dedup of repeat references.
    _binding_of: dict[str, str] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    #: every identifier already bound into scope -- drives alias-on-collision.
    _used_names: set[str] = dataclasses.field(
        default_factory=set, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Rebuild the collision trackers from any pre-seeded ``items``.

        Bare prelude names are pre-claimed: they render import-free, so any
        same-leaf ``use`` must be the side that gets aliased.
        """
        self._used_names |= C.RUST_PRELUDE_NAMES
        for use in self.items:
            self._index(use)

    def _index(self, use: RustUse) -> None:
        """Register an already-appended ``use`` into the collision trackers."""
        self._binding_of.setdefault(use.full_name, use.name_in_scope)
        self._used_names.add(use.name_in_scope)

    def record(self, name: str, alias: str | None = None) -> str:
        """Record a ``use`` (alias-aware, deduped) and return the in-scope name.

        * bare prelude/primitive types (``i64`` / ``bool`` / ``()`` / ``Option``)
          carry no ``::`` -> no ``use`` is recorded and the bare name is returned;
        * a path in :data:`~.consts.RUST_NO_IMPORT_FULLPATH` is rendered
          fully-qualified inline with no ``use`` (avoids shadowing a prelude name
          -- e.g. ``String`` vs ``std::string::String``);
        * a path already recorded reuses its binding (possibly an alias) rather
          than emitting a duplicate ``use``;
        * if a *different* path wants a name already taken (e.g. boilerplate's
          ``tvm_ffi::object::Object`` and a field's ``tvm_ffi::Object``), the
          later one is aliased -- ``use tvm_ffi::Object as Object2;`` -- and the
          alias is returned. This type-vs-type clash is the only ``use`` collision
          that arises in Rust (function/method names live in a separate namespace,
          and methods are scoped inside ``impl`` blocks, so they never shadow a
          ``use``).
        """
        probe = RustUse(name, alias=alias)
        if not probe.as_use_line():
            return probe.leaf  # bare prelude/primitive: no import, no tracking.
        if probe.full_name in RUST_NO_IMPORT_FULLPATH:
            return probe.full_name  # rendered fully-qualified inline; no `use`.
        full = probe.full_name
        if full in self._binding_of:
            return self._binding_of[full]  # same path already imported (maybe aliased).
        bound = probe.name_in_scope
        if bound in self._used_names:
            n = 2
            while f"{probe.name_in_scope}{n}" in self._used_names:
                n += 1
            bound = f"{probe.name_in_scope}{n}"
        use = probe if bound == probe.name_in_scope else RustUse(full, alias=bound)
        self.items.append(use)
        self._index(use)
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


def _rust_ident(name: str) -> str:
    """Make ``name`` a usable Rust identifier (raw-escape keywords).

    Raises :class:`UnsupportedTypeError` for the reserved names that cannot be
    raw identifiers (``self``/``Self``/``super``/``crate``): no Rust spelling of
    such a field/method exists, so the enclosing object is skipped.
    """
    if name in C.RUST_RAW_IDENT_FORBIDDEN:
        raise UnsupportedTypeError(name, f"name {name!r} cannot be a Rust identifier")
    if name in C.RUST_KEYWORDS:
        return f"r#{name}"
    return name


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


def _deref_impl(ref: str, target: str, field: str, deref: str, derefmut: str) -> list[str]:
    """Emit `Deref` (+ `DerefMut` unless ``derefmut`` is empty) for `ref` -> `target`.

    ``deref``/``derefmut`` are the traits' in-scope (possibly aliased) names; the
    method names stay ``deref``/``deref_mut`` regardless -- implementing an
    aliased trait still defines its real methods.
    """
    out = [
        f"impl {deref} for {ref} {{",
        f"    type Target = {target};",
        f"    fn deref(&self) -> &{target} {{",
        f"        &self.{field}",
        "    }",
        "}",
        "",
    ]
    if derefmut:
        out += [
            f"impl {derefmut} for {ref} {{",
            f"    fn deref_mut(&mut self) -> &mut {target} {{",
            f"        &mut self.{field}",
            "    }",
            "}",
            "",
        ]
    return out


def _packed_args_expr(
    params: list[tuple[str, str]], is_member: bool, anyview: str = "AnyView"
) -> str:
    """Build the ``&[AnyView]`` element list for a packed call.

    ``anyview`` is the in-scope (possibly aliased) name of ``tvm_ffi::AnyView``;
    param types equal to it are passed through as-is.
    """
    parts = [f"{anyview}::from(&*self)"] if is_member else []
    for name, ty in params:
        parts.append(name if ty == anyview else f"{anyview}::from(&{name})")
    return ", ".join(parts)


def _packed_call_lines(fvar: str, getter: list[str], packed: str, ret: str) -> list[str]:
    """Build the body lines for a reflected call via ``Function::call_packed``.

    ``getter`` is the (multi-line) binding of ``fvar`` to the reflected method.
    """
    if ret == "Any":
        return [*getter, f"    {fvar}.call_packed(&[{packed}])"]
    return [*getter, f"    Ok({fvar}.call_packed(&[{packed}])?.try_into()?)"]
