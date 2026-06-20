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
"""Rust generator helpers for ``tvm-ffi-stubgen``.

Import/use modelling (:class:`RustUse`, :class:`RustImports`) and stateless
rendering helpers; the stateful per-object orchestration lives in
``rust_generator.codegen``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

from ..utils import UnsupportedTypeError
from . import consts as C
from .consts import RUST_NOT_ANY_COMPATIBLE_ORIGINS, RUST_UNSUPPORTED_ORIGINS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tvm_ffi.core import TypeSchema


@dataclasses.dataclass(frozen=True, eq=True)
class RustUse:
    """A single Rust ``use`` item: ``use <path>;``.

    Construction normalizes dotted FFI names into ``::`` paths. A builtin
    ``ffi.*`` type key is rewritten to the external ``tvm_ffi`` crate via
    :data:`~.consts.RUST_MOD_MAP` (``ffi.String -> tvm_ffi::String``); any other
    dotted name is a generated-tree type key and is rooted at ``crate`` (the
    tree is mounted at the crate root, so cross-module paths resolve from there:
    ``ir.Attrs -> crate::ir::Attrs``). Already ``::``-qualified paths pass
    through; bare names (``i64``, ``bool``) stay bare and need no ``use``.
    """

    path: str

    def __init__(self, name: str) -> None:
        """Normalize ``name`` into a Rust ``use`` path and store it."""
        if "::" not in name and "." in name:
            head, _, tail = name.partition(".")
            if head in C.RUST_MOD_MAP:
                # A builtin `ffi.*` type key: lives in the external `tvm_ffi` crate.
                name = f"{C.RUST_MOD_MAP[head]}.{tail}"
            else:
                # A generated-tree type key (e.g. `ir.Attrs`): its module path
                # resolves from the crate root, where the tree is mounted.
                name = f"crate.{name}"
        object.__setattr__(self, "path", name.replace(".", "::"))

    @property
    def leaf(self) -> str:
        """The final path segment (the in-scope name), e.g. ``Array`` for ``tvm_ffi::Array``."""
        return self.path.rsplit("::", 1)[-1]

    def as_use_line(self) -> str:
        """Render the ``use`` statement, or ``""`` for a bare prelude/primitive type."""
        if "::" not in self.path:
            return ""
        return f"use {self.path};"


@dataclasses.dataclass
class RustImports:
    """Collects the ``use`` items of one generated file (all via :meth:`record`).

    A cross-module import whose in-scope leaf is already claimed -- by a type
    this file defines (see :meth:`seed_local_types`) or by another import -- is
    referenced by its full ``crate::``-rooted path inline instead of being
    imported under the colliding leaf. This is the disambiguation strategy
    (e.g. ``tirx.Call`` defined locally and ``relax.expr.Call`` referenced both
    spell ``Call``); it avoids both the E0255 import-vs-local collision and the
    E0252 two-imports collision without auto-aliasing.
    """

    items: list[RustUse] = dataclasses.field(default_factory=list)
    #: Canonical ``crate::``-rooted paths of the types this file defines (each
    #: type contributes both its ref name and its ``<Name>Obj`` value-struct
    #: twin). A reference to one of these is an in-scope sibling: no ``use``, no
    #: qualification. Seeded once per file before any block is rendered.
    local_paths: set[str] = dataclasses.field(default_factory=set)
    #: The in-scope leaf names claimed by :attr:`local_paths`; a cross-module
    #: import landing on one of these is full-qualified to dodge the collision.
    local_leaves: set[str] = dataclasses.field(default_factory=set)

    def seed_local_types(self, type_keys: Iterable[str]) -> None:
        """Register the file's own type definitions (and their ``<Name>Obj`` twins).

        Called once per file before rendering so a forward reference to a
        same-file type is recognised as in-scope and a colliding cross-module
        import is disambiguated regardless of declaration order.
        """
        for type_key in type_keys:
            for use in (RustUse(type_key), RustUse(f"{type_key}Obj")):
                self.local_paths.add(use.path)
                self.local_leaves.add(use.leaf)

    def record(self, name: str) -> str:
        """Record a ``use`` and return the in-scope spelling to use in the body.

        Bare prelude/primitive names record no ``use``. A reference to a
        same-file type returns its bare leaf (the sibling is already in scope;
        its ``use`` is dropped by the import section). A cross-module import
        returns its bare leaf when the leaf is free, or its full
        ``crate::``-rooted path (no ``use``) when the leaf is already claimed.
        """
        probe = RustUse(name)
        if not probe.as_use_line():
            return probe.leaf
        # `items` stays small (a handful of `use`s per file): linear scans.
        for item in self.items:
            if item.path == probe.path:  # dedup by path (local or cross-module)
                return item.leaf
        if probe.path not in self.local_paths and (
            probe.leaf in self.local_leaves or any(item.leaf == probe.leaf for item in self.items)
        ):
            # Cross-module import on an already-claimed leaf: reference it by its
            # full path inline; emit no `use` (which would collide).
            return probe.path
        self.items.append(probe)
        return probe.leaf


def schema_contains(schema: TypeSchema, origins: frozenset[str]) -> bool:
    """Whether ``schema`` mentions any origin in ``origins`` anywhere (recursive)."""
    if schema.origin in origins:
        return True
    return any(schema_contains(arg, origins) for arg in (schema.args or ()))


def render_rust_type(schema: TypeSchema, ty_render: Callable[[str], str]) -> str:
    """Render a :class:`TypeSchema` into a Rust type expression.

    ``ty_render`` maps a leaf origin name to its Rust leaf name, recording the
    ``use`` it needs via :meth:`RustImports.record`. Raises
    :class:`UnsupportedTypeError` for origins the crate cannot represent.
    """
    origin = schema.origin
    args = schema.args or ()

    if origin in RUST_UNSUPPORTED_ORIGINS:
        raise UnsupportedTypeError(origin)

    if origin == "Optional":
        # Native `Option<T>` at function boundaries and when nested. A *direct*
        # struct field is special-cased to the layout-mirror in
        # `render_struct_field`, which never reaches this branch. The inner must
        # be AnyCompatible to cross the Any boundary (`Option<Object>` etc. would
        # not compile), otherwise there is no native rendering -> skip the object.
        assert args  # TypeSchema's post_init fills a missing inner type.
        if schema_contains(args[0], RUST_NOT_ANY_COMPATIBLE_ORIGINS):
            raise UnsupportedTypeError(origin)
        return f"Option<{render_rust_type(args[0], ty_render)}>"

    if origin == "Array":
        # `Array<T>` requires `T: AnyCompatible`; an `Any` / bare-`Object` element
        # has no such rendering (`Array<Any>` would not compile), so it skips the
        # enclosing object -- symmetric with `Map` and `Optional`.
        assert args  # TypeSchema's post_init fills a missing element type.
        if schema_contains(args[0], RUST_NOT_ANY_COMPATIBLE_ORIGINS):
            raise UnsupportedTypeError(origin)
        elem = render_rust_type(args[0], ty_render)
        return f"{ty_render('Array')}<{elem}>"

    if origin == "Map":
        # The crate's read-only `Map<K, V>`. Both the key and value must be
        # AnyCompatible to cross the Any boundary (`Map<Any, Any>` -- the default
        # for an untyped map -- is not, so it skips the enclosing object, exactly
        # as before). As a `#[repr(C)]` field a `Map` is a single pointer, so it
        # is layout-compatible with C++ `Map : ObjectRef` and needs no mirror.
        assert args  # TypeSchema's post_init fills a missing K/V pair with (Any, Any).
        if any(schema_contains(a, RUST_NOT_ANY_COMPATIBLE_ORIGINS) for a in args):
            raise UnsupportedTypeError(origin)
        params = ", ".join(render_rust_type(a, ty_render) for a in args)
        return f"{ty_render('Map')}<{params}>"

    if origin == "Callable":
        # The crate's Function is type-erased: no generic params.
        return ty_render("Callable")

    return ty_render(origin)  # leaf / object type


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
