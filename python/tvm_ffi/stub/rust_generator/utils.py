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
import re
from typing import TYPE_CHECKING, Callable

from ..utils import UnsupportedTypeError
from . import consts as C
from .consts import RUST_UNSUPPORTED_ORIGINS

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema


_RUST_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_RUST_KEYWORDS = frozenset(
    {
        "abstract",
        "as",
        "async",
        "await",
        "become",
        "box",
        "break",
        "const",
        "continue",
        "do",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "final",
        "fn",
        "for",
        "gen",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "macro",
        "match",
        "mod",
        "move",
        "mut",
        "override",
        "priv",
        "pub",
        "ref",
        "return",
        "static",
        "struct",
        "trait",
        "true",
        "try",
        "type",
        "typeof",
        "union",
        "unsafe",
        "unsized",
        "use",
        "virtual",
        "where",
        "while",
        "yield",
    }
)
_RUST_NON_RAW_IDENTIFIERS = frozenset({"crate", "self", "Self", "super"})


def rust_identifier(name: str) -> str:
    """Return a valid Rust identifier, or reject a name Rust cannot express."""
    if name == "_" or not _RUST_IDENTIFIER.fullmatch(name):
        raise UnsupportedTypeError(name, f"invalid Rust identifier {name!r}")
    if name in _RUST_NON_RAW_IDENTIFIERS:
        raise UnsupportedTypeError(name, f"Rust reserves identifier {name!r}")
    return f"r#{name}" if name in _RUST_KEYWORDS else name


def rust_type_key_path(type_key: str, *, object_data: bool = False) -> str:
    """Render a reflected dotted type key as an escaped Rust module path."""
    parts = type_key.split(".")
    if object_data:
        parts[-1] += "Obj"
    return "::".join(rust_identifier(part) for part in parts)


@dataclasses.dataclass(frozen=True, eq=True)
class RustUse:
    """A single Rust ``use`` item: ``use <path>;``.

    Construction normalizes dotted FFI names into ``::`` paths, rewriting the
    leading module via :data:`~.consts.RUST_MOD_MAP` (``ffi.String ->
    tvm_ffi::String``); ``::`` paths pass through; bare names (``i64``,
    ``bool``) stay bare and need no ``use``.
    """

    path: str

    def __init__(self, name: str) -> None:
        """Normalize ``name`` into a Rust ``use`` path and store it."""
        if "::" not in name and "." in name:
            head, _, tail = name.partition(".")
            head = C.RUST_MOD_MAP.get(head, head)
            name = f"{head}.{tail}"
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

    If two paths have the same leaf, the later use stays fully qualified in the
    generated signature instead of being confused with the first import.
    """

    items: list[RustUse] = dataclasses.field(default_factory=list)
    reserved_leaves: set[str] = dataclasses.field(default_factory=set)

    def reserve(self, *names: str) -> None:
        """Reserve local item names so qualified imports cannot shadow them."""
        self.reserved_leaves.update(RustUse(name).leaf for name in names)

    def record(self, name: str) -> str:
        """Record a ``use`` (deduped by path) and return the in-scope name (the leaf).

        Bare prelude/primitive names record no ``use``.
        """
        probe = RustUse(name)
        if not probe.as_use_line():
            return probe.leaf
        if probe.leaf in self.reserved_leaves:
            return probe.path
        # `items` stays small (a handful of `use`s per file): linear scans.
        for item in self.items:
            if item.path == probe.path:
                return item.leaf
        if any(item.leaf == probe.leaf for item in self.items):
            return probe.path
        self.items.append(probe)
        return probe.leaf


def _element_rust_type(elem: TypeSchema, ty_render: Callable[[str], str]) -> str:
    """Render a container element / ``Optional`` payload type.

    ``Any`` cannot be replaced by ``ObjectRef``: dynamic containers may also
    hold scalars and strings.  The object/method renderer catches this error and
    exposes the whole value as owning ``Any`` (or an ``AnyView`` parameter).
    """
    if elem.origin == "Any":
        raise UnsupportedTypeError("Any", "dynamic container elements have no Rust generic type")
    return render_rust_type(elem, ty_render)


def render_rust_type(schema: TypeSchema, ty_render: Callable[[str], str]) -> str:
    """Render a :class:`TypeSchema` into a Rust type expression.

    ``ty_render`` maps a leaf origin name to its Rust leaf name, recording the
    ``use`` it needs via :meth:`RustImports.record`. Raises
    :class:`UnsupportedTypeError` for origins the crate cannot represent.
    """
    origin = schema.origin
    args = schema.args

    if origin in RUST_UNSUPPORTED_ORIGINS:
        raise UnsupportedTypeError(origin)

    if origin == "Array":
        assert args  # TypeSchema's post_init fills a missing element type.
        elem = _element_rust_type(args[0], ty_render)
        return f"{ty_render('Array')}<{elem}>"

    if origin == "Map":
        assert len(args) == 2  # TypeSchema's post_init fills a bare Map to (Any, Any).
        key = _element_rust_type(args[0], ty_render)
        value = _element_rust_type(args[1], ty_render)
        return f"{ty_render('Map')}<{key}, {value}>"

    if origin == "Optional":
        # Reflected values use `None` <-> kTVMFFINone through the owning Any.
        (payload,) = args  # TypeSchema's post_init enforces exactly one argument.
        return f"::core::option::Option<{_element_rust_type(payload, ty_render)}>"

    if origin == "Callable":
        # The crate's Function is type-erased: no generic params.
        return ty_render("Callable")

    return ty_render(origin)  # leaf / object type


def _deref_impl(ref: str, target: str, field: str) -> list[str]:
    """Emit an immutable ``Deref`` from ``ref`` to ``target``."""
    return [
        f"impl ::std::ops::Deref for {ref} {{",
        f"    type Target = {target};",
        f"    fn deref(&self) -> &{target} {{",
        f"        &self.{field}",
        "    }",
        "}",
        "",
    ]


def _packed_args_expr(params: list[tuple[str, str, bool]], self_expr: str | None) -> str:
    """Build the ``&[AnyView]`` element list for a packed call.

    Dynamic parameters are already ``AnyView`` and pass through unchanged.
    ``self_expr`` supplies the borrowed object for an instance method.
    """
    parts = [self_expr] if self_expr is not None else []
    for name, _ty, dynamic in params:
        parts.append(name if dynamic else f"::tvm_ffi::AnyView::from(&{name})")
    return ", ".join(parts)


def _packed_call_lines(
    fvar: str,
    getter: list[str],
    packed: str,
    dynamic_result: bool,
    *,
    kwargs: str | None = None,
) -> list[str]:
    """Build the body lines for a reflected call via ``Function::call_packed``.

    ``getter`` is the (multi-line) binding of ``fvar`` to the reflected method.
    """
    if kwargs is None:
        call_expr = f"{fvar}.call_packed(&[{packed}])"
    else:
        call_expr = f"{fvar}.call_packed_with_kwargs(&[{packed}], &[{kwargs}])"
    if dynamic_result:
        return [*getter, f"    {call_expr}"]
    return [*getter, f"    {call_expr}?.try_into()"]
