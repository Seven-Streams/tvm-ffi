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
from .consts import RUST_KEYWORDS, RUST_NON_RAW_IDENTS, RUST_UNSUPPORTED_ORIGINS

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema


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

    Two *different* paths wanting the same in-scope name raise
    :class:`UnsupportedTypeError`: the backend declares such pathological type
    names unsupported rather than inventing an alias. Codegen transactions
    ensure a failed block does not leak partial imports.
    """

    items: list[RustUse] = dataclasses.field(default_factory=list)

    def record(self, name: str) -> str:
        """Record a ``use`` (deduped by path) and return the in-scope name (the leaf).

        Bare prelude/primitive names record no ``use``.
        """
        probe = RustUse(name)
        if not probe.as_use_line():
            return probe.leaf
        # `items` stays small (a handful of `use`s per file): linear scans.
        for item in self.items:
            if item.path == probe.path:
                return item.leaf
        if any(item.leaf == probe.leaf for item in self.items):
            raise UnsupportedTypeError(
                name, f"`use` name {probe.leaf!r} collides with an existing import"
            )
        self.items.append(probe)
        return probe.leaf


def _escape_ident(name: str) -> str:
    """Escape a reflected field/method name into a valid Rust identifier.

    A Rust-keyword name (``impl``, ``type``, ``match``, ...) becomes the raw
    identifier ``r#<name>``, valid in every code position the generator emits
    (struct fields, constructor parameters, and ``fn`` names). The few names
    rustc rejects even raw (``crate``/``self``/``Self``/``super``/``_``) have
    no rendering, so the enclosing declaration fails loudly.
    Message strings and FFI lookup names keep the original spelling -- escape
    only at code positions.
    """
    if name in RUST_NON_RAW_IDENTS:
        raise UnsupportedTypeError(
            name, f"name {name!r} cannot be a Rust identifier (not even raw: `r#{name}`)"
        )
    if name in RUST_KEYWORDS:
        return f"r#{name}"
    return name


def _element_rust_type(elem: TypeSchema, ty_render: Callable[[str], str]) -> str:
    """Render a container element / ``Optional`` payload type.

    An ``Any`` element renders as the owning dynamic ``AnyValue`` carrier. It
    implements ``AnyCompatible`` without narrowing the value domain, so
    ``Array<AnyValue>`` / ``Map<_, AnyValue>`` preserve scalars, strings,
    objects, containers, and ``None`` from the C++ ``...<Any>`` field. Every
    other origin recurses through :func:`render_rust_type`, which rejects the unrepresentable
    (``Dict``/``List``/``Union``/``tuple`` up front, and ``void*`` / unmapped
    leaves at ``ty_render``).
    """
    if elem.origin == "Any":
        return ty_render("AnyValue")
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
        # Value position only (`None` <-> kTVMFFINone via Any); FIELD position
        # is layout-sensitive and routes through `render_struct_field`.
        (payload,) = args  # TypeSchema's post_init enforces exactly one argument.
        return f"Option<{_element_rust_type(payload, ty_render)}>"

    if origin == "Callable":
        # The crate's Function is type-erased: no generic params.
        return ty_render("Callable")

    return ty_render(origin)  # leaf / object type


def _deref_impl(ref: str, target: str, field: str) -> list[str]:
    """Emit an immutable ``Deref`` for ``ref`` -> ``target``.

    Object references are shared owning handles. Even when C++ marks the
    underlying object type mutable, ``&mut`` access to one Rust handle cannot
    prove the allocation is unaliased, so generated mirrors never implement
    ``DerefMut``.
    """
    return [
        f"impl Deref for {ref} {{",
        f"    type Target = {target};",
        f"    fn deref(&self) -> &{target} {{",
        f"        &self.{field}",
        "    }",
        "}",
        "",
    ]


def _packed_args_expr(params: list[tuple[str, str]], is_member: bool) -> str:
    """Build the ``&[AnyView]`` element list for a packed call.

    A param whose type already rendered as ``AnyView`` (a top-level ``Any``
    argument) is passed through as-is.
    """
    parts = ["tvm_ffi::object::as_any_view(self)"] if is_member else []
    for name, ty in params:
        is_any_view = ty == "AnyView" or ty.startswith("AnyView<")
        parts.append(name if is_any_view else f"AnyView::from(&{name})")
    return ", ".join(parts)


def _packed_call_lines(fvar: str, getter: list[str], packed: str, ret: str) -> list[str]:
    """Build the body lines for a reflected call via ``Function::call_packed``.

    ``getter`` is the (multi-line) binding of ``fvar`` to the reflected method.
    """
    if ret == "Any":
        return [*getter, f"    {fvar}.call_packed(&[{packed}])"]
    return [*getter, f"    Ok({fvar}.call_packed(&[{packed}])?.try_into()?)"]
