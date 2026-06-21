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

Codegen orchestration lives here; low-level rendering helpers live in
``rust_generator.utils``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
from typing import TYPE_CHECKING

from tvm_ffi.core import MISSING

from .. import consts as C
from ..lib_state import object_info_from_type_key
from . import consts as C_RUST
from .utils import (
    RustImports,
    UnsupportedTypeError,
    _deref_impl,
    _packed_args_expr,
    _packed_call_lines,
    render_rust_type,
    schema_contains,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, NamedTypeSchema, ObjectInfo, Options


# --- native (FFI-free) construction eligibility ------------------------------


def _rust_string_literal(s: str) -> str:
    """Escape ``s`` as a double-quoted Rust string literal."""
    out = ['"']
    for ch in s:
        if ch in ('"', "\\"):
            out.append("\\" + ch)
        elif ch.isprintable():
            out.append(ch)
        else:
            out.append(f"\\u{{{ord(ch):x}}}")
    out.append('"')
    return "".join(out)


def _rust_ident(name: str) -> str:
    """Escape a C++-derived name for use as a bare Rust identifier.

    A name that is a Rust keyword is spelled as the raw identifier ``r#<name>``;
    the few keywords ``r#`` cannot spell (``crate`` / ``self`` / ``super`` /
    ``Self``) are suffix-renamed (``self -> self_``) instead. Non-keywords pass
    through unchanged. The *original* name is still used wherever the C++ name
    must appear verbatim -- the reflection field/method-name string literals and
    human-readable error text -- so escaping is purely a Rust-syntax concern.
    """
    if name in C_RUST.RUST_RAW_IDENT_FORBIDDEN:
        return f"{name}_"
    if name in C_RUST.RUST_KEYWORDS:
        return f"r#{name}"
    return name


#: Field origins whose Rust rendering is ``tvm_ffi::String`` (so a ``str`` default
#: materializes as a ``String`` literal). A ``str`` default on any other field --
#: notably the ``dtype`` string registered for a ``DLDataType`` field -- has no
#: such rendering and must NOT be emitted as a ``String`` (it would be an E0308
#: type mismatch).
_DEFAULT_STRING_ORIGINS = frozenset({"str", "ffi.String"})


def _default_expr(field: NamedTypeSchema) -> str | None:
    """Render ``field``'s registered default as a Rust expression (``None``: can't).

    The rendering must match the field's Rust type, so each literal is gated on
    the field's ``origin``: a ``bool`` / ``int`` / finite-``float`` literal only
    for a ``bool`` / ``int`` / ``float`` field (an ``int`` default on a ``float``
    field coerces to a float literal), and a ``str`` only for a ``String`` field
    (-> ``tvm_ffi::String``). A value whose type does not match the field -- e.g.
    the string default registered for a ``DLDataType`` / ``Device`` field, or any
    object/container/non-finite-float -- has no self-evident Rust literal and
    yields ``None`` (the caller then blocks native construction rather than
    emitting a type-mismatched default).
    """
    value = field.default
    origin = field.origin
    if isinstance(value, bool):
        # `bool` is an `int` subclass; handle it first. Only a `bool` field.
        return ("true" if value else "false") if origin == "bool" else None
    if origin == "int" and isinstance(value, int):
        return repr(value)
    if origin == "float" and isinstance(value, (int, float)):
        return repr(float(value)) if math.isfinite(value) else None
    if origin in _DEFAULT_STRING_ORIGINS and isinstance(value, str):
        return f"tvm_ffi::String::from({_rust_string_literal(value)})"
    return None


def _native_blocker(info: ObjectInfo) -> str | None:
    """Why ``info`` cannot be constructed natively; ``None`` when it can.

    The native builder allocates the struct directly, binding every own
    field from its setter or a stubgen-rendered default and silently
    bypassing any C++ constructor logic -- that is the opted-in behavior, so
    native is used whenever possible. There is no FFI fallback: a blocked
    type gets no generated constructor at all (the user hand-writes one).
    """
    if not info.has_init:
        return "the type has no reflected constructor"
    for field in info.fields:
        if field.origin == "Optional":
            # A direct `Optional<T>` field is the view-only layout-mirror
            # `tvm_ffi::Optional` (no Rust constructor); created on the C++ side.
            return f"field {field.name!r} is an ffi::Optional (view-only, C++-constructed)"
        if field.default_is_factory:
            return f"field {field.name!r} uses a default factory (FFI-only)"
        if field.default is not MISSING and _default_expr(field) is None:
            return f"the default value of field {field.name!r} has no Rust rendering"
    parent = info.parent_type_key
    if parent in (None, "ffi.Object") or _native_eligible(parent):
        return None
    return f"parent {parent!r} is not natively constructible"


def _info_native_eligible(info: ObjectInfo) -> bool:
    """Whether ``info`` can be constructed natively (see :func:`_native_blocker`)."""
    return _native_blocker(info) is None


def _native_eligible(type_key: str) -> bool:
    """Type-key wrapper of :func:`_info_native_eligible` (parent recursion).

    A type that cannot be resolved is warned about and treated as non-native.
    Deliberately uncached: a cache would go stale across registry changes.
    """
    try:
        info = object_info_from_type_key(type_key)
    except Exception as e:  # any failure means "cannot prove native-safe"
        print(
            f"{C.TERM_YELLOW}[Warning] cannot resolve type {type_key!r} for native "
            f"construction ({type(e).__name__}: {e}); treating it as non-native"
            f"{C.TERM_RESET}"
        )
        return False
    return _info_native_eligible(info)


def _is_any_compatible(schema: TypeSchema) -> bool:
    """Whether ``schema``'s Rust rendering implements ``AnyCompatible``.

    True unless ``Any`` or the bare base ``Object`` appears anywhere -- those are
    the only renderable leaves that are not ``AnyCompatible`` (so they cannot be
    the ``V`` of a native ``Option<V>`` accessor / marshal).
    """
    return not schema_contains(schema, C_RUST.RUST_NOT_ANY_COMPATIBLE_ORIGINS)


def _layout_fields(fields: list[NamedTypeSchema]) -> list[NamedTypeSchema]:
    """Sort own fields by reflection ``offset`` (C++ memory order).

    Registration order need not match memory order, but the ``#[repr(C)]``
    struct is positional. Fields without an offset (synthetic ``ObjectInfo``s
    in tests) keep registration order.
    """
    if any(f.offset is None for f in fields):
        return list(fields)
    return sorted(fields, key=lambda f: f.offset)


def _warn_offset_mismatch(type_key: str | None, fields: list[NamedTypeSchema]) -> None:
    """Warn when ``#[repr(C)]`` cannot reproduce the recorded field offsets.

    Recomputes each field's ``#[repr(C)]`` placement from the previous field's
    end, using the reflected per-field ``alignment``. When alignment is missing
    (synthetic ``ObjectInfo``s in tests) it is approximated from ``size``
    (largest power of two, capped at 8), which can false-positive on composite
    FFI structs like ``DLDevice``. A mismatch only warns; the binding is still
    emitted. Fields without offset/size metadata are skipped and reset the
    running position.
    """
    prev_end: int | None = None
    for field in fields:
        if field.offset is None or field.size is None:
            prev_end = None
            continue
        if prev_end is not None:
            align = field.alignment or min(8, field.size & -field.size)
            placed = (prev_end + align - 1) // align * align
            if placed != field.offset:
                print(
                    f"{C.TERM_YELLOW}[Warning] object {type_key}: field "
                    f"{field.name!r} is at C++ offset {field.offset}, but the "
                    f"generated #[repr(C)] layout places it at offset {placed}; "
                    f"the Rust struct may not match the C++ object layout"
                    f"{C.TERM_RESET}"
                )
        # Resync to the recorded offset so one hole yields one warning.
        prev_end = field.offset + field.size


@dataclasses.dataclass
class _ObjectRenderer:
    """Renders one ``object/<key>`` block into Rust source lines.

    Holds the per-object rendering context (imports, ``ty_map``, resolved
    names) so helper methods don't have to thread it through.
    """

    info: ObjectInfo
    leaf: str
    obj_struct: str
    base_type: str
    is_root: bool
    imports: RustImports
    ty_map: dict[str, str]
    #: Whether to verify that every referenced object type is itself buildable
    #: (skip the enclosing type if it references a skipped type). Disabled while
    #: dry-rendering for the buildability probe itself, to break the recursion.
    check_refs: bool = True
    #: Rendered inner type of each AnyCompatible Optional field, keyed by field
    #: name. Populated when the struct field is rendered and reused by the
    #: accessor pass so the inner is rendered once (not twice) per field.
    _opt_inner: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def base_field(self) -> str:
        """Name of the synthetic parent-embed field, avoiding a real-field clash.

        The parent is embedded as the first ``#[repr(C)]`` field, normally named
        ``base``. But a C++ type may itself have a reflected field named ``base``
        (e.g. ``tirx.Ramp``, ``arith.IterSumExpr``); two ``base`` fields would be
        an E0124 collision, so prefix ``_`` until the name is unique among the
        own fields. ``#[derive(Object)]`` reads the first field positionally, so
        any name works.
        """
        names = {f.name for f in self.info.fields}
        name = "base"
        while name in names:
            name = f"_{name}"
        return name

    def _ty_render(self, origin: str) -> str:
        """Resolve a leaf origin to its Rust name and record its ``use``.

        Unmapped dotted names that are generated-tree object type keys pass
        through (``RustUse`` roots them at ``crate::``). Two kinds raise
        :class:`UnsupportedTypeError` (skipping the enclosing object): an unmapped
        bare origin (e.g. ``const char*``), and an unmapped ``ffi.*`` builtin --
        ``ffi.*`` is reserved for the crate's own types, exposed only through the
        explicit ``ty_map``, so an unmapped one (e.g. ``ffi.reflection.AccessPath``,
        which the crate has no Rust type for) must not be passed through as a
        nonexistent ``tvm_ffi::`` path.
        """
        mapped = self.ty_map.get(origin)
        if mapped is None:
            if "." not in origin or origin.startswith("ffi."):
                raise UnsupportedTypeError(origin)
            # A generated-tree object type key. If its own binding was/will-be
            # skipped (not buildable), referencing it would dangle -- the `use`
            # would point at a type the crate never emits and the field/arg type
            # would not exist. Skip the enclosing type instead (raise *before*
            # recording, so no dangling import leaks). The probe is disabled while
            # dry-rendering for buildability, which breaks the recursion.
            if self.check_refs and not _base_buildable(origin, self.ty_map):
                raise UnsupportedTypeError(origin, f"referenced type {origin!r} is skipped")
            mapped = origin
        return self.imports.record(mapped)

    def render_struct_field(self, schema: NamedTypeSchema) -> str:
        """Render a directly-laid-out struct field type, width-correct for scalars.

        An ``int32_t`` field must render as ``i32``, not the schema-erased
        default ``i64``; the width comes from reflection's per-field ``size``.
        A direct ``Optional<T>`` field renders as the layout-mirror
        ``tvm_ffi::Optional<T, A, N>`` (size/alignment from reflection) -- this is
        the *only* position the mirror is used; elsewhere ``Optional`` is the
        native ``Option<T>``. A direct ``Map<K, V>`` field is a single pointer
        whose params are phantom, so it renders even with non-AnyCompatible K/V
        (see :meth:`_render_map_field`). Non-scalar origins render plainly. #0: any
        unrepresentable single-pointer field (a skipped object ref, ``Array``/
        ``List``/``Dict``/``Map`` of an unrepresentable element) degrades to the
        opaque ``ObjectRef`` rather than skipping the struct -- see
        :meth:`_render_opaque_field_or_raise`.
        """
        try:
            if schema.origin == "Optional":
                return self._render_optional_field(schema)
            if schema.origin == "Map":
                return self._render_map_field(schema)
            if schema.origin == "Array":
                return self._render_array_field(schema)
            narrowed = C_RUST.RUST_SCALAR_BY_SIZE.get((schema.origin, schema.size))
            if narrowed is not None:
                return narrowed
            return render_rust_type(schema, self._ty_render)
        except UnsupportedTypeError:
            return self._render_opaque_field_or_raise(schema)

    def _render_opaque_field_or_raise(self, schema: NamedTypeSchema) -> str:
        """Degrade an unrepresentable object-pointer field to an opaque ref, else re-raise.

        #0 graceful degradation: one unrepresentable field would otherwise skip
        the whole struct, cascading to every type that references it. But a
        single-pointer field -- any object reference (a generated/builtin/``ffi.*``
        type key) or a pointer-shaped container (``Array``/``List``/``Dict``/
        ``Map``/``tuple``, each one ``ObjectArc<...Obj>``) -- is layout-safe to
        substitute with the base ``tvm_ffi::ObjectRef``: the struct compiles, the
        field is read via the runtime API / downcast through ``Any``, and the
        cascade is eliminated. Origins that are NOT a single object pointer
        (``Union`` -- an inline two-word ``Any`` -- and scalars, plus an
        ``Optional`` whose layout-mirror itself failed) have no pointer fallback,
        so the original :class:`UnsupportedTypeError` is re-raised (struct
        skipped). The reflected ``size`` guards the substitution: it degrades only
        when the field is genuinely pointer-wide (``8`` bytes, or unknown in unit
        tests that omit sizes).
        """
        # A dotted origin is an object type key (generated / builtin ``ffi.*``);
        # the listed container origins are each a single ``ObjectArc<...Obj>``.
        is_object_pointer = (
            "." in schema.origin or schema.origin in C_RUST.RUST_OPAQUE_POINTER_ORIGINS
        )
        if is_object_pointer and schema.size in (None, 8):
            return self.imports.record(C_RUST.RUST_OPAQUE_OBJECT_REF)
        raise UnsupportedTypeError(schema.origin)

    def _render_map_field(self, schema: NamedTypeSchema) -> str:
        """Render a direct ``Map<K, V>`` struct field, allowing non-AnyCompatible K/V.

        As a ``#[repr(C)]`` field a ``Map`` is a single pointer
        (``ObjectArc<MapObj>``) whose ``K``/``V`` are phantom markers, so its
        layout is independent of them -- unlike an argument/return/nested
        position, where the ``Map`` must itself be ``AnyCompatible`` to marshal
        across the ``Any`` boundary. This lets the ``DictAttrs.__dict__ : Map<str,
        Any>`` keystone (and the large cascade hanging off it) render as a
        typed-pointer field with no typed accessor; the field is read via the
        runtime reflection API. Each ``K``/``V`` still renders through
        ``render_rust_type``, so a param that is not a valid Rust type at all
        (``Dict`` / ``List`` / ``Array<Any>`` / ...) still raises and skips the
        object. (A nested ``Map<_, Map<_, Any>>`` is conservatively skipped: the
        inner map keeps the arg-position guard. The flat keystone is the goal.)
        """
        args = schema.args or ()
        assert args  # TypeSchema's post_init fills a missing K/V pair with (Any, Any).
        params = ", ".join(render_rust_type(a, self._ty_render) for a in args)
        return f"{self._ty_render('Map')}<{params}>"

    def _render_array_field(self, schema: NamedTypeSchema) -> str:
        """Render a direct ``Array<T>`` struct field, allowing a type-erased element.

        #1: an ``Array`` is always a single ``ObjectArc<ArrayObj>`` pointer and every
        element is stored as a ``TVMFFIAny`` slot, so a concretely-renderable element
        becomes a typed ``Array<T>`` (e.g. ``Array<PrimExpr>``) and *anything else* --
        ``Any``, the bare base ``Object``/``ffi.Object``, or an element with no Rust
        rendering at all (``Array<Array<Any>>``, ``Array<Dict>``, an array of a
        skipped type) -- is stored type-erased as ``Array<Any>`` (``Any:
        ArrayElement``), layout-identical and still iterable: each element reads back
        as an ``Any`` and downcasts. An ``Array`` field therefore never degrades to a
        bare opaque ``ObjectRef`` (which would hide that it is even an array). This is
        FIELD-ONLY: ``Array<Any>`` is not ``AnyCompatible`` (no ``TryFrom<Any>``), so
        it cannot marshal as a method arg/return -- ``render_rust_type`` stays strict
        there and #0 omits such methods.
        """
        args = schema.args or ()
        assert args  # TypeSchema's post_init fills a missing element type.
        elem = args[0]
        erased = elem.origin in ("Any", "Object", "ffi.Object") and not elem.args
        if not erased:
            try:
                return render_rust_type(schema, self._ty_render)  # typed Array<T>
            except UnsupportedTypeError:
                erased = True  # element has no Rust rendering -> store as Array<Any>
        return f"{self._ty_render('Array')}<{self._ty_render('Any')}>"

    def _render_optional_field(self, schema: NamedTypeSchema) -> str:
        """Render a direct ``Optional<T>`` field as ``tvm_ffi::Optional<T, AlignK, N>``.

        ``N``/``AlignK`` are the reflected ``size``/``alignment`` (so the layout
        matches C++); ``T`` is a marker only. A non-AnyCompatible inner (``Any``
        or bare ``Object``) has no usable rendering, so the marker falls back to
        ``()`` and the accessor pass skips it; an AnyCompatible inner's rendering
        is cached for reuse by :meth:`_optional_accessor_blocks`.
        """
        if schema.size is None:
            raise UnsupportedTypeError("Optional", "reflected size is unavailable for the field")
        # A missing or unsupported alignment both land here (`.get(None)` -> None).
        marker = C_RUST.RUST_ALIGN_MARKER.get(schema.alignment)
        if marker is None:
            raise UnsupportedTypeError(
                "Optional", f"unsupported ffi::Optional alignment {schema.alignment}"
            )
        args = schema.args or ()
        assert args  # TypeSchema's post_init fills a missing inner type.
        inner = "()"
        if _is_any_compatible(args[0]):
            try:
                inner = render_rust_type(args[0], self._ty_render)
                self._opt_inner[schema.name] = inner  # reused by the accessor pass
            except UnsupportedTypeError:
                # #0: the inner type is itself skipped -> fall back to the opaque
                # `()` marker (the layout-mirror keeps the field; no typed accessor).
                inner = "()"
        opt = self.imports.record(C_RUST.RUST_OPTIONAL_TYPE)
        align = self.imports.record(marker)
        return f"{opt}<{inner}, {align}, {schema.size}>"

    def render_param(self, schema: TypeSchema) -> str:
        """Render an argument type (a top-level ``Any`` is the non-owning ``AnyView``)."""
        if schema.origin == "Any":
            return self.imports.record("tvm_ffi::AnyView")
        return render_rust_type(schema, self._ty_render)

    def body(self) -> list[str]:
        """Build the Rust source lines for the object (raises on unsupported types)."""
        # Boilerplate `use`s, recorded through the same collector as field types
        # so leaf collisions raise and skip the object. The derive macros are
        # spelled by full path in the attribute, never imported: their leaves
        # collide with `tvm_ffi::Object`/`ObjectRef`.
        self.imports.record("std::ops::Deref")
        # `ObjectCore` must be in scope for the generated `type_index()` calls.
        self.imports.record("tvm_ffi::ObjectCore")
        self.imports.record("tvm_ffi::ObjectArc")
        if self.is_root:
            # Same path the ty_map uses for `Object` fields, so they dedup
            # instead of colliding.
            self.base_type = self.imports.record("tvm_ffi::Object")
        else:
            # Embed the parent's `<Parent>Obj` as `base`. The parent may live in
            # another generated module, so its `use` is collected like any field
            # type (crate::-rooted; a same-file parent's import is filtered out by
            # the import section). The recorded leaf is the in-scope name used by
            # the struct field, the `Deref` target, and the builder `base` setter.
            self.base_type = self.imports.record(f"{self.info.parent_type_key}Obj")
        # C++ `_type_mutable`: class-level mutability dominates per-field `def_ro`.
        if self.info.mutable:
            self.imports.record("std::ops::DerefMut")

        leaf, obj_struct, base_type = self.leaf, self.obj_struct, self.base_type
        lines: list[str] = []
        lines += [
            "#[repr(C)]",
            "#[derive(tvm_ffi::derive::Object)]",
            f'#[type_key = "{self.info.type_key}"]',
            f"pub struct {obj_struct} {{",
            f"    {self.base_field}: {base_type},",
        ]
        for field in _layout_fields(self.info.fields):
            lines.append(f"    pub {_rust_ident(field.name)}: {self.render_struct_field(field)},")
        lines += ["}", ""]

        lines += [
            "#[repr(C)]",
            "#[derive(tvm_ffi::derive::ObjectRef, Clone)]",
            f"pub struct {leaf} {{",
            f"    data: ObjectArc<{obj_struct}>,",
            "}",
            "",
        ]

        lines += _deref_impl(leaf, obj_struct, "data", self.info.mutable)
        if not self.is_root:
            lines += _deref_impl(obj_struct, base_type, self.base_field, self.info.mutable)

        # Native (FFI-free) construction whenever the whole chain is eligible;
        # there is no FFI fallback -- a blocked constructor is skipped loudly.
        blocker = _native_blocker(self.info)
        native = blocker is None
        if self.info.has_init and not native:
            print(
                f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping "
                f"`ffi_new` because {blocker}; hand-write a constructor outside "
                f"the generated markers{C.TERM_RESET}"
            )
        lines += self._impl_block(native)
        if native:
            lines += self._builder_lines()

        lines.pop()  # every section above ends with a `""` separator
        return lines

    def _impl_block(self, native: bool) -> list[str]:
        """Emit `impl <T> { ffi_new; methods; optional accessors }` (empty when nothing)."""
        methods = [
            m for m in self.info.methods if m.schema.name.rsplit(".", 1)[-1] != "__ffi_init__"
        ]
        # Each entry is one self-contained `fn` block; joined with a blank line.
        blocks: list[list[str]] = []
        if native:  # `native` implies `has_init` (see `_native_blocker`)
            blocks.append(self._new_fn_native())
        # #0 graceful degradation: a method whose signature mentions an
        # unrepresentable type (a skipped type, `Dict`/`List`/`tuple`/`Union`, ...)
        # is OMITTED rather than skipping the whole type -- unlike a field, a method
        # has no opaque fallback, but dropping it keeps the struct + its other
        # members usable. (Field bindings remain; reach the dropped method via the
        # runtime `Function` API.)
        for method in methods:
            try:
                blocks.append(self._method_fn(method))
            except UnsupportedTypeError as e:
                print(
                    f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping method "
                    f"{method.schema.name!r} ({e}); call it via the runtime Function API"
                    f"{C.TERM_RESET}"
                )
        blocks += self._optional_accessor_blocks()
        if not blocks:
            return []

        inner: list[str] = []
        for i, block in enumerate(blocks):
            inner += block
            if i != len(blocks) - 1:
                inner.append("")

        return [
            f"impl {self.leaf} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _optional_accessor_blocks(self) -> list[list[str]]:
        """Emit a get (+ set, when the class is mutable) accessor per Optional field.

        Reads/writes the opaque field via the C++ reflection getter/setter as a
        native ``Option<V>``, caching the field lookup per call site in a
        ``OnceLock`` (resolved once, not on every call). A non-AnyCompatible inner
        (``Any`` / bare ``Object``) has no native ``Option<V>`` rendering, so its
        accessor is skipped with a warning (the field is still reachable through
        the runtime ``tvm_ffi::optional`` API). The AnyCompatible decision is made
        here directly; ``_opt_inner`` is only reused as a render cache (re-rendered
        on miss), so this pass does not depend on the struct-field pass order.
        """
        blocks: list[list[str]] = []
        for field in _layout_fields(self.info.fields):
            if field.origin != "Optional":
                continue
            args = field.args or ()
            if not args or not _is_any_compatible(args[0]):
                print(
                    f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: optional field "
                    f"{field.name!r} has no typed accessor (inner type is not AnyCompatible); "
                    f"the field is emitted -- read/write it via the runtime "
                    f"tvm_ffi::optional API{C.TERM_RESET}"
                )
                continue
            # Reuse the inner rendered in the struct-field pass; re-render on miss
            # so this pass is independent of pass ordering. #0: a skipped inner has
            # no typed accessor (the field is emitted via the `()`-marker mirror).
            inner = self._opt_inner.get(field.name)
            if inner is None:
                try:
                    inner = render_rust_type(args[0], self._ty_render)
                except UnsupportedTypeError:
                    print(
                        f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: optional field "
                        f"{field.name!r} has no typed accessor (inner type is skipped); the field "
                        f"is emitted -- read/write it via the runtime tvm_ffi::optional API"
                        f"{C.TERM_RESET}"
                    )
                    continue
            self.imports.record("tvm_ffi::Result")
            name, obj = field.name, self.obj_struct
            ident = _rust_ident(name)  # bare-identifier spelling; `name` stays the FFI field name
            # Per-call-site cache so the field-table scan runs once, not per call.
            cell = "    static CELL: std::sync::OnceLock<tvm_ffi::FieldAccess> = std::sync::OnceLock::new();"
            block = [
                f"pub fn {ident}(&self) -> Result<Option<{inner}>> {{",
                cell,
                f"    unsafe {{ self.{ident}.read_cached::<{inner}>(&CELL, "
                f'{obj}::type_index(), "{name}") }}',
                "}",
            ]
            if self.info.mutable:
                block += [
                    "",
                    f"pub fn set_{name}(&self, value: Option<{inner}>) -> Result<()> {{",
                    cell,
                    f"    unsafe {{ self.{ident}.write_cached::<{inner}>(&CELL, "
                    f'{obj}::type_index(), "{name}", value) }}',
                    "}",
                ]
            blocks.append(block)
        return blocks

    def _obj_literal_lines(self) -> list[str]:
        """Render the ``<Obj> { .. }`` literal moving the builder's fields in.

        Defaulted fields move straight from the builder; the rest bind the
        like-named locals that :meth:`_unwrap_lines` just checked (on derived
        types ``base`` binds the local :meth:`_base_resolve_lines` produced).
        """
        bf = self.base_field
        base_entry = f"    {bf}: self.{bf}," if self.is_root else f"    {bf},"
        lines = [f"{self.obj_struct} {{", base_entry]
        # Entries bind by name; memory order just mirrors the struct definition.
        for field in _layout_fields(self.info.fields):
            fid = _rust_ident(field.name)
            if field.default is MISSING:
                lines.append(f"    {fid},")  # the unwrapped local
            else:
                lines.append(f"    {fid}: self.{fid},")
        lines.append("}")
        return lines

    def _base_resolve_lines(self) -> list[str]:
        """``let base = ..`` resolving a derived builder's base (empty for roots).

        An unset ``base`` falls back to the parent's all-default builder. Its
        error is re-contextualized: the parent's bare "field `x` is not set"
        would point at a field this type does not have.
        """
        if self.is_root:
            return []
        # The parent's ref type drives the default-construction fallback; record
        # its `use` (crate::-rooted) so a cross-module parent resolves. Non-root
        # always has a parent key (root is parent None / `ffi.Object`).
        parent_key = self.info.parent_type_key
        assert parent_key is not None
        parent_ref = self.imports.record(parent_key)
        bf = self.base_field
        return [
            f"let {bf} = match self.{bf} {{",
            f"    Some({bf}) => {bf},",
            f"    None => {parent_ref}::ffi_new().build_obj().map_err(|e| tvm_ffi::Error::new(",
            "        tvm_ffi::VALUE_ERROR,",
            f'        &format!("field `{bf}` is not set and default `{parent_ref}` '
            'construction failed: {}", e.message()),',
            '        "",',
            "    ))?,",
            "};",
        ]

    def _unwrap_lines(self) -> list[str]:
        """``let <f> = self.<f>.ok_or_else(..)?;`` for every field without a default."""
        return [
            f"let {_rust_ident(field.name)} = self.{_rust_ident(field.name)}.ok_or_else(|| "
            f'tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, "field `{field.name}` is not set", ""))?;'
            for field in _layout_fields(self.info.fields)
            if field.default is MISSING
        ]

    def _new_fn_native(self) -> list[str]:
        """Emit ``fn ffi_new() -> <T>Builder``, opening the builder chain.

        Uniformly nullary: every input -- own fields and a derived type's
        ``base`` alike -- is set through its like-named builder setter.
        Defaulted fields start prefilled with their stubgen-rendered default,
        the rest start unset and ``build()`` errors on any still missing (an
        unset ``base`` is default-constructed through the parent's builder
        instead; see :meth:`_base_resolve_lines`). Named ``ffi_new`` (not
        ``new``); a user who needs the faithful C++ constructor semantics
        hand-writes ``new`` (outside the markers) delegating to the builder.
        """
        builder = f"{self.leaf}Builder"
        lines = [f"pub fn ffi_new() -> {builder} {{", f"    {builder} {{"]
        if self.is_root:
            lines.append(f"        {self.base_field}: {self.base_type}::new(),")
        else:
            lines.append(f"        {self.base_field}: None,")
        for field in _layout_fields(self.info.fields):
            fid = _rust_ident(field.name)
            if field.default is MISSING:
                lines.append(f"        {fid}: None,")
            else:
                # `_native_blocker` already guaranteed the default renders.
                lines.append(f"        {fid}: {_default_expr(field)},")
        lines += ["    }", "}"]
        return lines

    def _builder_lines(self) -> list[str]:
        """Emit ``pub struct <T>Builder`` + its ``impl`` (setters, ``build``, ``build_obj``).

        One consuming setter per own field, plus ``base`` on derived types
        (stored ``Option<ParentObj>``; left unset it is default-constructed
        through the parent's builder at build time). Defaulted fields are
        stored prefilled; fields without a default are stored as ``Option<T>``
        and checked by ``build_obj``, which returns ``Err`` when one is still
        unset. ``build_obj`` is public -- it returns the bare struct value a
        derived type's ``base`` setter takes -- and ``build`` delegates to it,
        wrapping the struct in the allocated ref type.
        """
        builder = f"{self.leaf}Builder"
        fields = _layout_fields(self.info.fields)
        bf = self.base_field
        base_store = self.base_type if self.is_root else f"Option<{self.base_type}>"
        lines = [f"pub struct {builder} {{", f"    {bf}: {base_store},"]
        for field in fields:
            ty = self.render_struct_field(field)
            store = ty if field.default is not MISSING else f"Option<{ty}>"
            lines.append(f"    {_rust_ident(field.name)}: {store},")
        lines += ["}", ""]

        inner: list[str] = []
        if not self.is_root:
            inner += [
                f"pub fn {bf}(mut self, {bf}: {self.base_type}) -> Self {{",
                f"    self.{bf} = Some({bf});",
                "    self",
                "}",
                "",
            ]
        for field in fields:
            ty = self.render_struct_field(field)
            fid = _rust_ident(field.name)
            value = fid if field.default is not MISSING else f"Some({fid})"
            inner += [
                f"pub fn {fid}(mut self, {fid}: {ty}) -> Self {{",
                f"    self.{fid} = {value};",
                "    self",
                "}",
                "",
            ]
        self.imports.record("tvm_ffi::Result")
        prelude = [*self._base_resolve_lines(), *self._unwrap_lines()]
        literal = self._obj_literal_lines()
        inner += [
            f"pub fn build(self) -> Result<{self.leaf}> {{",
            f"    Ok({self.leaf} {{",
            "        data: ObjectArc::new(self.build_obj()?),",
            "    })",
            "}",
            "",
            f"pub fn build_obj(self) -> Result<{self.obj_struct}> {{",
            *[f"    {line}" for line in prelude],
            f"    Ok({literal[0]}",
            *[f"    {line}" for line in literal[1:-1]],
            f"    {literal[-1]})",
            "}",
        ]
        lines += [
            f"impl {builder} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]
        return lines

    def _cached_getter_lines(self, fvar: str, ffi_name: str) -> list[str]:
        """Body lines binding ``fvar`` to the reflected method, cached per call site.

        A ``thread_local!`` ``OnceCell`` makes the crate's method-table scan run
        once per thread (``Function`` is not ``Sync``, ruling out a ``OnceLock``).
        """
        cell = fvar.upper()
        return [
            f"    thread_local!(static {cell}: std::cell::OnceCell<tvm_ffi::Function> = "
            "const { std::cell::OnceCell::new() });",
            f"    let {fvar} = tvm_ffi::Function::from_type_method_cached(&{cell}, "
            f'{self.obj_struct}::type_index(), "{ffi_name}")?;',
        ]

    def _method_fn(self, method: FuncInfo) -> list[str]:
        """Emit one reflected method (instance or static) on `impl <T>`."""
        ffi_name = method.schema.name.rsplit(".", 1)[-1]
        args = method.schema.args or ()
        # The return type stays owning (a top-level `Any` is `Any`, not `AnyView`).
        ret = render_rust_type(args[0], self._ty_render) if args else self._ty_render("Any")
        rest = args[2:] if method.is_member else args[1:]
        params = [(f"_{i}", self.render_param(p)) for i, p in enumerate(rest)]

        self_recv = "&mut self" if self.info.mutable else "&self"
        if method.is_member:
            sig_parts = [self_recv, *[f"{n}: {t}" for n, t in params]]
        else:
            sig_parts = [f"{n}: {t}" for n, t in params]
        self.imports.record("tvm_ffi::Result")
        if method.is_member or params:
            self.imports.record("tvm_ffi::AnyView")
        packed = _packed_args_expr(params, method.is_member)
        getter = self._cached_getter_lines("f", ffi_name)  # raw: the FFI method name
        header = f"pub fn {_rust_ident(ffi_name)}({', '.join(sig_parts)}) -> Result<{ret}> {{"
        return [header, *_packed_call_lines("f", getter, packed, ret), "}"]


def _make_renderer(
    obj_info: ObjectInfo,
    imports: RustImports,
    ty_map: dict[str, str],
    check_refs: bool = True,
) -> _ObjectRenderer:
    """Build the per-object renderer (resolves leaf / ``<T>Obj`` / base / root-ness)."""
    type_key = obj_info.type_key
    assert isinstance(type_key, str)
    leaf = type_key.rsplit(".", 1)[-1]
    parent_key = obj_info.parent_type_key
    is_root = parent_key in (None, "ffi.Object")
    base_type = "Object" if is_root else f"{parent_key.rsplit('.', 1)[-1]}Obj"  # type: ignore[union-attr]
    return _ObjectRenderer(
        info=obj_info,
        leaf=leaf,
        obj_struct=f"{leaf}Obj",
        base_type=base_type,
        is_root=is_root,
        imports=imports,
        ty_map=ty_map,
        check_refs=check_refs,
    )


def _renders(obj_info: ObjectInfo, ty_map: dict[str, str]) -> bool:
    """Whether ``obj_info``'s own fields/methods all have a Rust rendering.

    Dry-renders the body into a throwaway collector (stdout suppressed, so the
    real generation pass owns the warnings); a raised :class:`UnsupportedTypeError`
    means an own field/arg/return type is unsupported -- i.e. the type would be
    skipped. Reference checks are disabled (``check_refs=False``) so the probe
    asks only "do this type's own field/arg/return *origins* render" without
    recursing into referenced types -- which would otherwise loop back here on a
    cyclic type reference. Does not consider ancestors (:func:`_base_buildable`).
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _make_renderer(obj_info, RustImports(), ty_map, check_refs=False).body()
    except UnsupportedTypeError:
        return False
    return True


#: Per-(type_key) cache of ``(own_renders, parent_key)`` records, scoped to the
#: current ``(registry resolver, ty_map)`` so a monkeypatched resolver (tests) or
#: a different ``ty_map`` invalidates it. Both are reference-INDEPENDENT (own-body
#: renderability + the static parent link), so caching is always safe.
_BUILDABLE_RECORDS: dict = {"key": None, "data": {}}


def _type_record(type_key: str, ty_map: dict[str, str]) -> tuple[bool, str | None] | None:
    """Return the cached ``(own_renders, parent_key)`` record (``None`` if unresolvable).

    ``own_renders`` is whether the type's OWN body renders -- with #0 field
    degradation, so it is False only when a field is genuinely unrepresentable
    (a non-pointer ``Dict`` / ``List`` / ``Union`` / ``tuple`` / scalar that has
    no opaque fallback), not merely because it references a skipped type.
    """
    cache_key = (object_info_from_type_key, id(ty_map))
    if _BUILDABLE_RECORDS["key"] != cache_key:
        _BUILDABLE_RECORDS["key"] = cache_key
        _BUILDABLE_RECORDS["data"] = {}
    cache = _BUILDABLE_RECORDS["data"]
    if type_key not in cache:
        try:
            info = object_info_from_type_key(type_key)
        except Exception:
            cache[type_key] = None  # unresolvable
        else:
            cache[type_key] = (_renders(info, ty_map), info.parent_type_key)
    return cache[type_key]


def _base_buildable(type_key: str | None, ty_map: dict[str, str]) -> bool:
    """Whether ``type_key`` can be generated -- its own body AND its ancestor chain.

    This is NOT reference-aware: a type that merely *references* a skipped type
    still generates, because #0 degrades the offending field to an opaque
    ``ObjectRef`` (see :meth:`_ObjectRenderer._render_opaque_field_or_raise`) --
    so the only things that block generation are an own non-pointer unrepresentable
    field (``own_renders`` False) or an unbuildable ancestor (a ``base`` embed
    cannot be made opaque -- it carries the object header). Rules: roots
    (``None`` / ``ffi.Object``) and crate builtins (mapped in ``ty_map``) are
    buildable; an unmapped ``ffi.*`` key is NOT (the crate has no Rust type for
    it); an unresolvable key is lenient-buildable (only skip on *provable*
    unrenderability). The recursion is over the parent chain only (a tree), so it
    terminates without cycle handling.

    Consequence: ``_base_buildable(X)`` is True iff ``generate_rust_object(X)``
    succeeds -- so a typed field reference always points at a generated type (no
    dangling), and a reference to a genuinely-skipped type is degraded by the
    field renderer.
    """
    if type_key in (None, "ffi.Object") or type_key in ty_map:
        return True
    if type_key.startswith("ffi."):
        return False  # unmapped ffi.* builtin: the crate has no Rust type for it
    record = _type_record(type_key, ty_map)
    if record is None:
        return True  # cannot prove unbuildable -> do not skip
    own_renders, parent_key = record
    return own_renders and _base_buildable(parent_key, ty_map)


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit a Rust ``struct``/``impl`` binding for an ``object/<key>`` block.

    Emits ``<T>Obj`` (``#[repr(C)]``, parent embedded as ``base``), the ``<T>``
    ref wrapper, ``Deref``/``DerefMut``, ``impl <T>`` with ``ffi_new`` plus the
    reflected methods, and the ``<T>Builder`` (when natively constructible).
    Raises :class:`UnsupportedTypeError` for types the crate cannot represent;
    ``cli`` catches it and skips the block (any ``use``s already recorded are
    harmless -- generated files allow unused imports).
    """
    assert len(code.lines) >= 2
    type_key = obj_info.type_key
    assert isinstance(type_key, str)
    parent_key = obj_info.parent_type_key
    # A derived type embeds `base: <Parent>Obj`; if the parent (or any ancestor)
    # is itself skipped, that `Obj` is never emitted, so the descendant would not
    # compile. Skip the descendant transitively with a clear message instead.
    if parent_key not in (None, "ffi.Object") and not _base_buildable(parent_key, ty_map):
        raise UnsupportedTypeError(
            parent_key,
            f"base type {parent_key!r} (or one of its ancestors) cannot be "
            "generated, so this type is skipped too",
        )

    body = _make_renderer(obj_info, imports, ty_map).body()

    _warn_offset_mismatch(type_key, _layout_fields(obj_info.fields))

    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust object layout needs no `opt`


# --- import section (`use` statements) --------------------------------------


def generate_rust_import_section(
    code: CodeBlock,
    imports: RustImports,
    opt: Options,
    defined_types: set[str],
) -> None:
    """Render the collected ``use`` statements into an ``import-section`` block.

    Imports for types defined in this same file are dropped; the rest are
    deduped and sorted.
    """
    assert len(code.lines) >= 2
    # A type defined in this file owns BOTH its ref name (in `defined_types`) and
    # its `<Name>Obj` value type, whose path is the ref path + "Obj". Filter
    # self-imports of either, so an embedded same-file base (`base: <Parent>Obj`)
    # is not re-imported -- which would collide with the local definition.
    defined = defined_types | {name + "Obj" for name in defined_types}
    # `record` never admits bare types, so every `as_use_line()` is non-empty.
    use_lines = sorted({item.as_use_line() for item in imports.items if item.path not in defined})
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in use_lines],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust needs no indent/TYPE_CHECKING handling


# --- whole-file scaffolding (`--init` mode) ---------------------------------


def generate_rust_api_file(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: InitConfig,
    is_root: bool,
    syntax: C.MarkerSyntax,
) -> str:
    """Scaffold a single Rust binding file (one file per module prefix)."""
    append = ""
    if not code_blocks:
        append += "#![allow(dead_code, unused_imports)]\n"
        append += f"\n//! FFI bindings for `{module_name}` (generated by tvm-ffi-stubgen).\n\n"
    if not any(c.kind == "import-section" for c in code_blocks):
        append += f"{syntax.begin} import-section\n{syntax.end}\n\n"
    defined = {c.param for c in code_blocks if c.kind == "object"}
    for info in object_infos:
        type_key = info.type_key
        if type_key is None or type_key in defined:
            continue
        append += f"{syntax.begin} object/{type_key}\n{syntax.end}\n\n"
    _ = (ty_map, init_cfg, is_root)  # unused for the Rust single-file layout
    return append


# --- module-tree stitching (auto-form `pub mod` declarations) ----------------


def finalize_rust_module_tree(init_path: Path, prefixes: set[str]) -> None:
    """Stitch the generated tree under ``init_path`` into a valid Rust module tree.

    Ensures every generated prefix is declared via ``pub mod`` in its parent's
    ``mod.rs``, creating intermediate ``mod.rs`` files as needed; declarations
    are appended only when absent. The user still mounts ``init_path`` with one
    ``mod`` line at the crate root (stubgen does not edit ``lib.rs``/``main.rs``).
    """
    children: dict[Path, set[str]] = {}
    for prefix in prefixes:
        segs = [s for s in prefix.split(".") if s]
        for i, seg in enumerate(segs):
            parent = init_path.joinpath(*segs[:i])
            children.setdefault(parent, set()).add(seg)

    for parent, names in children.items():
        parent.mkdir(parents=True, exist_ok=True)
        mod_rs = parent / "mod.rs"
        existing = mod_rs.read_text(encoding="utf-8") if mod_rs.exists() else ""
        to_add = [f"pub mod {n};" for n in sorted(names) if f"pub mod {n};" not in existing]
        if not to_add:
            continue
        text = existing
        if text and not text.endswith("\n"):
            text += "\n"
        if text.strip():  # separate from any existing bindings
            text += "\n"
        text += "\n".join(to_add) + "\n"
        mod_rs.write_text(text, encoding="utf-8")
