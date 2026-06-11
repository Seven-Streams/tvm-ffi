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

This module owns Rust codegen orchestration. Low-level rendering helpers live in
``rust_generator.utils`` so the block-generation pipeline here stays focused on
directive handling and source assembly.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Callable

from tvm_ffi._ffi_api import GetRegisteredTypeKeys

from .. import consts as C
from ..lib_state import object_info_from_type_key
from . import consts as C_RUST
from .utils import (
    RustImports,
    UnsupportedTypeError,
    _class_is_mutable,
    _deref_impl,
    _packed_args_expr,
    _packed_call_lines,
    render_rust_type,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FieldInit, FuncInfo, InitConfig, NamedTypeSchema, ObjectInfo, Options


# --- native (FFI-free) construction eligibility & default rendering ----------


_STR_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _rust_str_lit(value: str) -> str:
    r"""Render a Python string as a double-quoted Rust string literal.

    Only the escapes a reasonable default needs (``\`` ``"`` ``\n`` ``\r``
    ``\t``) are handled; other control characters are passed through verbatim
    (a default containing them is not supported).
    """
    body = "".join(_STR_ESCAPES.get(ch, ch) for ch in value)
    return f'"{body}"'


def _render_scalar_default(value: object) -> str | None:
    """Render a bool/int/float default as a Rust literal, or ``None`` otherwise.

    Bare int/float/bool literals rely on struct-literal type inference (no width
    suffix needed). ``bool`` must precede ``int`` (Python ``bool`` is an ``int``
    subclass).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return repr(int(value))
    if isinstance(value, float):
        text = repr(float(value))  # non-finite floats are handled by the caller
        if not any(c in text for c in ".eE"):  # ensure a float literal (e.g. `1` -> `1.0`)
            text += ".0"
        return text
    return None


def _render_default(
    fi: FieldInit,
    schema: NamedTypeSchema,
    render_type: Callable[[NamedTypeSchema], str],
) -> str | None:
    """Render a field's default as a Rust expression, or ``None`` if not renderable.

    Only *trivially constructible* defaults are renderable: a concrete scalar or
    string. A factory default (mutable container), a ``None`` default, or a
    missing/object default returns ``None`` -> the caller falls back to the FFI
    ``__ffi_init__`` path. (A ``None`` default could only fill an ``Optional``
    field, and any ``Optional`` field already forces the FFI fallback -- see
    :data:`~.consts.RUST_NATIVE_UNSAFE_ORIGINS`.)

    Non-finite floats have no bare literal (``inf`` does not compile) and render
    as the typed constants ``<ty>::INFINITY`` / ``NEG_INFINITY`` / ``NAN``, so
    ``render_type`` must produce the *width-correct* scalar type for directly
    laid-out fields (``f32`` vs ``f64`` -- i.e. ``render_struct_field``).
    """
    if fi.has_factory or not fi.has_default or fi.default is None:
        return None
    value = fi.default
    if isinstance(value, str):
        return f"{render_type(schema)}::from({_rust_str_lit(value)})"
    if isinstance(value, float) and not math.isfinite(value):
        ty = render_type(schema)
        if math.isnan(value):
            return f"{ty}::NAN"
        return f"{ty}::INFINITY" if value > 0 else f"{ty}::NEG_INFINITY"
    return _render_scalar_default(value)


def _info_native_eligible(info: ObjectInfo) -> bool:
    """Decide whether ``info`` can be constructed natively (no FFI ``__ffi_init__``).

    Both construction paths are emitted as ``ffi_new`` (the generated constructor;
    a user who wants faithful C++ semantics hand-writes ``new``, outside the
    markers, delegating to ``ffi_new``). The native path allocates the struct
    directly: a derived type's ``ffi_new`` takes the already-built parent value as
    a single ``base: <Parent>Obj`` parameter (a root type omits it) followed by
    the *own* init fields, in declaration order, with own non-init fields taking a
    rendered default. The ``param_i`` <-> ``field_i`` correspondence is **assumed**,
    not verified: a C++ constructor that validates, derives extra fields, or has
    side effects is silently bypassed -- that is the opted-in behavior.

    Native is therefore used whenever it is *possible*. It falls back to the FFI
    ``__ffi_init__`` path only when native construction genuinely cannot reproduce
    the object: an own field's top-level type is not memory-safe to write natively
    (``Optional`` / ``tuple`` -- see :data:`~.consts.RUST_NATIVE_UNSAFE_ORIGINS`);
    a parent is itself non-native (then no ``<Parent>Obj::ffi_new`` exists to
    build the ``base`` argument); or an own non-init field lacks a statically
    renderable default.
    """
    if not info.has_init or any(f.origin in C_RUST.RUST_NATIVE_UNSAFE_ORIGINS for f in info.fields):
        return False
    parent = info.parent_type_key
    if parent not in (None, "ffi.Object") and not _native_eligible(parent):
        return False
    # `fields` and `own_field_inits` are built from the same `type_info.fields`
    # list, so direct indexing is safe (a mismatch is a bug -> fail fast).
    schema_of = {f.name: f for f in info.fields}
    init_names = {f.name for f in info.init_fields}
    for fi in info.own_field_inits:
        if fi.name in init_names:
            continue  # own init field -> constructor parameter
        if _render_default(fi, schema_of[fi.name], lambda _: "T") is None:
            return False
    return True


def _native_eligible(type_key: str) -> bool:
    """Type-key keyed wrapper of :func:`_info_native_eligible` (parent recursion).

    A parent that cannot be characterized (lookup or schema parsing fails) is
    reported and treated as non-native -- the child then dispatches the FFI
    ``__ffi_init__``, which works regardless. Deliberately uncached: ancestor
    chains are short, and a cache would pin the first answer for the lifetime
    of the process (stale across registry changes, e.g. between test cases).
    """
    try:
        info = object_info_from_type_key(type_key)
    except Exception as e:  # any failure means "cannot prove native-safe"
        print(
            f"{C.TERM_YELLOW}[Warning] cannot resolve type {type_key!r} for native "
            f"construction ({type(e).__name__}: {e}); using the FFI constructor"
            f"{C.TERM_RESET}"
        )
        return False
    return _info_native_eligible(info)


def _has_native_child(type_key: str) -> bool:
    """Whether any *registered* type derives from ``type_key`` and is itself native.

    Such a child's native ``ffi_new`` takes ``base: <Self>Obj``, and the only way
    a caller can produce that bare value is this type's ``<Self>Obj::ffi_new``
    builder -- so the builder is emitted exactly for these parent types. Every
    other type (in particular an ordinary root that nobody derives from) keeps
    the builder-free shape: a single ``impl`` on the ref with the inline literal.
    """
    for child_key in GetRegisteredTypeKeys():
        try:
            child = object_info_from_type_key(child_key)
        except Exception:
            continue  # unrelated registry entries must not break this object
        if child.parent_type_key == type_key and _info_native_eligible(child):
            return True
    return False


def _layout_fields(fields: list[NamedTypeSchema]) -> list[NamedTypeSchema]:
    """Own fields in C++ memory order (reflection ``offset``), not registration order.

    Reflection stores fields in *registration* order (the ``def_field`` call
    sequence), which need not match the C++ declaration (memory) order. The
    generated ``#[repr(C)]`` struct lays fields out positionally, so it must
    emit them sorted by their recorded byte offset. Fields without an offset
    (synthetic ``ObjectInfo``s in tests) keep registration order.
    """
    if any(f.offset is None for f in fields):
        return list(fields)
    return sorted(fields, key=lambda f: f.offset)


def _warn_offset_mismatch(type_key: str | None, fields: list[NamedTypeSchema]) -> None:
    """Warn when ``#[repr(C)]`` cannot reproduce the recorded field offsets.

    Walks the offset-sorted fields and recomputes where ``#[repr(C)]`` would
    place each one: the previous field's end, rounded up to the field's
    alignment. Reflection records ``size`` but not ``alignof``, so alignment is
    approximated as the largest power of two dividing ``size``, capped at 8 --
    exact for scalars (align == size) and pointer-based renders, but it
    over-estimates composite FFI structs (``DLDevice`` is size 8 / align 4,
    ``DLDataType`` size 4 / align 2), which can yield a false-positive warning
    for a perfectly valid layout. A mismatch means the generated struct *may*
    read/write that field at the wrong address (e.g. an unregistered C++ member
    leaves a hole the Rust layout won't reproduce); the binding is still
    emitted, so only warn. A field without offset/size metadata cannot be
    checked and is skipped -- and the field after it has no known predecessor
    end, so checking resumes one field later.
    """
    prev_end: int | None = None
    for field in fields:
        if field.offset is None or field.size is None or field.size <= 0:
            prev_end = None
            continue
        if prev_end is not None:
            align = min(8, field.size & -field.size)
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

    Bundles the per-object rendering environment -- the import collector, the
    ``ty_map``, the resolved struct/ref identifiers, and the mutability shape --
    that is invariant across the whole of one object's rendering. Holding it as
    state (rather than threading it through every helper signature) keeps the
    method signatures small and makes adding new render context a one-field
    change. The type-render entry points (:meth:`render_field` /
    :meth:`render_param`) feed the stateless :func:`.utils.render_rust_type` seam.
    """

    info: ObjectInfo
    leaf: str
    obj_struct: str
    base_type: str
    is_root: bool
    mutable: bool
    imports: RustImports
    ty_map: dict[str, str]

    def _ty_render(self, origin: str) -> str:
        """Resolve a leaf origin to its Rust leaf name and record its ``use``.

        Only dotted names (object type keys like ``my_pkg.Foo``) may pass through
        unmapped; an unmapped *bare* origin (e.g. ``const char*``) has no Rust
        rendering and would otherwise be emitted verbatim as invalid source, so
        it raises and the enclosing object is skipped.
        """
        mapped = self.ty_map.get(origin)
        if mapped is None:
            if "." not in origin:
                raise UnsupportedTypeError(origin)
            mapped = origin
        return self.imports.record(mapped)

    def render_field(self, schema: TypeSchema) -> str:
        """Render a field/return type (owning form: a top-level ``Any`` is ``Any``)."""
        return render_rust_type(schema, self._ty_render)

    def render_struct_field(self, schema: NamedTypeSchema) -> str:
        """Render a directly-laid-out struct field type, width-correct for scalars.

        The ``#[repr(C)]`` struct (and the native ``ffi_new`` parameters that bind
        straight into it) accesses fields at their real C++ offsets, so an
        ``int32_t`` field must render as ``i32``, not the schema-erased default
        ``i64``. The width comes from reflection's per-field ``sizeof(T)``
        (:attr:`NamedTypeSchema.size`); schemas without one (or non-scalar
        origins) fall through to the ordinary :meth:`render_field`.
        """
        narrowed = C_RUST.RUST_SCALAR_BY_SIZE.get((schema.origin, schema.size))
        return narrowed if narrowed is not None else self.render_field(schema)

    def render_param(self, schema: TypeSchema) -> str:
        """Render an argument type (a top-level ``Any`` is the non-owning ``AnyView``)."""
        if schema.origin == "Any":
            return self.imports.record("tvm_ffi::AnyView")
        return render_rust_type(schema, self._ty_render)

    def body(self) -> list[str]:
        """Build the Rust source lines for the object (raises on unsupported types)."""
        # Boilerplate `use`s the generated items rely on, recorded through the
        # same collector as field types: a pathological user type whose leaf
        # collides with one of them raises and skips the object (see
        # `RustImports.record`). The derive macros carry fixed aliases -- their
        # leaves collide with `tvm_ffi::Object`/`ObjectRef` by construction.
        self.imports.record("std::ops::Deref")
        # `ObjectCore` only needs to be in scope so the generated
        # `<T>Obj::type_index()` trait-method calls resolve.
        self.imports.record("tvm_ffi::ObjectCore")
        self.imports.record("tvm_ffi::ObjectArc")
        self.imports.record("tvm_ffi::derive::Object", alias="DeriveObject")
        self.imports.record("tvm_ffi::derive::ObjectRef", alias="DeriveObjectRef")
        if self.is_root:
            # The crate-root re-export: the same path the ty_map uses for
            # `Object`/`ffi.Object` fields, so such a field dedups against the
            # boilerplate instead of colliding with it.
            self.base_type = self.imports.record("tvm_ffi::Object")
        if self.mutable:
            self.imports.record("std::ops::DerefMut")

        leaf, obj_struct, base_type = self.leaf, self.obj_struct, self.base_type
        lines: list[str] = []
        # The `ObjectCore` impl (TYPE_KEY / per-class static `type_index` /
        # `object_header_mut`) is folded into the `#[derive(Object)]` proc macro,
        # which derives `object_header_mut` from the first field and caches the
        # type index in a per-type static -- no shared hashmap lookup.
        lines += [
            "#[repr(C)]",
            "#[derive(DeriveObject)]",
            f'#[type_key = "{self.info.type_key}"]',
            f"pub struct {obj_struct} {{",
            f"    base: {base_type},",
        ]
        for field in _layout_fields(self.info.fields):
            lines.append(f"    pub {field.name}: {self.render_struct_field(field)},")
        lines += ["}", ""]

        lines += [
            "#[repr(C)]",
            "#[derive(DeriveObjectRef, Clone)]",
            f"pub struct {leaf} {{",
            f"    data: ObjectArc<{obj_struct}>,",
            "}",
            "",
        ]

        lines += _deref_impl(leaf, obj_struct, "data", self.mutable)
        if not self.is_root:
            lines += _deref_impl(obj_struct, base_type, "base", self.mutable)

        # Native (FFI-free) construction whenever possible (the whole inheritance
        # chain must be eligible -- see `_info_native_eligible`). Otherwise the
        # reflected `__ffi_init__` is dispatched as before. The bare-struct
        # builder is an extra that only types with native children need (their
        # `base` argument is constructible no other way); everything else keeps
        # the single-`impl`, builder-free shape.
        native = _info_native_eligible(self.info)
        if native and _has_native_child(self.info.type_key or ""):
            lines += self._obj_new_fn_native()
        lines += self._impl_block(native)

        if lines and lines[-1] == "":
            lines.pop()
        return lines

    def _impl_block(self, native: bool) -> list[str]:
        """Emit `impl <T> { new; methods }`; empty list when there's nothing to emit."""
        init_method = next(
            (m for m in self.info.methods if m.schema.name.rsplit(".", 1)[-1] == "__ffi_init__"),
            None,
        )
        methods = [
            m for m in self.info.methods if m.schema.name.rsplit(".", 1)[-1] != "__ffi_init__"
        ]
        if not self.info.has_init and not methods:
            return []

        inner: list[str] = []
        if self.info.has_init:
            inner += self._new_fn_native() if native else self._new_fn(init_method)
            if methods:
                inner.append("")
        for i, method in enumerate(methods):
            inner += self._method_fn(method)
            if i != len(methods) - 1:
                inner.append("")

        return [
            f"impl {self.leaf} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _native_params(self) -> list[tuple[str, str]]:
        """Native ``ffi_new`` parameter list: ``base`` (derived only) + own init fields.

        Unlike the FFI path (which must pass every flattened ancestor init field
        through ``__ffi_init__``), the native signature takes the already-built
        parent value as a single ``base: <Parent>Obj`` parameter -- a root type
        omits it (its base is always ``Object::new()``). The caller builds the
        ``base`` argument with the parent's own ``<Parent>Obj::ffi_new``.
        """
        params: list[tuple[str, str]] = []
        if not self.is_root:
            params.append(("base", self.base_type))
        init_names = {f.name for f in self.info.init_fields}
        params += [
            (f.name, self.render_struct_field(f)) for f in self.info.fields if f.name in init_names
        ]
        return params

    def _struct_literal_lines(self) -> list[str]:
        """Render the flat inline ``<Obj> { .. }`` struct literal for this object.

        Mirrors the crate's own native construction idiom (e.g.
        ``TensorObjFromNDAlloc { base: TensorObj { object: Object::new(), .. } }``
        in ``collections/tensor.rs``). A root bottoms out at ``Object::new()``; a
        derived type binds the in-scope ``base`` parameter (field-init shorthand)
        -- ancestors are never inlined. Own init fields bind from the in-scope
        ``ffi_new`` parameters (which carry the field names); own non-init fields
        take their rendered defaults. Returned as lines:
        ``["<Obj> {", "    base: ..", .., "}"]``.
        """
        lines = [f"{self.obj_struct} {{"]
        if self.is_root:
            # For a root, `base_type` is the in-scope name of `tvm_ffi::Object`,
            # recorded in :meth:`body`.
            lines.append(f"    base: {self.base_type}::new(),")
        else:
            lines.append("    base,")  # shorthand: the `base` parameter
        init_names = {f.name for f in self.info.init_fields}
        field_inits = {f.name: f for f in self.info.own_field_inits}
        # Struct-literal entries bind by name, so order is semantically free;
        # memory order is used to mirror the struct definition.
        for field in _layout_fields(self.info.fields):
            if field.name in init_names:
                lines.append(f"    {field.name},")  # shorthand: param name == field name
            else:
                # `render_struct_field`, not `render_field`: a non-finite float
                # default renders as a typed constant (`f32::INFINITY`), which
                # must match the width-narrowed field type.
                default = _render_default(field_inits[field.name], field, self.render_struct_field)
                lines.append(f"    {field.name}: {default},")
        lines.append("}")
        return lines

    def _obj_new_fn_native(self) -> list[str]:
        """Emit ``impl <T>Obj { pub fn ffi_new(..) -> Self }`` -- the bare-struct builder.

        Builds the struct *value* only (no allocation), with the same signature
        as the ref-level ``ffi_new``. Emitted ONLY for types some native child
        derives from (see :func:`_has_native_child`): its output is what the
        child's ``ffi_new`` takes as ``base``. Types without native children
        never get this extra ``impl``.
        """
        params = self._native_params()
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        literal = self._struct_literal_lines()
        return [
            f"impl {self.obj_struct} {{",
            f"    pub fn ffi_new({sig}) -> Self {{",
            *[f"        {line}" for line in literal],
            "    }",
            "}",
            "",
        ]

    def _new_fn_native(self) -> list[str]:
        """Emit `fn ffi_new(..) -> Result<Self>` that allocates the object natively.

        No FFI round-trip: a single inline struct literal (built by
        :meth:`_struct_literal_lines`) is handed to `ObjectArc::new`, which writes
        the object header + Rust deleter. The `Result` wrapper is kept for
        signature parity with the FFI path -- the body is infallible (`Ok(..)`).

        Named ``ffi_new`` (not ``new``): it binds fields directly and bypasses any
        C++ constructor logic (validation / derived fields / side effects). A user
        who needs the faithful semantics hand-writes ``new`` (outside the stubgen
        markers) delegating to this ``ffi_new``.
        """
        params = self._native_params()
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        literal = self._struct_literal_lines()
        self.imports.record("tvm_ffi::Result")
        return [
            f"pub fn ffi_new({sig}) -> Result<Self> {{",
            "    Ok(Self {",
            f"        data: ObjectArc::new({literal[0]}",
            *[f"        {line}" for line in literal[1:-1]],
            f"        {literal[-1]}),",
            "    })",
            "}",
        ]

    def _new_fn(self, init_method: FuncInfo | None) -> list[str]:
        """Emit `fn ffi_new(...) -> Result<Self>` calling reflected `__ffi_init__`."""
        if init_method is not None:
            arg_schemas = list(init_method.schema.args[1:]) if init_method.schema.args else []
            params = [(f"_{i}", self.render_param(s)) for i, s in enumerate(arg_schemas)]
        else:
            params = [(f.name, self.render_param(f.schema)) for f in self.info.init_fields]
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        self.imports.record("tvm_ffi::Result")
        if params:
            self.imports.record("tvm_ffi::AnyView")
        packed = _packed_args_expr(params, is_member=False)
        getter = self._cached_getter_lines("ctor", "__ffi_init__")
        return [
            f"pub fn ffi_new({sig}) -> Result<Self> {{",
            *_packed_call_lines("ctor", getter, packed, "Self"),
            "}",
        ]

    def _cached_getter_lines(self, fvar: str, ffi_name: str) -> list[str]:
        """Body lines binding ``fvar`` to the reflected method, cached per call site.

        Each generated function owns a ``thread_local!`` ``OnceCell`` so the
        method-table scan in ``get_type_method`` runs once per thread instead of
        on every call. (``Function`` is not ``Sync``, so a process-wide
        ``OnceLock`` cannot hold it.)
        """
        cell = fvar.upper()
        return [
            f"    thread_local!(static {cell}: std::cell::OnceCell<tvm_ffi::Function> = "
            "const { std::cell::OnceCell::new() });",
            f"    let {fvar} = get_type_method_cached(&{cell}, "
            f'{self.obj_struct}::type_index(), "{ffi_name}")?;',
        ]

    def _method_fn(self, method: FuncInfo) -> list[str]:
        """Emit one reflected method (instance or static) on `impl <T>`."""
        ffi_name = method.schema.name.rsplit(".", 1)[-1]
        args = method.schema.args or ()
        # Return type uses the owning render (a top-level `Any` stays `Any`).
        ret = self.render_field(args[0]) if args else self._ty_render("Any")
        rest = list(args[1:])
        if method.is_member:
            rest = rest[1:]
        params = [(f"_{i}", self.render_param(p)) for i, p in enumerate(rest)]

        self_recv = "&mut self" if self.mutable else "&self"
        if method.is_member:
            sig_parts = [self_recv, *[f"{n}: {t}" for n, t in params]]
        else:
            sig_parts = [f"{n}: {t}" for n, t in params]
        self.imports.record("tvm_ffi::Result")
        if method.is_member or params:
            self.imports.record("tvm_ffi::AnyView")
        packed = _packed_args_expr(params, method.is_member)
        getter = self._cached_getter_lines("f", ffi_name)
        header = f"pub fn {ffi_name}({', '.join(sig_parts)}) -> Result<{ret}> {{"
        return [header, *_packed_call_lines("f", getter, packed, ret), "}"]


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit a Rust ``struct``/``impl`` binding for an ``object/<key>`` block.

    Emits the standard binding shape: ``<T>Obj`` (``#[repr(C)]``,
    ``#[derive(Object)]``, parent embedded as ``base``), the ``<T>`` ref (wrapping
    ``ObjectArc``), ``Deref``/``DerefMut`` (the latter only for mutable classes),
    and ``impl <T>`` with an ``ffi_new`` constructor + reflected methods.

    Raises :class:`~..utils.UnsupportedTypeError` when the object uses a type the
    crate cannot represent; ``cli`` catches it and skips the block. A raise may
    leave already-recorded ``use``s behind in ``imports`` -- harmless, generated
    files open with ``#![allow(unused_imports)]``.
    """
    assert len(code.lines) >= 2
    type_key = obj_info.type_key
    assert isinstance(type_key, str)
    leaf = type_key.rsplit(".", 1)[-1]
    obj_struct = f"{leaf}Obj"
    parent_key = obj_info.parent_type_key
    is_root = parent_key in (None, "ffi.Object")
    if is_root:
        base_type = "Object"
    else:
        assert isinstance(parent_key, str)
        base_type = f"{parent_key.rsplit('.', 1)[-1]}Obj"
    mutable, mixed = _class_is_mutable(obj_info)

    renderer = _ObjectRenderer(
        info=obj_info,
        leaf=leaf,
        obj_struct=obj_struct,
        base_type=base_type,
        is_root=is_root,
        mutable=mutable,
        imports=imports,
        ty_map=ty_map,
    )

    body = renderer.body()

    if mixed:
        print(
            f"{C.TERM_YELLOW}[Warning] object {type_key}: mixed read-only/read-write "
            f"fields; treating the whole type as read-only{C.TERM_RESET}"
        )
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

    Imports whose target is a type *defined in this same file* are dropped
    (``RustUse.full_name in defined_types``) -- you don't ``use`` what you
    define locally. The remaining uses are deduped and sorted for a stable,
    one-per-line rendering (Rust has no ``TYPE_CHECKING`` split).
    """
    assert len(code.lines) >= 2
    # `RustImports.record` never admits a bare / no-import type into `items`, so
    # every `as_use_line()` here is a real, non-empty `use` line.
    use_lines = sorted(
        {item.as_use_line() for item in imports.items if item.full_name not in defined_types}
    )
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in use_lines],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust needs no indent/TYPE_CHECKING handling


# --- whole-file scaffolding (`--init` mode) ---------------------------------

#: Shared per-file helper functions. Written fully-qualified with
#: zero `use`s so they never clash with the import-section block (which carries
#: every object-driven `use`). `get_type_method` pulls a reflected method off the
#: type's method table; the type index is resolved by each object's
#: `#[derive(Object)]`-generated per-class static (no shared hashmap lookup).
#: `get_type_method_cached` fronts it with a per-call-site `thread_local!`
#: `OnceCell` so the linear method-table scan runs once per thread.
_RUST_HELPERS = """fn get_type_method_cached(
    cell: &'static std::thread::LocalKey<std::cell::OnceCell<tvm_ffi::Function>>,
    type_index: i32,
    method_name: &str,
) -> tvm_ffi::Result<tvm_ffi::Function> {
    cell.with(|c| {
        if let Some(f) = c.get() {
            return Ok(f.clone());
        }
        let f = get_type_method(type_index, method_name)?;
        let _ = c.set(f.clone());
        Ok(f)
    })
}

fn get_type_method(
    type_index: i32,
    method_name: &str,
) -> tvm_ffi::Result<tvm_ffi::Function> {
    unsafe {
        let info = tvm_ffi::tvm_ffi_sys::TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::TYPE_ERROR,
                &format!("no type info for type_index `{type_index}`"),
                "",
            ));
        }
        let info = &*info;
        for i in 0..info.num_methods {
            let mi = &*info.methods.add(i as usize);
            if mi.name.as_str() == method_name {
                if !<tvm_ffi::Function as tvm_ffi::type_traits::AnyCompatible>::check_any_strict(
                    &mi.method,
                ) {
                    return Err(tvm_ffi::Error::new(
                        tvm_ffi::TYPE_ERROR,
                        &format!(
                            "method `{method_name}` on type_index `{type_index}` is not a Function"
                        ),
                        "",
                    ));
                }
                return Ok(<tvm_ffi::Function as tvm_ffi::type_traits::AnyCompatible>::copy_from_any_view_after_check(&mi.method));
            }
        }
    }
    Err(tvm_ffi::Error::new(
        tvm_ffi::TYPE_ERROR,
        &format!("method `{method_name}` not found on type_index `{type_index}`"),
        "",
    ))
}"""


def generate_rust_api_file(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: InitConfig,
    is_root: bool,
    is_new_file: bool = True,
    syntax: C.MarkerSyntax = C.RUST_SYNTAX,
) -> str:
    """Scaffold a single Rust binding file (option A: one file per module prefix).

    Only a genuinely new/empty file (``is_new_file``) gets the ``#![allow(...)]``
    header: it is an inner attribute that must precede every item, so appending
    it to a pre-existing file (e.g. an intermediate ``mod.rs`` holding ``pub
    mod`` lines but no markers) would not compile. A ``helpers`` marker block
    (filled by :func:`generate_rust_helpers` during stage processing with the
    shared support functions), an ``import-section`` marker, and an
    ``object/<type_key>`` marker per registered type are added if missing. Putting
    the helpers in a marker means they are (re)generated on every run -- even into
    a pre-existing file -- rather than only when the file is brand new. No
    ``global``/``__all__``/``export`` blocks are emitted.
    """
    append = ""
    if is_new_file:
        append += "#![allow(dead_code, unused_imports)]\n"
        append += f"\n//! FFI bindings for `{module_name}` (generated by tvm-ffi-stubgen).\n\n"
    if not any(c.kind == "helpers" for c in code_blocks):
        append += f"{syntax.begin} helpers\n{syntax.end}\n\n"
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


def generate_rust_helpers(code: CodeBlock, opt: Options) -> None:
    """Fill a ``helpers`` block with the shared per-file support functions."""
    assert len(code.lines) >= 2
    indent = " " * code.indent
    body = _RUST_HELPERS.split("\n")
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt


def generate_rust_init(
    code_blocks: list[CodeBlock],
    module_name: str,
    submodule: str = "mod",
    syntax: C.MarkerSyntax = C.RUST_SYNTAX,
) -> str:
    """No init/re-export file for Rust (option A: the API file *is* the module)."""
    _ = (code_blocks, module_name, submodule, syntax)
    return ""


# --- module-tree stitching (auto-form `pub mod` declarations) ----------------


def finalize_rust_module_tree(init_path: Path, prefixes: set[str]) -> None:
    """Stitch the generated tree under ``init_path`` into a valid Rust module tree.

    For every generated prefix (e.g. ``a.b.c``) and each of its ancestor dirs,
    ensure the parent's ``mod.rs`` declares ``pub mod <child>;`` -- creating
    intermediate ``mod.rs`` files for prefixes that hold no types themselves, and
    a root ``init_path/mod.rs`` declaring the top-level modules. Declarations are
    idempotent (appended only when absent).

    The only thing left to the user is one ``mod``/``pub mod`` line at their crate
    root mounting ``init_path`` (stubgen cannot safely edit ``lib.rs``/``main.rs``).
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
