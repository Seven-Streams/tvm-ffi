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
    _rust_ident,
    _use,
    render_rust_type,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FieldInit, FuncInfo, NamedTypeSchema, ObjectInfo, Options


# --- native (FFI-free) construction eligibility & default rendering ----------


def _rust_str_lit(value: str) -> str:
    """Render a Python string as a double-quoted Rust string literal (escaped)."""
    body = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
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
    directly and binds every flattened init field (``info.init_fields``) from a
    constructor parameter, in declaration order, with own non-init fields taking a
    rendered default. The ``param_i`` <-> ``field_i`` correspondence is **assumed**,
    not verified: a C++ constructor that validates, derives extra fields, or has
    side effects is silently bypassed -- that is the opted-in behavior.

    Native is therefore used whenever it is *possible*. It falls back to the FFI
    ``__ffi_init__`` path only when native construction genuinely cannot reproduce
    the object: the type opts out (``no_native``); an own field's top-level type is
    not memory-safe to write natively (``Optional`` / ``tuple`` -- see
    :data:`~.consts.RUST_NATIVE_UNSAFE_ORIGINS`); a parent is itself non-native; or
    an own non-init field lacks a statically renderable default.
    """
    if (
        not info.has_init
        or info.no_native
        or any(f.origin in C_RUST.RUST_NATIVE_UNSAFE_ORIGINS for f in info.fields)
    ):
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
        """Resolve a leaf origin to its Rust leaf name and record its ``use``."""
        return self.imports.record(self.ty_map.get(origin, origin))

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
            _use(self.imports, "tvm_ffi::AnyView")
            return "AnyView"
        return render_rust_type(schema, self._ty_render)

    def body(self) -> list[str]:
        """Build the Rust source lines for the object (raises on unsupported types)."""
        # Boilerplate `use`s the generated items rely on. Recorded first so they
        # claim their (un-aliased) leaf names before any field/method type does.
        _use(self.imports, "std::ops::Deref")
        _use(self.imports, "tvm_ffi::object::ObjectArc")
        _use(self.imports, "tvm_ffi::object::ObjectCore")
        _use(self.imports, "tvm_ffi::derive::Object", alias="DeriveObject")
        _use(self.imports, "tvm_ffi::derive::ObjectRef", alias="DeriveObjectRef")
        if self.is_root:
            _use(self.imports, "tvm_ffi::object::Object")
        if self.mutable:
            _use(self.imports, "std::ops::DerefMut")

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
        for field in self.info.fields:
            lines.append(f"    pub {_rust_ident(field.name)}: {self.render_struct_field(field)},")
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

        lines += self._impl_block()

        if lines and lines[-1] == "":
            lines.pop()
        return lines

    def _impl_block(self) -> list[str]:
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

        _use(self.imports, "tvm_ffi::Result")

        # Native (FFI-free) construction whenever possible (the whole inheritance
        # chain must be eligible -- see `_info_native_eligible`). Otherwise the
        # reflected `__ffi_init__` is dispatched as before.
        native = _info_native_eligible(self.info)

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

    def _struct_literal_lines(self, info: ObjectInfo) -> list[str]:
        """Render the inline nested ``<Obj> { .. }`` struct literal for ``info``.

        Mirrors the crate's own native construction idiom (e.g.
        ``TensorObjFromNDAlloc { base: TensorObj { object: Object::new(), .. } }``
        in ``collections/tensor.rs``): one nested struct literal, the parent
        embedded inline as ``base`` (bottoming out at ``Object::new()`` at the
        root). Own init fields bind from the in-scope ``ffi_new`` parameters (which
        carry the field names); own non-init fields take their rendered defaults.
        Returned as lines: ``["<Obj> {", "    base: ..", .., "}"]``.
        """
        leaf = info.type_key.rsplit(".", 1)[-1]  # type: ignore[union-attr]
        obj_struct = f"{leaf}Obj"
        parent_key = info.parent_type_key

        lines = [f"{obj_struct} {{"]
        if parent_key in (None, "ffi.Object"):
            lines.append("    base: Object::new(),")
        else:
            parent = self._struct_literal_lines(object_info_from_type_key(parent_key))
            lines.append(f"    base: {parent[0]}")
            lines += [f"    {pl}" for pl in parent[1:-1]]
            lines.append(f"    {parent[-1]},")  # close the embedded parent + `,`

        init_names = {f.name for f in info.init_fields}
        field_inits = {f.name: f for f in info.own_field_inits}
        for field in info.fields:
            ident = _rust_ident(field.name)
            if field.name in init_names:
                lines.append(f"    {ident},")  # shorthand: param name == field name
            else:
                # `render_struct_field`, not `render_field`: a non-finite float
                # default renders as a typed constant (`f32::INFINITY`), which
                # must match the width-narrowed field type.
                default = _render_default(field_inits[field.name], field, self.render_struct_field)
                lines.append(f"    {ident}: {default},")
        lines.append("}")
        return lines

    def _new_fn_native(self) -> list[str]:
        """Emit `fn ffi_new(..) -> Result<Self>` that allocates the object natively.

        No FFI round-trip: a single inline nested struct literal (built by
        :meth:`_struct_literal_lines`) is handed to `ObjectArc::new`, which writes
        the object header + Rust deleter. The `Result` wrapper is kept for
        signature parity with the FFI path -- the body is infallible (`Ok(..)`).

        Named ``ffi_new`` (not ``new``): it binds fields directly and bypasses any
        C++ constructor logic (validation / derived fields / side effects). A user
        who needs the faithful semantics hand-writes ``new`` (outside the stubgen
        markers) delegating to this ``ffi_new``.
        """
        params = [
            (_rust_ident(f.name), self.render_struct_field(f.schema)) for f in self.info.init_fields
        ]
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        # `Object::new()` bottoms out the (possibly inlined-parent) literal, so the
        # `Object` import is needed for any native type, not just a root one.
        _use(self.imports, "tvm_ffi::object::Object")
        literal = self._struct_literal_lines(self.info)
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
            params = [
                (_rust_ident(f.name), self.render_param(f.schema)) for f in self.info.init_fields
            ]
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        if params:
            _use(self.imports, "tvm_ffi::AnyView")
        packed = _packed_args_expr(params, is_member=False)
        getter = (
            f'    let ctor = get_type_method({self.obj_struct}::type_index(), "__ffi_init__")?;'
        )
        return [
            f"pub fn ffi_new({sig}) -> Result<Self> {{",
            *_packed_call_lines("ctor", getter, packed, "Self"),
            "}",
        ]

    def _method_fn(self, method: FuncInfo) -> list[str]:
        """Emit one reflected method (instance or static) on `impl <T>`."""
        ffi_name = method.schema.name.rsplit(".", 1)[-1]
        rust_name = _rust_ident(ffi_name)
        args = method.schema.args or ()
        # Return type uses the owning render (a top-level `Any` stays `Any`).
        ret = self.render_field(args[0]) if args else "Any"
        rest = list(args[1:])
        if method.is_member:
            rest = rest[1:]
        params = [(f"_{i}", self.render_param(p)) for i, p in enumerate(rest)]

        self_recv = "&mut self" if self.mutable else "&self"
        if method.is_member:
            sig_parts = [self_recv, *[f"{n}: {t}" for n, t in params]]
        else:
            sig_parts = [f"{n}: {t}" for n, t in params]
        if method.is_member or params:
            _use(self.imports, "tvm_ffi::AnyView")
        packed = _packed_args_expr(params, method.is_member)
        getter = f'    let f = get_type_method({self.obj_struct}::type_index(), "{ffi_name}")?;'
        header = f"pub fn {rust_name}({', '.join(sig_parts)}) -> Result<{ret}> {{"
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
    and ``impl <T>`` with an ``ffi_new`` constructor + reflected methods. On an
    :class:`UnsupportedTypeError` the whole object is skipped with a warning.
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

    # Render into a local collector so a skip leaves `imports` untouched. Seeding
    # from `imports.items` is enough to carry forward every prior `use`: a fresh
    # `RustImports` rebuilds its collision trackers (`_binding_of`/`_used_names`)
    # from `items` in `__post_init__`. `items` is the single source of truth, so
    # the writeback below only needs to copy `items` back (see the writeback note).
    local = RustImports(items=list(imports.items))
    renderer = _ObjectRenderer(
        info=obj_info,
        leaf=leaf,
        obj_struct=obj_struct,
        base_type=base_type,
        is_root=is_root,
        mutable=mutable,
        imports=local,
        ty_map=ty_map,
    )

    try:
        body = renderer.body()
    except UnsupportedTypeError as e:
        # When the whole object is skipped, the `mixed` warning below is moot, so
        # the early return suppresses it on purpose.
        print(
            f"{C.TERM_YELLOW}[Skipped] object {type_key}: "
            f"unsupported type {e.origin!r}{C.TERM_RESET}"
        )
        code.lines = [code.lines[0], code.lines[-1]]
        return

    if mixed:
        print(
            f"{C.TERM_YELLOW}[Warning] object {type_key}: mixed read-only/read-write "
            f"fields; treating the whole type as read-only{C.TERM_RESET}"
        )

    # Writeback note: copy only `items` -- the next object block re-seeds a fresh
    # `local` from it and rebuilds the trackers, so `imports`'s own (now stale)
    # `_binding_of`/`_used_names` are never read again and need not be updated.
    imports.items[:] = local.items
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
_RUST_HELPERS = """fn get_type_method(
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
    init_cfg: object,
    is_root: bool,
    syntax: C.MarkerSyntax = C.RUST_SYNTAX,
) -> str:
    """Scaffold a single Rust binding file (option A: one file per module prefix).

    A fresh file gets the ``#![allow(...)]`` header. A ``helpers`` marker block
    (filled by :func:`generate_rust_helpers` during stage processing with the
    shared support functions), an ``import-section`` marker, and an
    ``object/<type_key>`` marker per registered type are added if missing. Putting
    the helpers in a marker means they are (re)generated on every run -- even into
    a pre-existing file -- rather than only when the file is brand new. No
    ``global``/``__all__``/``export`` blocks are emitted.
    """
    append = ""
    if not code_blocks:
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
