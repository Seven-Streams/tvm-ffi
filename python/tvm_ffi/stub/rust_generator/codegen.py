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
    r"""Render a Python string as a Rust string literal.

    Escapes ``\`` ``"`` ``\n`` ``\r`` ``\t``; other control characters are not
    supported and pass through verbatim.
    """
    body = "".join(_STR_ESCAPES.get(ch, ch) for ch in value)
    return f'"{body}"'


def _render_scalar_default(value: object) -> str | None:
    """Render a bool/int/float default as a Rust literal, or ``None`` otherwise.

    ``bool`` must be checked before ``int`` (Python ``bool`` subclasses ``int``).
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

    Only concrete scalar/string defaults render; anything else (factory, ``None``,
    missing) returns ``None`` and the caller falls back to the FFI ``__ffi_init__``
    path. Non-finite floats render as typed constants (``f32::INFINITY`` etc.), so
    ``render_type`` must produce the width-correct scalar type.
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

    The native ``ffi_new`` allocates the struct directly, binding fields from its
    parameters and silently bypassing any C++ constructor logic -- that is the
    opted-in behavior, so native is used whenever possible. The FFI path remains
    only when an own field is not memory-safe to write natively
    (:data:`~.consts.RUST_NATIVE_UNSAFE_ORIGINS`), a parent is itself non-native,
    or an own non-init field has no renderable default.
    """
    if not info.has_init or any(f.origin in C_RUST.RUST_NATIVE_UNSAFE_ORIGINS for f in info.fields):
        return False
    parent = info.parent_type_key
    if parent not in (None, "ffi.Object") and not _native_eligible(parent):
        return False
    # `fields` and `own_field_inits` share a source; a missing key is a bug.
    schema_of = {f.name: f for f in info.fields}
    init_names = {f.name for f in info.init_fields}
    for fi in info.own_field_inits:
        if fi.name in init_names:
            continue  # own init field -> constructor parameter
        if _render_default(fi, schema_of[fi.name], lambda _: "T") is None:
            return False
    return True


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
            f"construction ({type(e).__name__}: {e}); using the FFI constructor"
            f"{C.TERM_RESET}"
        )
        return False
    return _info_native_eligible(info)


def _has_native_child(type_key: str) -> bool:
    """Whether any registered type derives from ``type_key`` and is itself native.

    Only such parents need the bare-struct ``<Self>Obj::ffi_new`` builder -- it
    is the only way a child's ``base:`` argument can be produced.
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
    end. Reflection has no ``alignof``, so alignment is approximated from
    ``size`` (largest power of two, capped at 8) -- exact for scalars, but
    composite FFI structs like ``DLDevice`` can trigger a false positive. A
    mismatch only warns; the binding is still emitted. Fields without
    offset/size metadata are skipped and reset the running position.
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

    Holds the per-object rendering context (imports, ``ty_map``, resolved
    names, mutability) so helper methods don't have to thread it through.
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
        """Resolve a leaf origin to its Rust name and record its ``use``.

        Unmapped dotted names (object type keys) pass through; an unmapped bare
        origin (e.g. ``const char*``) has no Rust rendering and raises, which
        skips the enclosing object.
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

        An ``int32_t`` field must render as ``i32``, not the schema-erased
        default ``i64``; the width comes from reflection's per-field ``size``.
        Non-scalar origins (or schemas without a size) fall through to
        :meth:`render_field`.
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
        # Boilerplate `use`s, recorded through the same collector as field types
        # so leaf collisions raise and skip the object. The derive imports need
        # aliases: their leaves collide with `tvm_ffi::Object`/`ObjectRef`.
        self.imports.record("std::ops::Deref")
        # `ObjectCore` must be in scope for the generated `type_index()` calls.
        self.imports.record("tvm_ffi::ObjectCore")
        self.imports.record("tvm_ffi::ObjectArc")
        self.imports.record("tvm_ffi::derive::Object", alias="DeriveObject")
        self.imports.record("tvm_ffi::derive::ObjectRef", alias="DeriveObjectRef")
        if self.is_root:
            # Same path the ty_map uses for `Object` fields, so they dedup
            # instead of colliding.
            self.base_type = self.imports.record("tvm_ffi::Object")
        if self.mutable:
            self.imports.record("std::ops::DerefMut")

        leaf, obj_struct, base_type = self.leaf, self.obj_struct, self.base_type
        lines: list[str] = []
        # TYPE_KEY / `type_index` / `object_header_mut` all come from the
        # `#[derive(Object)]` proc macro.
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

        # Native (FFI-free) construction whenever the whole chain is eligible;
        # otherwise dispatch the reflected `__ffi_init__`. The bare-struct
        # builder is only needed by types with native children.
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

        Unlike the FFI path, ancestor init fields are not flattened in: the
        caller passes an already-built ``base: <Parent>Obj`` (from the parent's
        ``<Parent>Obj::ffi_new``); a root type omits it.
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
        """Render the inline ``<Obj> { .. }`` struct literal, as source lines.

        A root bottoms out at ``Object::new()``; a derived type binds the
        in-scope ``base`` parameter. Init fields bind from the like-named
        ``ffi_new`` parameters; non-init fields take their rendered defaults.
        """
        lines = [f"{self.obj_struct} {{"]
        if self.is_root:
            lines.append(f"    base: {self.base_type}::new(),")
        else:
            lines.append("    base,")  # shorthand: the `base` parameter
        init_names = {f.name for f in self.info.init_fields}
        field_inits = {f.name: f for f in self.info.own_field_inits}
        # Entries bind by name; memory order just mirrors the struct definition.
        for field in _layout_fields(self.info.fields):
            if field.name in init_names:
                lines.append(f"    {field.name},")  # shorthand: param name == field name
            else:
                # `render_struct_field`: a non-finite default (`f32::INFINITY`)
                # must match the width-narrowed field type.
                default = _render_default(field_inits[field.name], field, self.render_struct_field)
                lines.append(f"    {field.name}: {default},")
        lines.append("}")
        return lines

    def _obj_new_fn_native(self) -> list[str]:
        """Emit ``impl <T>Obj { pub fn ffi_new(..) -> Self }`` -- the bare-struct builder.

        Builds the struct value only (no allocation); emitted only for types
        with native children, whose ``ffi_new`` takes this output as ``base``.
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

        No FFI round-trip: the struct literal goes straight to `ObjectArc::new`.
        The `Result` is kept for signature parity with the FFI path. Named
        ``ffi_new`` (not ``new``) because it bypasses any C++ constructor logic;
        a user who needs faithful semantics hand-writes ``new`` (outside the
        markers) delegating to it.
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

        A ``thread_local!`` ``OnceCell`` makes the method-table scan run once
        per thread (``Function`` is not ``Sync``, ruling out a ``OnceLock``).
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

    Emits ``<T>Obj`` (``#[repr(C)]``, parent embedded as ``base``), the ``<T>``
    ref wrapper, ``Deref``/``DerefMut``, and ``impl <T>`` with ``ffi_new`` plus
    the reflected methods. Raises :class:`UnsupportedTypeError` for types the
    crate cannot represent; ``cli`` catches it and skips the block (any ``use``s
    already recorded are harmless -- generated files allow unused imports).
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

    Imports for types defined in this same file are dropped; the rest are
    deduped and sorted.
    """
    assert len(code.lines) >= 2
    # `record` never admits bare types, so every `as_use_line()` is non-empty.
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

#: Shared per-file helpers, written fully-qualified (zero `use`s) so they never
#: clash with the import-section block. `get_type_method` scans the type's
#: method table; `get_type_method_cached` fronts it with a per-call-site
#: `thread_local!` `OnceCell`.
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
    """Scaffold a single Rust binding file (one file per module prefix).

    Only a genuinely new file gets the ``#![allow(...)]`` header: an inner
    attribute appended to a pre-existing file would not compile. Missing
    ``helpers`` / ``import-section`` / ``object/<type_key>`` marker blocks are
    appended; keeping the helpers in a marker means they are regenerated on
    every run, not just for brand-new files.
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
