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

import ctypes
import dataclasses
import re
from typing import TYPE_CHECKING

from .. import consts as C
from ..lib_state import object_info_from_type_key
from . import consts as C_RUST
from .utils import (
    RustImports,
    UnsupportedTypeError,
    _deref_impl,
    _element_rust_type,
    _escape_ident,
    _packed_args_expr,
    _packed_call_lines,
    render_rust_type,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, NamedTypeSchema, ObjectInfo, Options


# --- reflected field ABI helpers ---------------------------------------------


_OBJECT_TYPE_INDEX_BEGIN = 64
_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)
_RUST_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CAMEL_WORD_BOUNDARY_RE = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_ACRONYM_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_RUST_APACHE_LICENSE = """// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License."""


class _TVMFFIAnyData(ctypes.Union):
    _fields_ = [("v_int64", ctypes.c_int64), ("v_ptr", ctypes.c_void_p)]  # noqa: RUF012


class _TVMFFIAny(ctypes.Structure):
    _fields_ = [
        ("type_index", ctypes.c_int32),
        ("small_str_len", ctypes.c_uint32),
        ("data", _TVMFFIAnyData),
    ]


class _DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


_ANY_ABI = (ctypes.sizeof(_TVMFFIAny), ctypes.alignment(_TVMFFIAny))
_POINTER_ABI = (ctypes.sizeof(ctypes.c_void_p), ctypes.alignment(ctypes.c_void_p))
_DEVICE_ABI = (ctypes.sizeof(_DLDevice), ctypes.alignment(_DLDevice))
_DTYPE_ABI = (ctypes.sizeof(_DLDataType), ctypes.alignment(_DLDataType))
_POINTER_VALUE_ORIGINS = frozenset(
    {
        "Array",
        "Callable",
        "Map",
        "Object",
        "Shape",
        "Tensor",
        "ffi.Error",
        "ffi.Function",
        "ffi.Module",
        "ffi.Object",
        "ffi.Shape",
        "ffi.Tensor",
    }
)


def _snake_case_global_ident(name: str) -> str:
    """Normalize one reflected global-function leaf to a Rust identifier.

    Registry names are preserved separately for lookup. This conversion only
    affects the Rust-facing free-function name; punctuation and non-ASCII
    identifiers are rejected instead of being silently rewritten.
    """
    name = _CAMEL_WORD_BOUNDARY_RE.sub(r"\1_\2", name)
    name = _CAMEL_ACRONYM_BOUNDARY_RE.sub(r"\1_\2", name).lower()
    if not _RUST_IDENT_RE.fullmatch(name):
        raise UnsupportedTypeError(
            name, f"global function leaf {name!r} cannot be normalized to a Rust identifier"
        )
    return _escape_ident(name)


def _snake_case_method_ident(name: str) -> str:
    """Normalize a reflected method name without changing its FFI lookup key."""
    name = _CAMEL_WORD_BOUNDARY_RE.sub(r"\1_\2", name)
    name = _CAMEL_ACRONYM_BOUNDARY_RE.sub(r"\1_\2", name).lower()
    if not _RUST_IDENT_RE.fullmatch(name):
        raise UnsupportedTypeError(
            name, f"reflected method {name!r} cannot be normalized to a Rust identifier"
        )
    return _escape_ident(name)


def _canonical_ident(name: str) -> str:
    """Return the identifier namespace spelling (raw ``r#`` is not distinct)."""
    return name.removeprefix("r#")


def _rust_string_literal(value: str) -> str:
    """Quote an arbitrary registry name as a Rust UTF-8 string literal."""
    escaped: list[str] = []
    for char in value:
        if char == "\\":
            escaped.append("\\\\")
        elif char == '"':
            escaped.append('\\"')
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            escaped.append(f"\\u{{{ord(char):x}}}")
        else:
            escaped.append(char)
    return f'"{"".join(escaped)}"'


def _is_pointer_object_schema(schema: TypeSchema) -> bool:
    """Whether ``schema`` has a pointer-backed Rust object-reference value.

    Registered object types use one owning pointer unless their canonical Rust
    value is the inline ``String``/``Bytes`` cell. The latter also have object
    type indices, but cannot be used as an 8-byte nullable field mirror.
    """
    return schema.origin_type_index >= _OBJECT_TYPE_INDEX_BEGIN and schema.origin not in (
        "str",
        "bytes",
    )


def _is_nullable_object_ref_field(field: NamedTypeSchema) -> bool:
    """Whether ``field`` is a raw nullable ObjectRef, not ``ffi::Optional``.

    Both are represented as ``Optional<T>`` in the language-level schema. The
    reflected field width preserves the ABI distinction: a raw ObjectRef is one
    pointer, while ``ffi::Optional<T>`` is a 16-byte ``TVMFFIAny`` cell.
    """
    return (
        field.origin == "Optional"
        and field.size == _POINTER_SIZE
        and len(field.args) == 1
        and _is_pointer_object_schema(field.args[0])
    )


def _is_exact_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _uses_default_abi_mapping(origin: str, ty_map: dict[str, str]) -> bool:
    """Whether a mapped leaf still names the Rust carrier whose ABI we know."""
    default = C_RUST.RUST_TY_MAP_DEFAULTS.get(origin)
    if default is not None:
        return ty_map.get(origin, default) == default
    return origin not in ty_map


def _integer_rust_type(field: NamedTypeSchema) -> str | None:
    """Return the exact iN/uN carrier only with explicit signedness evidence."""
    signed = getattr(field, "signed", None)
    if type(signed) is not bool:
        return None
    return C_RUST.RUST_INT_BY_SIGNED_SIZE.get((signed, field.size))


def _direct_field_abi(  # noqa: PLR0911
    field: NamedTypeSchema,
    ty_map: dict[str, str],
) -> tuple[int, int] | None:
    """Known Rust ABI for one direct field, or ``None`` when it is not proven.

    This deliberately recognizes carriers, not merely language-level schemas.
    For example, ``Optional`` may describe either ``ffi::Optional<T>`` or a
    ``std::optional<T>`` field after schema normalization, so it remains opaque
    until reflection publishes carrier-kind evidence.
    """
    origin = field.origin
    if not _uses_default_abi_mapping(origin, ty_map):
        return None
    if origin == "int":
        rust_type = _integer_rust_type(field)
        if rust_type is None:
            return None
        c_type = {
            1: ctypes.c_int8,
            2: ctypes.c_int16,
            4: ctypes.c_int32,
            8: ctypes.c_int64,
        }[field.size]
        return ctypes.sizeof(c_type), ctypes.alignment(c_type)
    if origin == "float":
        c_type = {4: ctypes.c_float, 8: ctypes.c_double}.get(field.size)
        if c_type is None:
            return None
        return ctypes.sizeof(c_type), ctypes.alignment(c_type)
    if origin == "bool":
        return ctypes.sizeof(ctypes.c_bool), ctypes.alignment(ctypes.c_bool)
    if origin in ("Any", "str", "bytes", "ffi.String", "ffi.Bytes"):
        return _ANY_ABI
    if origin in ("Device",):
        return _DEVICE_ABI
    if origin in ("dtype", "DataType"):
        return _DTYPE_ABI
    if origin == "Optional":
        # A raw nullable ObjectRef is exactly one pointer. The generated ref is
        # repr(transparent) over ObjectArc, which is repr(transparent) over
        # NonNull; Option uses its null niche. Generated const assertions still
        # verify the concrete Rust compiler's size/alignment before use. The
        # 16-byte ffi::Optional carrier remains opaque because normalized
        # schema metadata cannot yet distinguish it from std::optional.
        return _POINTER_ABI if _is_nullable_object_ref_field(field) else None
    if origin in _POINTER_VALUE_ORIGINS:
        return _POINTER_ABI
    if "." in origin and _is_pointer_object_schema(field):
        # An unmapped registered object key renders as the generated
        # repr(transparent) ObjectRef wrapper.
        return _POINTER_ABI
    return None


@dataclasses.dataclass(frozen=True)
class _DirectLayout:
    """A complete, validated direct layout for one generated object."""

    fields: tuple[NamedTypeSchema, ...]
    padding_before: tuple[int, ...]
    tail_padding: int
    total_size: int
    alignment: int


@dataclasses.dataclass(frozen=True)
class _ConstructorPlan:
    """One unambiguous reflected constructor calling convention.

    ``keyword_names is None`` identifies an explicit ``refl::init`` method,
    whose Callable schema is positional.  An auto-generated type-attribute
    initializer carries the original reflected field names and is packed
    entirely through the native KWARGS protocol.
    """

    params: tuple[tuple[str, str], ...]
    keyword_names: tuple[str, ...] | None


def _own_layout_blocker(  # noqa: PLR0911
    info: ObjectInfo, ty_map: dict[str, str]
) -> str | None:
    """Return why ``info`` itself lacks direct-layout proof."""
    if not info.has_native_layout_metadata or not _is_exact_positive_int(info.native_total_size):
        return "native total size is missing or invalid"
    if not info.has_native_alignment_metadata or not _is_exact_positive_int(info.native_alignment):
        return "native alignment is missing or invalid"
    if not info.has_mutability_metadata or info.mutable:
        return "mutability is missing or not explicitly immutable"
    assert info.native_total_size is not None
    assert info.native_alignment is not None
    if info.native_alignment & (info.native_alignment - 1):
        return "native alignment is not a power of two"
    if info.native_total_size % info.native_alignment:
        return "native total size is not a multiple of native alignment"
    for field in info.fields:
        if (
            not all(_is_exact_positive_int(value) for value in (field.size, field.alignment))
            or type(field.offset) is not int
            or field.offset < 0
        ):
            return f"field {field.name!r} has incomplete or invalid layout metadata"
        expected = _direct_field_abi(field, ty_map)
        if expected != (field.size, field.alignment):
            return f"field {field.name!r} has no proven Rust ABI carrier"
    return None


def _field_placement_blocker(info: ObjectInfo) -> str | None:
    """Validate that own fields fit after the exact parent prefix without overlap."""
    parent_size = 0 if info.parent_type_key is None else info.parent_native_total_size
    if type(parent_size) is not int:
        return None  # the ancestry check below reports missing parent metadata
    cursor = parent_size
    assert info.native_total_size is not None
    for field in sorted(info.fields, key=lambda item: item.offset):
        assert field.offset is not None
        assert field.size is not None
        assert field.alignment is not None
        if field.offset < cursor:
            return "reflected fields overlap or reuse parent tail padding"
        if field.offset % field.alignment:
            return f"field {field.name!r} is misaligned"
        cursor = field.offset + field.size
        if cursor > info.native_total_size:
            return f"field {field.name!r} exceeds native total size"
    return None


def _native_chain_blocker(  # noqa: PLR0911
    info: ObjectInfo,
    ty_map: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> str | None:
    """Validate direct-layout evidence recursively through the complete ancestry."""
    type_key = info.type_key
    if not isinstance(type_key, str):
        return "type key is missing"
    if type_key in seen:
        raise UnsupportedTypeError(type_key, f"cyclic object inheritance at {type_key!r}")
    if blocker := _own_layout_blocker(info, ty_map):
        return f"{type_key}: {blocker}"
    if blocker := _field_placement_blocker(info):
        return f"{type_key}: {blocker}"
    parent_key = info.parent_type_key
    if parent_key is None:
        return None
    if not info.parent_has_native_layout_metadata or not _is_exact_positive_int(
        info.parent_native_total_size
    ):
        return f"{type_key}: parent native total size is missing or invalid"
    if not info.parent_has_native_alignment_metadata or not _is_exact_positive_int(
        info.parent_native_alignment
    ):
        return f"{type_key}: parent native alignment is missing or invalid"
    try:
        parent = object_info_from_type_key(parent_key)
    except Exception as err:  # registry absence is lack of proof, not an object skip
        return f"{type_key}: cannot resolve parent layout ({type(err).__name__})"
    if parent.type_key != parent_key:
        raise UnsupportedTypeError(
            parent_key,
            f"resolved parent key {parent.type_key!r} does not match {parent_key!r}",
        )
    if (
        parent.native_total_size != info.parent_native_total_size
        or parent.native_alignment != info.parent_native_alignment
    ):
        return f"{type_key}: parent layout evidence disagrees with the registry"
    return _native_chain_blocker(parent, ty_map, seen | {type_key})


def _plan_direct_layout(info: ObjectInfo, ty_map: dict[str, str]) -> _DirectLayout | None:
    """Plan exact padding only when the complete immutable native chain is proven."""
    if _native_chain_blocker(info, ty_map) is not None:
        return None
    assert info.native_total_size is not None
    assert info.native_alignment is not None
    parent_size = 0 if info.parent_type_key is None else info.parent_native_total_size
    assert parent_size is not None
    fields = tuple(sorted(info.fields, key=lambda field: field.offset))
    cursor = parent_size
    padding: list[int] = []
    for field in fields:
        assert field.offset is not None
        assert field.size is not None
        assert field.alignment is not None
        if field.offset < cursor:
            # Covers both field overlap and C++ tail-padding reuse in the base.
            return None
        if field.offset % field.alignment:
            return None
        end = field.offset + field.size
        if end > info.native_total_size:
            return None
        padding.append(field.offset - cursor)
        cursor = end
    if cursor > info.native_total_size:
        return None
    return _DirectLayout(
        fields=fields,
        padding_before=tuple(padding),
        tail_padding=info.native_total_size - cursor,
        total_size=info.native_total_size,
        alignment=info.native_alignment,
    )


class _RustTypeRenderer:
    """Shared Rust type/path resolution for object and global bindings."""

    imports: RustImports
    ty_map: dict[str, str]
    mod_segments: tuple[str, ...]

    def _ty_render(self, origin: str) -> str:
        """Resolve a leaf origin to its Rust name and record its ``use``.

        Unmapped dotted names (object type keys) resolve against the generated
        module tree via :meth:`_generated_type_path`. An unmapped bare origin
        (e.g. ``const char*``) or a ``ctypes.*`` sentinel raises instead of
        emitting an invalid or invented Rust type.
        """
        mapped = self.ty_map.get(origin)
        if mapped is None:
            if "." not in origin or origin.startswith("ctypes."):
                raise UnsupportedTypeError(origin)
            mapped = self._generated_type_path(origin)
        return self.imports.record(mapped)

    def _generated_type_path(self, type_key: str) -> str:
        """Resolve a generated-tree type key to a path valid from this file."""
        head, _, _ = type_key.partition(".")
        if head in C_RUST.RUST_MOD_MAP:
            return type_key
        mod, _, type_leaf = type_key.rpartition(".")
        if tuple(mod.split(".")) == self.mod_segments:
            return type_leaf
        supers = "super::" * len(self.mod_segments)
        return f"{supers or 'self::'}{type_key.replace('.', '::')}"

    def render_param(self, schema: TypeSchema) -> str:
        """Render an argument type (top-level ``Any`` is borrowed)."""
        if schema.origin == "Any":
            return f"{self.imports.record('tvm_ffi::AnyView')}<'_>"
        return render_rust_type(schema, self._ty_render)


@dataclasses.dataclass
class _ObjectRenderer(_RustTypeRenderer):
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
    #: Module segments of the file this object lands in (its type key minus the
    #: leaf; ``tirx.transform.X`` -> ``("tirx", "transform")``): one file per
    #: prefix, mounted at ``<out>/<seg>/.../mod.rs`` (see ``cli`` and
    #: :func:`finalize_rust_module_tree`).
    mod_segments: tuple[str, ...]
    #: In-scope name of the parent's REF type (set by :meth:`_resolve_parent`
    #: for derived types; unused for roots, whose ``base`` is the bare
    #: ``tvm_ffi::Object`` data struct with no generated ref).
    parent_ref: str = ""
    #: Name of the synthesized parent-embed slot. ``base`` normally; ``__base``
    #: when a REFLECTED field is itself named ``base`` (e.g. ``tirx.Ramp``'s
    #: base expression) -- E0124 otherwise. C++ reserves ``__``-prefixed
    #: identifiers, so no reflected field can collide with the renamed slot.
    base_slot: str = "base"

    def render_struct_field(self, schema: NamedTypeSchema) -> str:
        """Render a directly-laid-out struct field type, width-correct for scalars.

        An ``int32_t`` field must render as ``i32``, not the schema-erased
        default ``i64``; the width comes from reflection's per-field ``size``.
        ``Optional`` fields are layout-sensitive: a raw nullable ObjectRef uses
        pointer-sized ``Option<T>``, while ``ffi::Optional<T>`` uses its 16-byte
        in-place mirror. Non-scalar origins (or schemas without a size) render
        plainly.
        """
        if schema.origin == "Optional":
            return self._render_optional_field(schema)
        if schema.origin == "int":
            narrowed = _integer_rust_type(schema)
            if narrowed is None:
                raise UnsupportedTypeError(
                    schema.name,
                    f"integer field {schema.name!r} lacks exact width/signedness metadata",
                )
            return narrowed
        if schema.origin == "float":
            narrowed = C_RUST.RUST_FLOAT_BY_SIZE.get(schema.size)
            if narrowed is None:
                raise UnsupportedTypeError(
                    schema.name, f"float field {schema.name!r} lacks an exact 4/8-byte width"
                )
            return narrowed
        return render_rust_type(schema, self._ty_render)

    def render_getter_type(self, schema: NamedTypeSchema) -> str:
        """Render an owning semantic field value returned by reflection.

        Unlike a direct physical carrier, an opaque getter always receives a
        normalized owning ``Any``. Optional fields therefore return ``Option``;
        integer width is narrowed only when signedness is explicit. Older or
        foreign registries without signedness fall back locally to owning
        ``Any`` rather than guessing that an unsigned field is ``i64``.
        """
        if schema.origin == "int":
            narrowed = _integer_rust_type(schema)
            if narrowed is not None:
                return narrowed
            raise UnsupportedTypeError(
                schema.name,
                f"integer field {schema.name!r} lacks exact width/signedness metadata",
            )
        if schema.origin == "float":
            if schema.size is None:
                return "f64"
            narrowed = C_RUST.RUST_FLOAT_BY_SIZE.get(schema.size)
            if narrowed is None:
                raise UnsupportedTypeError(
                    schema.name, f"float field {schema.name!r} has an invalid reflected width"
                )
            return narrowed
        return render_rust_type(schema, self._ty_render)

    def _render_optional_field(self, schema: NamedTypeSchema) -> str:
        """Render an ``Optional<T>`` FIELD according to its reflected carrier.

        A nullable C++ ObjectRef is an 8-byte owning pointer and renders as
        niche-optimized Rust ``Option<T>``. This is distinct from C++
        ``ffi::Optional<T>``, which is uniformly a 16-byte ``TVMFFIAny`` and
        renders as ``tvm_ffi::Optional<T>``.

        C++ ``ffi::Optional<T>`` is uniformly a single 16-byte ``TVMFFIAny`` for
        every storage-enabled ``T`` -- including ``Optional<ObjectRef>`` (the
        #657 ABI: layout independent of ``T``, ``nullopt == kTVMFFINone``). There
        is no pointer-sized niche form. The payload rules are exactly the
        container-element rules (``Any`` -> ``AnyValue``); the size guard
        rejects the ``std::optional`` fallback layout of storage-disabled types.
        """
        (payload,) = schema.args or (None,)  # Optional always has exactly one argument
        assert payload is not None
        if _is_nullable_object_ref_field(schema):
            return render_rust_type(schema, self._ty_render)
        payload_ty = _element_rust_type(payload, self._ty_render)
        if schema.size not in (None, C_RUST.RUST_OPTIONAL_FIELD_SIZE):
            raise UnsupportedTypeError(
                "Optional",
                f"`Optional<{payload.origin}>` field has size {schema.size}, not the "
                f"{C_RUST.RUST_OPTIONAL_FIELD_SIZE}-byte `TVMFFIAny`-backed "
                "`ffi::Optional` layout",
            )
        # No width recovery: the Any cell stores the widened v_int64/v_float64,
        # so the schema-erased scalar is the correct mirror for every C++ width.
        opt = self.imports.record(C_RUST.RUST_OPTIONAL_PATH)
        return f"{opt}<{payload_ty}>"

    def _resolve_parent(self) -> None:
        """Bring BOTH parent names into scope: ``<Parent>Obj`` and ``<Parent>``.

        The embedded ``base`` field and the ``Deref`` target need the parent's
        data struct; the upcast ``From`` target needs its ref type. Both are
        items of the parent's OWN module, so each resolves through the same
        generated-tree path rule as any cross-module type reference
        (``use super::ir::Attrs;`` + ``use super::ir::AttrsObj;``); a same-module
        parent stays a bare local name with no ``use``.
        """
        parent_key = self.info.parent_type_key
        assert isinstance(parent_key, str)  # non-root implies a parent key
        mod, dot, parent_leaf = parent_key.rpartition(".")
        self.parent_ref = self.imports.record(self._generated_type_path(parent_key))
        self.base_type = self.imports.record(
            self._generated_type_path(f"{mod}{dot}{parent_leaf}Obj")
        )

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
            self._resolve_parent()
        self.base_slot = self._plan_base_slot()
        constructor = self._constructor_plan()
        planned_methods = self._planned_methods(constructor is not None)
        getter_names = self._plan_getter_names(
            {_canonical_ident(name) for _, name in planned_methods}
            | {"same_as", "downcast"}
            | ({"ffi_new"} if constructor is not None else set())
        )

        # Direct rendering may discover that a Rust carrier cannot be named in
        # this module (for example an import collision). Roll back those imports
        # and render the same object opaquely instead.
        direct_layout = _plan_direct_layout(self.info, self.ty_map)
        direct_imports = list(self.imports.items)
        if direct_layout is not None:
            try:
                lines = self._direct_struct_lines(direct_layout)
            except UnsupportedTypeError:
                self.imports.items[:] = direct_imports
                lines = self._opaque_struct_lines()
        else:
            lines = self._opaque_struct_lines()

        lines += self._getter_impl_lines(getter_names)
        lines += self._member_impl_block(planned_methods)

        lines += [
            "#[repr(transparent)]",
            "#[derive(tvm_ffi::derive::ObjectRef, Clone)]",
            f"pub struct {self.leaf} {{",
            f"    data: ObjectArc<{self.obj_struct}>,",
            "}",
            "",
        ]

        lines += _deref_impl(self.leaf, self.obj_struct, "data")
        if not self.is_root:
            lines += _deref_impl(self.obj_struct, self.base_type, self.base_slot)
            lines += self._upcast_lines()

        # A generated binding is a foreign C++ layout mirror, not a Rust-owned
        # object definition. Construction is therefore available only through
        # the canonical reflected constructor; never allocate the mirror with
        # `ObjectArc::new` or synthesize field/default semantics in Rust.
        lines += self._impl_block(constructor, planned_methods)

        lines.pop()  # every section above ends with a `""` separator
        return lines

    def _object_struct_header(self, repr_attr: str) -> list[str]:
        return [
            repr_attr,
            "#[derive(tvm_ffi::derive::Object)]",
            f"#[type_key = {_rust_string_literal(self.info.type_key or '')}]",
            f"pub struct {self.obj_struct} {{",
            f"    {self.base_slot}: {self.base_type},",
        ]

    def _plan_base_slot(self) -> str:
        """Choose an internal parent-prefix name disjoint from reflected fields."""
        occupied = {_canonical_ident(_escape_ident(field.name)) for field in self.info.fields}
        for candidate in ("base", "__base", "__tvm_ffi_base"):
            if candidate not in occupied:
                return candidate
        suffix = 2
        while f"__tvm_ffi_base_{suffix}" in occupied:
            suffix += 1
        return f"__tvm_ffi_base_{suffix}"

    def _direct_struct_lines(self, layout: _DirectLayout) -> list[str]:
        """Render a proven foreign layout with explicit uninitialized padding."""
        lines = self._object_struct_header(f"#[repr(C, align({layout.alignment}))]")
        rendered: list[tuple[NamedTypeSchema, str, str]] = []
        occupied = {
            self.base_slot,
            *(_canonical_ident(_escape_ident(field.name)) for field in layout.fields),
        }
        pad_index = 0
        if any(layout.padding_before) or layout.tail_padding:
            self.imports.record("std::mem::MaybeUninit")
        for padding, field in zip(layout.padding_before, layout.fields, strict=True):
            if padding:
                padding_name = self._padding_name(pad_index, occupied)
                lines.append(f"    {padding_name}: MaybeUninit<[u8; {padding}]>,")
                pad_index += 1
            field_name = _escape_ident(field.name)
            field_type = self.render_struct_field(field)
            lines.append(f"    pub {field_name}: {field_type},")
            rendered.append((field, field_name, field_type))
        if layout.tail_padding:
            padding_name = self._padding_name(pad_index, occupied)
            lines.append(f"    {padding_name}: MaybeUninit<[u8; {layout.tail_padding}]>,")
        lines += ["}", ""]

        lines += [
            "const _: () = {",
            f"    assert!(std::mem::size_of::<{self.obj_struct}>() == {layout.total_size});",
            f"    assert!(std::mem::align_of::<{self.obj_struct}>() == {layout.alignment});",
            f"    assert!(std::mem::offset_of!({self.obj_struct}, {self.base_slot}) == 0);",
        ]
        for field, field_name, field_type in rendered:
            lines += [
                f"    assert!(std::mem::offset_of!({self.obj_struct}, {field_name}) "
                f"== {field.offset});",
                f"    assert!(std::mem::size_of::<{field_type}>() == {field.size});",
                f"    assert!(std::mem::align_of::<{field_type}>() == {field.alignment});",
            ]
        lines += ["};", ""]
        return lines

    @staticmethod
    def _padding_name(index: int, occupied: set[str]) -> str:
        """Allocate one deterministic internal padding name without collisions."""
        stem = f"__tvm_ffi_padding_{index}"
        candidate = stem
        suffix = 2
        while candidate in occupied:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        occupied.add(candidate)
        return candidate

    def _opaque_struct_lines(self) -> list[str]:
        """Render only the sound offset-zero inheritance prefix."""
        lines = self._object_struct_header("#[repr(C)]")
        if not self.info.has_mutability_metadata or self.info.mutable:
            # Shared C++ mutability must not acquire Rust's automatic Send/Sync
            # merely because the opaque prefix itself contains only pointers.
            lines.append("    __tvm_ffi_not_send_sync: std::marker::PhantomData<std::rc::Rc<()>>,")
        lines += ["}", ""]
        return lines

    def _planned_methods(
        self,
        has_ffi_new: bool,
    ) -> list[tuple[FuncInfo, str]]:
        """Normalize method names and reject duplicate/unusable Rust APIs early."""
        reserved = {"same_as", "downcast"} | ({"ffi_new"} if has_ffi_new else set())
        claimed: dict[str, str] = {name: f"generated helper {name!r}" for name in reserved}
        planned: list[tuple[FuncInfo, str]] = []
        for method in self.info.methods:
            ffi_name = method.schema.name.rsplit(".", 1)[-1]
            if ffi_name == "__ffi_init__":
                continue
            rust_name = _snake_case_method_ident(ffi_name)
            canonical = _canonical_ident(rust_name)
            if previous := claimed.get(canonical):
                raise UnsupportedTypeError(
                    ffi_name,
                    f"reflected method {ffi_name!r} normalizes to Rust identifier "
                    f"{rust_name!r}, already claimed by {previous}",
                )
            claimed[canonical] = f"reflected method {ffi_name!r}"
            planned.append((method, rust_name))
        return planned

    def _plan_getter_names(self, occupied: set[str]) -> list[str]:
        """Allocate stable getter names without silent method shadowing."""
        planned: list[str] = []
        for field in self.info.fields:
            preferred = _canonical_ident(_snake_case_method_ident(field.name))
            candidates = [preferred, f"get_{preferred}", f"get_{preferred}_field"]
            suffix = 2
            while all(candidate in occupied for candidate in candidates):
                candidates.append(f"get_{preferred}_field_{suffix}")
                suffix += 1
            chosen = next(candidate for candidate in candidates if candidate not in occupied)
            occupied.add(chosen)
            planned.append(_escape_ident(chosen))
        return planned

    def _getter_impl_lines(self, getter_names: list[str]) -> list[str]:
        """Emit owning reflection getters in registration-index order."""
        if not self.info.fields:
            return []
        sections: list[list[str]] = []
        for field_index, (field, getter_name) in enumerate(
            zip(self.info.fields, getter_names, strict=True)
        ):
            before = list(self.imports.items)
            fallback_any = False
            try:
                ret = self.render_getter_type(field)
            except UnsupportedTypeError:
                self.imports.items[:] = before
                ret = self.imports.record("tvm_ffi::Any")
                fallback_any = True
            result = self.imports.record("tvm_ffi::Result")
            header = f"pub fn {getter_name}(&self) -> {result}<{ret}> {{"
            call = f"tvm_ffi::object::get_reflected_field(self, {field_index})"
            if field.origin == "Any" or fallback_any:
                body = call
            else:
                body = f"Ok({call}?.try_into()?)"
            sections.append([header, f"    {body}", "}"])

        inner: list[str] = []
        for index, section in enumerate(sections):
            if index:
                inner.append("")
            inner += section
        return [
            f"impl {self.obj_struct} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _member_impl_block(self, methods: list[tuple[FuncInfo, str]]) -> list[str]:
        """Place instance methods on Obj so derived refs inherit them by Deref."""
        sections = [
            self._method_fn(method, rust_name) for method, rust_name in methods if method.is_member
        ]
        if not sections:
            return []
        inner: list[str] = []
        for index, section in enumerate(sections):
            if index:
                inner.append("")
            inner += section
        return [
            f"impl {self.obj_struct} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _ref_helper_lines(self) -> list[str]:
        """`same_as` (pointer identity) and `downcast` (checked concrete retype).

        Present on every generated ref, mirroring the C++ ref-class
        `ObjectRef::same_as` and `obj.as<N>()`: pass code compares object
        identity and narrows a base handle to a concrete node. `downcast`
        returns a borrow of `N` when the dynamic object is `N` or any subtype;
        the offset-zero inheritance prefix makes that reborrow valid.
        """
        self.imports.record("tvm_ffi::ObjectRefCore")
        return [
            "/// C++ `ObjectRef::same_as`: pointer identity of the underlying object.",
            "pub fn same_as<O: tvm_ffi::ObjectRefCore>(&self, other: &O) -> bool {",
            "    unsafe {",
            "        ObjectArc::as_raw(&self.data) as *const u8",
            "            == ObjectArc::as_raw(<O as tvm_ffi::ObjectRefCore>::data(other)) as *const u8",
            "    }",
            "}",
            "",
            "/// Checked downcast to a concrete object `N` (C++ `obj.as<N>()`):",
            "/// `Some(&N)` iff the dynamic object is `N` or a subtype, else `None`.",
            "pub fn downcast<N: tvm_ffi::ObjectCore>(&self) -> Option<&N> {",
            "    unsafe {",
            "        let raw = ObjectArc::as_raw(&self.data) as *const N;",
            "        let header = raw as *const tvm_ffi::tvm_ffi_sys::TVMFFIObject;",
            "        if tvm_ffi::object::is_instance_of::<N>((*header).type_index) {",
            "            Some(&*raw)",
            "        } else {",
            "            None",
            "        }",
            "    }",
            "}",
        ]

    def _constructor_plan(self) -> _ConstructorPlan | None:
        """Return the canonical reflected constructor protocol, if unique.

        An explicit ``refl::init<...>`` carries its signature as a reflected
        ``__ffi_init__`` TypeMethod. Auto-generated field initialization lives
        in the TypeAttrColumn instead, so its signature comes from
        ``init_fields`` and every value must be passed by its original field
        name. Multiple reflected constructor overloads do not define one Rust
        function signature and are therefore left ungenerated.
        """
        init_methods = [
            method
            for method in self.info.methods
            if method.schema.name.rsplit(".", 1)[-1] == "__ffi_init__"
        ]
        if len(init_methods) > 1:
            return None
        if init_methods:
            args = init_methods[0].schema.args or ()
            return _ConstructorPlan(
                params=tuple(
                    (f"_{i}", self.render_param(schema)) for i, schema in enumerate(args[1:])
                ),
                keyword_names=None,
            )
        if not self.info.has_init:
            return None

        claimed: set[str] = set()
        params: list[tuple[str, str]] = []
        keyword_names: list[str] = []
        for field in self.info.init_fields:
            if field.name in claimed:
                raise UnsupportedTypeError(
                    field.name,
                    f"auto-init field {field.name!r} occurs more than once in the "
                    "parent-to-child init chain; the keyword protocol cannot "
                    "address both fields",
                )
            claimed.add(field.name)
            if not _RUST_IDENT_RE.fullmatch(field.name):
                raise UnsupportedTypeError(
                    field.name,
                    f"auto-init field {field.name!r} is not a valid Rust identifier",
                )
            params.append((_escape_ident(field.name), self.render_param(field.schema)))
            keyword_names.append(field.name)
        return _ConstructorPlan(tuple(params), tuple(keyword_names))

    def _impl_block(
        self,
        constructor: _ConstructorPlan | None,
        methods: list[tuple[FuncInfo, str]],
    ) -> list[str]:
        """Emit `impl <T> { same_as; downcast; ffi_new; methods }`."""
        sections: list[list[str]] = [self._ref_helper_lines()]
        if constructor is not None:
            sections.append(self._new_fn_ffi(constructor))
        sections += [
            self._method_fn(method, rust_name)
            for method, rust_name in methods
            if not method.is_member
        ]

        inner: list[str] = []
        for i, section in enumerate(sections):
            if i:
                inner.append("")
            inner += section

        return [
            f"impl {self.leaf} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _upcast_lines(self) -> list[str]:
        """`impl From<Leaf> for <ParentRef>` -- offset-0 prefix retype (upcast).

        Sound because `<Leaf>Obj` embeds the parent as its offset-0 `base`, so
        the object pointer is unchanged; only the arc's static type moves
        (ownership transfers, no refcount change). Emitted for derived types
        only -- the parent's ref is the generated `<ParentLeaf>`; a root object
        has no ref-typed parent (its `base` is the bare `Object` data struct).
        """
        self.imports.record("tvm_ffi::ObjectRefCore")
        parent_ref = self.parent_ref  # in scope via `_resolve_parent`
        parent_obj = self.base_type
        return [
            f"impl From<{self.leaf}> for {parent_ref} {{",
            f"    fn from(x: {self.leaf}) -> {parent_ref} {{",
            f"        let arc = <{self.leaf} as tvm_ffi::ObjectRefCore>::into_data(x);",
            "        let up = unsafe {",
            f"            ObjectArc::from_raw(ObjectArc::into_raw(arc) as *const {parent_obj})",
            "        };",
            f"        <{parent_ref} as tvm_ffi::ObjectRefCore>::from_data(up)",
            "    }",
            "}",
            "",
        ]

    @staticmethod
    def _fresh_internal_ident(stem: str, occupied: set[str]) -> str:
        """Return a deterministic generated local disjoint from parameters."""
        candidate = stem
        suffix = 2
        while candidate in occupied:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        occupied.add(candidate)
        return candidate

    def _new_fn_ffi(self, constructor: _ConstructorPlan) -> list[str]:
        """Emit ``fn ffi_new(<init args>) -> Result<T>`` through C++ ``__ffi_init__``.

        Explicit TypeMethods use their positional Callable schema. Attr-only
        auto-init uses original field names through the KWARGS sentinel, which
        preserves native ``kw_only`` and default-field ordering. This is the
        only generated construction path: a foreign mirror is never allocated
        or initialized natively by Rust.
        """
        self.imports.record("tvm_ffi::Result")
        params = list(constructor.params)
        if params:
            self.imports.record("tvm_ffi::AnyView")
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        header = f"pub fn ffi_new({sig}) -> Result<{self.leaf}> {{"

        # Explicit `refl::init` has a real reflected Callable signature.
        if constructor.keyword_names is None:
            packed = _packed_args_expr(params, False)
            getter = self._cached_getter_lines("f", "__ffi_init__")
            return [header, *_packed_call_lines("f", getter, packed, self.leaf), "}"]

        # An auto initializer with no init=True fields still needs to run so
        # native defaults and creator-initialized fields are applied.
        if not params:
            getter = self._cached_getter_lines("f", "__ffi_init__")
            return [header, *_packed_call_lines("f", getter, "", self.leaf), "}"]

        any_view = self.imports.record("tvm_ffi::AnyView")
        occupied = {_canonical_ident(name) for name, _ in params}
        init_var = self._fresh_internal_ident("__tvm_ffi_init", occupied)
        init_cell = self._fresh_internal_ident("__TVM_FFI_INIT", occupied)
        kwargs_fn = self._fresh_internal_ident("__tvm_ffi_get_kwargs", occupied)
        kwargs_cell = self._fresh_internal_ident("__TVM_FFI_KWARGS", occupied)
        kwargs = self._fresh_internal_ident("__tvm_ffi_kwargs", occupied)
        keys = self._fresh_internal_ident("__tvm_ffi_keys", occupied)
        args = self._fresh_internal_ident("__tvm_ffi_args", occupied)

        lines = [
            header,
            *self._cached_getter_lines(
                init_var,
                "__ffi_init__",
                cell=init_cell,
            ),
            f"    static {kwargs_cell}: std::sync::OnceLock<tvm_ffi::Function> = "
            "std::sync::OnceLock::new();",
            f"    let {kwargs_fn} = tvm_ffi::Function::get_global_cached("
            f'&{kwargs_cell}, "ffi.GetKwargsObject")?;',
            f"    let {kwargs} = {kwargs_fn}.call_packed(&[])?;",
            f"    let {keys} = [",
            *[
                f"        tvm_ffi::String::from({_rust_string_literal(name)}),"
                for name in constructor.keyword_names
            ],
            "    ];",
            f"    let {args} = [",
            f"        {any_view}::from(&{kwargs}),",
        ]
        for index, (name, ty) in enumerate(params):
            lines.append(f"        {any_view}::from(&{keys}[{index}]),")
            value = name if ty == f"{any_view}<'_>" else f"{any_view}::from(&{name})"
            lines.append(f"        {value},")
        lines += [
            "    ];",
            f"    Ok({init_var}.call_packed(&{args})?.try_into()?)",
            "}",
        ]
        return lines

    def _cached_getter_lines(
        self,
        fvar: str,
        ffi_name: str,
        *,
        cell: str | None = None,
    ) -> list[str]:
        """Body lines binding ``fvar`` to the reflected method, cached per call site.

        A process-wide ``OnceLock`` makes the reflected method-table scan run
        once per generated call site.
        """
        cell = cell or fvar.upper()
        return [
            f"    static {cell}: std::sync::OnceLock<tvm_ffi::Function> = "
            "std::sync::OnceLock::new();",
            f"    let {fvar} = tvm_ffi::Function::from_type_method_cached(&{cell}, "
            f'{self.obj_struct}::type_index(), "{ffi_name}")?;',
        ]

    def _method_fn(self, method: FuncInfo, rust_name: str) -> list[str]:
        """Emit one reflected method (instance or static) on `impl <T>`."""
        ffi_name = method.schema.name.rsplit(".", 1)[-1]
        args = method.schema.args or ()
        # The return type stays owning (a top-level `Any` is `Any`, not `AnyView`).
        ret = render_rust_type(args[0], self._ty_render) if args else self._ty_render("Any")
        rest = args[2:] if method.is_member else args[1:]
        params = [(f"_{i}", self.render_param(p)) for i, p in enumerate(rest)]

        if method.is_member:
            # A shared owning handle never proves unique access to the foreign
            # allocation. Mutable C++ methods use their own interior-mutation
            # contract behind FFI, so the Rust receiver remains shared.
            sig_parts = ["&self", *[f"{n}: {t}" for n, t in params]]
        else:
            sig_parts = [f"{n}: {t}" for n, t in params]
        self.imports.record("tvm_ffi::Result")
        if method.is_member or params:
            self.imports.record("tvm_ffi::AnyView")
        packed = _packed_args_expr(params, method.is_member)
        # The FFI lookup string keeps the reflected name while the Rust-facing
        # identifier is normalized during the all-method planning pass.
        getter = self._cached_getter_lines("f", ffi_name)
        header = f"pub fn {rust_name}({', '.join(sig_parts)}) -> Result<{ret}> {{"
        return [header, *_packed_call_lines("f", getter, packed, ret), "}"]


@dataclasses.dataclass
class _GlobalRenderer(_RustTypeRenderer):
    """Render schema-driven free functions for one ``global/<prefix>`` block."""

    imports: RustImports
    ty_map: dict[str, str]
    mod_segments: tuple[str, ...]

    @staticmethod
    def _cached_getter_lines(fvar: str, ffi_name: str) -> list[str]:
        """Bind one fallible, successful-lookup-only process-wide global cache."""
        cell = fvar.upper()
        return [
            f"    static {cell}: std::sync::OnceLock<tvm_ffi::Function> = "
            "std::sync::OnceLock::new();",
            f"    let {fvar} = tvm_ffi::Function::get_global_cached(&{cell}, "
            f"{_rust_string_literal(ffi_name)})?;",
        ]

    def _packed_fallback(self, ffi_name: str, fn_name: str) -> list[str]:
        """Render ``Callable[..., Any]`` without inventing arity or types."""
        result = self.imports.record("tvm_ffi::Result")
        any_ty = self.imports.record("tvm_ffi::Any")
        any_view = self.imports.record("tvm_ffi::AnyView")
        getter = self._cached_getter_lines("f", ffi_name)
        return [
            f"pub fn {fn_name}(args: &[{any_view}<'_>]) -> {result}<{any_ty}> {{",
            *getter,
            "    f.call_packed(args)",
            "}",
        ]

    def function(self, func: FuncInfo, fn_name: str) -> list[str]:
        """Render one typed global or an honest fully-packed fallback."""
        schema = func.schema
        ffi_name = schema.name
        if schema.origin != "Callable":
            raise UnsupportedTypeError(
                schema.origin,
                f"global function {ffi_name!r} has non-Callable schema {schema.origin!r}",
            )
        if not schema.args:
            return self._packed_fallback(ffi_name, fn_name)

        ret_schema, *param_schemas = schema.args
        result = self.imports.record("tvm_ffi::Result")
        if ret_schema.origin == "Any":
            ret = self.imports.record("tvm_ffi::Any")
        else:
            ret = render_rust_type(ret_schema, self._ty_render)

        params: list[tuple[str, str, bool]] = []
        if param_schemas:
            any_view = self.imports.record("tvm_ffi::AnyView")
            for i, param_schema in enumerate(param_schemas):
                param_ty = (
                    f"{any_view}<'_>"
                    if param_schema.origin == "Any"
                    else render_rust_type(param_schema, self._ty_render)
                )
                params.append((f"_{i}", param_ty, param_schema.origin == "Any"))

        signature = ", ".join(f"{name}: {ty}" for name, ty, _ in params)
        packed = ", ".join(
            name if is_any else f"AnyView::from(&{name})" for name, _, is_any in params
        )
        getter = self._cached_getter_lines("f", ffi_name)
        call = f"f.call_packed(&[{packed}])"
        if ret_schema.origin != "Any":
            call = f"Ok({call}?.try_into()?)"
        return [
            f"pub fn {fn_name}({signature}) -> {result}<{ret}> {{",
            *getter,
            f"    {call}",
            "}",
        ]


def generate_rust_global_funcs(
    code: CodeBlock,
    global_funcs: list[FuncInfo],
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
) -> None:
    """Generate fallible Rust wrappers for one global-function prefix.

    A non-empty ``Callable`` argument tuple is a complete schema: element zero
    is the return and the remainder are parameters. Bare ``Callable()`` means
    ``Callable[..., Any]`` and therefore emits a packed-slice fallback. Name
    and type validation is transactional: neither the block nor ``imports`` is
    changed unless every function can be rendered without collisions.
    """
    assert len(code.lines) >= 2
    if not global_funcs:
        code.lines = [code.lines[0], code.lines[-1]]
        return
    assert isinstance(code.param, tuple)
    prefix, _ = code.param

    planned: list[tuple[FuncInfo, str]] = []
    claimed: dict[str, str] = {}
    for func in sorted(global_funcs, key=lambda item: item.schema.name):
        ffi_name = func.schema.name
        fn_name = _snake_case_global_ident(ffi_name.rsplit(".", 1)[-1])
        if previous := claimed.get(fn_name):
            raise UnsupportedTypeError(
                fn_name,
                f"global functions {previous!r} and {ffi_name!r} both normalize "
                f"to Rust identifier {fn_name!r}",
            )
        claimed[fn_name] = ffi_name
        planned.append((func, fn_name))

    scratch_imports = RustImports(items=list(imports.items))
    renderer = _GlobalRenderer(
        imports=scratch_imports,
        ty_map=ty_map,
        mod_segments=tuple(segment for segment in prefix.split(".") if segment),
    )
    body: list[str] = []
    for func, fn_name in planned:
        if body:
            body.append("")
        body.extend(renderer.function(func, fn_name))

    imports.items[:] = scratch_imports.items
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust formatting is fixed


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit a Rust ``struct``/``impl`` binding for an ``object/<key>`` block.

    Every object has an offset-zero parent prefix and owning reflected getters.
    Exact fields are exposed directly only when the complete native ancestry
    and every carrier are proven; otherwise the object remains opaque without
    disappearing from the generated API. A canonical reflected
    ``__ffi_init__`` additionally produces ``ffi_new``. Impossible inheritance
    or Rust namespaces still raise :class:`UnsupportedTypeError`, and both the
    block and import collector remain unchanged on such a failure.
    """
    assert len(code.lines) >= 2
    type_key = obj_info.type_key
    assert isinstance(type_key, str)
    leaf = type_key.rsplit(".", 1)[-1]
    obj_struct = f"{leaf}Obj"
    scratch_imports = RustImports(items=list(imports.items))
    renderer = _ObjectRenderer(
        info=obj_info,
        leaf=leaf,
        obj_struct=obj_struct,
        base_type="",  # resolved by `body()` (crate `Object` / `_resolve_parent`)
        is_root=obj_info.parent_type_key in (None, "ffi.Object"),
        imports=scratch_imports,
        ty_map=ty_map,
        mod_segments=tuple(type_key.split(".")[:-1]),
    )

    body = renderer.body()

    imports.items[:] = scratch_imports.items
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
        {item.as_use_line() for item in imports.items if item.path not in defined_types}
    )
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
        append += _RUST_APACHE_LICENSE + "\n\n"
        append += "#![allow(dead_code, unused_imports)]\n"
        append += f"\n//! FFI bindings for `{module_name}` (generated by tvm-ffi-stubgen).\n\n"
    if not any(c.kind == "import-section" for c in code_blocks):
        append += f"{syntax.begin} import-section\n{syntax.end}\n\n"
    if not any(c.kind == "global" for c in code_blocks):
        append += f"{syntax.begin} global/{module_name}\n{syntax.end}\n\n"
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
        existed = mod_rs.exists()
        existing = mod_rs.read_text(encoding="utf-8") if existed else ""
        to_add = [f"pub mod {n};" for n in sorted(names) if f"pub mod {n};" not in existing]
        if not to_add:
            continue
        text = existing if existed else _RUST_APACHE_LICENSE + "\n"
        if text and not text.endswith("\n"):
            text += "\n"
        if text.strip():  # separate from any existing bindings
            text += "\n"
        text += "\n".join(to_add) + "\n"
        mod_rs.write_text(text, encoding="utf-8")
