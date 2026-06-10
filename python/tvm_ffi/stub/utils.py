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
"""Language-agnostic data model for the `tvm-ffi-stubgen` tool.

These dataclasses describe the FFI reflection metadata (functions, object
fields/methods, init signatures) without committing to any target language.
Turning this metadata into source text is the job of a target language
generator (e.g. :mod:`tvm_ffi.stub.python_generator.codegen`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from tvm_ffi.core import MISSING, TypeInfo, TypeSchema, _lookup_type_attr

from . import consts as C


def _parse_type_schema(raw: str | dict[str, Any]) -> TypeSchema:
    """Parse a type schema from either a JSON string or an already-parsed dict."""
    if isinstance(raw, dict):
        return TypeSchema.from_json_obj(raw)
    return TypeSchema.from_json_str(raw)


@dataclasses.dataclass
class InitConfig:
    """Configuration for generating new stubs.

    Examples
    --------
    If we are generating type stubs for Python package `my-ffi-extension`,
    and the CMake target that generates the shared library is `my_ffi_extension_shared`,
    then we can run the following command to generate the stubs:

    --init-pypkg my-ffi-extension --init-lib my_ffi_extension_shared --init-prefix my_ffi_extension.

    """

    pkg: str
    """Name of the Python package to generate stubs for, e.g. apache-tvm-ffi (instead of tvm_ffi)"""

    shared_target: str
    """Name of CMake target that generates the shared library, e.g. tvm_ffi_shared

    This is used to determine the name of the shared library file.
    - macOS: lib{shared_target}.dylib or lib{shared_target}.so
    - Linux: lib{shared_target}.so
    - Windows: {shared_target}.dll
    """

    prefix: str
    """Only generate stubs for global function and objects with the given prefix, e.g. `tvm_ffi.`"""


@dataclasses.dataclass
class Options:
    """Command line options for stub generation."""

    imports: list[str] = dataclasses.field(default_factory=list)
    dlls: list[str] = dataclasses.field(default_factory=list)
    init: InitConfig | None = None
    indent: int = 4
    files: list[str] = dataclasses.field(default_factory=list)
    verbose: bool = False
    dry_run: bool = False
    target: str = "python"
    """Code generator target to use, e.g. ``"python"`` or ``"rust"``."""


@dataclasses.dataclass(init=False)
class NamedTypeSchema(TypeSchema):
    """A type schema with an associated name.

    ``frozen`` is the field's read-only flag from reflection (``def_ro`` ->
    ``True``, ``def_rw`` -> ``False``). It is only meaningful for object fields;
    method/init schemas leave it at the default ``False``. Backends that care
    about mutability (e.g. the Rust backend's mutable-vs-immutable class shape)
    read it; the Python backend ignores it.

    ``size`` is the field's in-memory byte width from reflection
    (``TVMFFIFieldInfo.size`` = ``sizeof(T)``), or ``None`` when the schema does
    not describe an object field (method args/returns carry no width). The type
    schema alone cannot distinguish e.g. ``int32_t`` from ``int64_t`` (both are
    ``{"type": "int"}``); backends that lay fields out directly (the Rust
    ``#[repr(C)]`` structs) need ``size`` to pick a width-correct type.

    ``offset`` is the field's byte offset within the object from reflection
    (``TVMFFIFieldInfo.offset``), or ``None`` outside object fields. Reflection
    stores fields in *registration* order, which need not match memory order;
    backends that lay fields out directly must order by ``offset``.
    """

    name: str
    frozen: bool = False
    size: int | None = None
    offset: int | None = None

    def __init__(
        self,
        name: str,
        schema: TypeSchema,
        frozen: bool = False,
        size: int | None = None,
        offset: int | None = None,
    ) -> None:
        """Initialize a `NamedTypeSchema` with the given name, schema and field metadata."""
        super().__init__(origin=schema.origin, args=schema.args)
        self.name = name
        self.frozen = frozen
        self.size = size
        self.offset = offset


@dataclasses.dataclass
class FuncInfo:
    """Information of a function."""

    schema: NamedTypeSchema
    is_member: bool

    @staticmethod
    def from_schema(name: str, schema: TypeSchema, *, is_member: bool = False) -> FuncInfo:
        """Construct a `FuncInfo` from a name and its type schema."""
        return FuncInfo(schema=NamedTypeSchema(name=name, schema=schema), is_member=is_member)


@dataclasses.dataclass
class InitFieldInfo:
    """A field that participates in the auto-generated ``__init__``."""

    name: str
    schema: NamedTypeSchema
    kw_only: bool
    has_default: bool


@dataclasses.dataclass
class FieldInit:
    """Per-own-field init/default metadata, used by native (FFI-free) construction.

    One entry per *own* field of a type (parent fields live on the parent). It
    records the default needed to populate a non-init field in a native Rust
    struct literal. ``default`` is the concrete value (or :data:`MISSING`);
    ``has_factory`` flags a mutable default produced by a factory function -- not
    statically renderable, so it forces the FFI fallback.
    """

    name: str
    has_default: bool
    default: Any
    has_factory: bool


@dataclasses.dataclass
class ObjectInfo:
    """Information of an object type, including its fields and methods."""

    fields: list[NamedTypeSchema]
    methods: list[FuncInfo]
    type_key: str | None = None
    parent_type_key: str | None = None
    init_fields: list[InitFieldInfo] = dataclasses.field(default_factory=list)
    has_init: bool = False
    own_field_inits: list[FieldInit] = dataclasses.field(default_factory=list)
    no_native: bool = False
    """Opt-out: the type's C++ ``__ffi_init__`` must be dispatched (no native build).

    Set by the ``__ffi_no_native__`` type attribute. Use it for types whose C++
    constructor does more than field assignment (validation, side effects,
    derived fields) -- native construction would silently skip that logic.
    """

    @staticmethod
    def from_type_info(type_info: TypeInfo) -> ObjectInfo:
        """Construct an `ObjectInfo` from a `TypeInfo` instance."""
        parent_type_key: str | None = None
        if type_info.parent_type_info is not None:
            parent_type_key = type_info.parent_type_info.type_key

        # Detect __ffi_init__ from either source: a TypeMethod (explicit C++
        # `refl::init<...>`) or a type-attr column (the auto-generated
        # field-binding init, `BindFieldArgs`).
        has_init_method = any(m.name == "__ffi_init__" for m in type_info.methods)
        has_init = has_init_method or (
            _lookup_type_attr(type_info.type_index, "__ffi_init__") is not None
        )
        # Opt-out marker: `refl::TypeAttrDef<T>().attr("__ffi_no_native__", true)`.
        no_native = bool(_lookup_type_attr(type_info.type_index, "__ffi_no_native__"))

        # Walk parent chain (parent-first) to collect all init-eligible fields.
        init_fields: list[InitFieldInfo] = []
        if has_init:
            ti: TypeInfo | None = type_info
            chain: list[TypeInfo] = []
            while ti is not None:
                chain.append(ti)
                ti = ti.parent_type_info
            for ancestor_info in reversed(chain):
                for field in ancestor_info.fields:
                    if not field.c_init:
                        continue
                    init_fields.append(
                        InitFieldInfo(
                            name=field.name,
                            schema=NamedTypeSchema(
                                name=field.name,
                                schema=_parse_type_schema(field.metadata["type_schema"]),
                                size=field.size,
                            ),
                            kw_only=field.c_kw_only,
                            has_default=field.c_has_default,
                        )
                    )

        return ObjectInfo(
            fields=[
                NamedTypeSchema(
                    name=field.name,
                    schema=_parse_type_schema(field.metadata["type_schema"]),
                    frozen=field.frozen,
                    size=field.size,
                    offset=field.offset,
                )
                for field in type_info.fields
            ],
            methods=[
                FuncInfo(
                    schema=NamedTypeSchema(
                        name=C.FN_NAME_MAP.get(method.name, method.name),
                        schema=_parse_type_schema(method.metadata["type_schema"]),
                    ),
                    is_member=not method.is_static,
                )
                for method in type_info.methods
            ],
            type_key=type_info.type_key,
            parent_type_key=parent_type_key,
            init_fields=init_fields,
            has_init=has_init,
            no_native=no_native,
            own_field_inits=[
                FieldInit(
                    name=field.name,
                    has_default=field.c_has_default,
                    default=field.c_default,
                    has_factory=field.c_default_factory is not MISSING,
                )
                for field in type_info.fields
            ],
        )
