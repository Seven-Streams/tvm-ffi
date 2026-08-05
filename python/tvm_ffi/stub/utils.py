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

_NATIVE_ALIGNMENT_ATTR = "__ffi_native_alignment__"
_TYPE_MUTABLE_ATTR = "__ffi_type_mutable__"
_TYPE_IS_SIGNED_METADATA = "type_is_signed"


def _parse_type_schema(raw: str | dict[str, Any]) -> TypeSchema:
    """Parse a type schema from either a JSON string or an already-parsed dict."""
    if isinstance(raw, dict):
        return TypeSchema.from_json_obj(raw)
    return TypeSchema.from_json_str(raw)


def _field_signedness(metadata: dict[str, Any]) -> bool | None:
    """Return exact reflected integral signedness, or ``None`` without proof."""
    value = metadata.get(_TYPE_IS_SIGNED_METADATA)
    return value if type(value) is bool else None


class UnsupportedTypeError(Exception):
    """Raised when a backend cannot represent an FFI construct in its target language."""

    def __init__(self, origin: str, reason: str | None = None) -> None:
        """Record the offending ``origin`` and build the message."""
        super().__init__(reason or f"unsupported FFI type {origin!r}")
        self.origin = origin


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
    """A named type schema with reflected layout and default-value metadata.

    ``default`` is the registered static default value (:data:`MISSING` when
    none); ``default_is_factory`` marks a ``default_factory`` registration,
    whose value only exists by calling the factory through FFI.

    ``size``, ``alignment``, and ``offset`` are copied verbatim from the
    reflection registry.  ``signed`` is an exact C++ integral or enum-underlying
    signedness value; it remains ``None`` for other fields and for missing or
    malformed metadata.  These values stay ``None`` for synthetic schemas
    rather than being inferred here, so target-language generators can
    distinguish proven native layout from fixtures or incomplete metadata.
    """

    name: str
    size: int | None = None
    offset: int | None = None
    alignment: int | None = None
    default: Any = MISSING
    default_is_factory: bool = False
    signed: bool | None = None

    def __init__(
        self,
        name: str,
        schema: TypeSchema,
        size: int | None = None,
        offset: int | None = None,
        alignment: int | None = None,
        default: Any = MISSING,
        default_is_factory: bool = False,
        signed: bool | None = None,
    ) -> None:
        """Initialize a `NamedTypeSchema` with the given name, schema and field metadata."""
        super().__init__(origin=schema.origin, args=schema.args)
        self.name = name
        self.size = size
        self.offset = offset
        self.alignment = alignment
        self.default = default
        self.default_is_factory = default_is_factory
        self.signed = signed


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
class ObjectInfo:
    """Information of an object type, including its fields and methods.

    ``mutable`` is the class-level mutability contract (C++ ``_type_mutable``).
    It is registry evidence only when ``has_mutability_metadata`` is true;
    missing or malformed metadata must not be interpreted as immutable merely
    because the synthetic-fixture default is ``False``.
    """

    fields: list[NamedTypeSchema]
    methods: list[FuncInfo]
    type_key: str | None = None
    parent_type_key: str | None = None
    init_fields: list[InitFieldInfo] = dataclasses.field(default_factory=list)
    has_init: bool = False
    mutable: bool = False
    has_mutability_metadata: bool = False
    native_total_size: int | None = None
    parent_native_total_size: int | None = None
    has_native_layout_metadata: bool = False
    parent_has_native_layout_metadata: bool = False
    native_alignment: int | None = None
    parent_native_alignment: int | None = None
    has_native_alignment_metadata: bool = False
    parent_has_native_alignment_metadata: bool = False

    @staticmethod
    def _native_layout_size(type_info: TypeInfo | None) -> tuple[bool, int | None]:
        """Return registry-authored fixed size without using ``TypeInfo`` fallback.

        ``TypeInfo.total_size`` computes a Python-side fallback when native type
        metadata is absent.  That value is useful to Python dataclasses but is
        not evidence for mirroring a foreign C++ object in another language.
        Only read it after ``_has_type_metadata`` proves a native metadata
        record exists, and treat the registry's documented zero sentinel as
        missing layout metadata.
        """
        if type_info is None or not type_info._has_type_metadata:
            return False, None
        total_size = int(type_info.total_size)
        if total_size <= 0:
            return False, None
        return True, total_size

    @staticmethod
    def _native_alignment(
        type_info: TypeInfo | None,
        native_total_size: int | None,
    ) -> tuple[bool, int | None]:
        """Return a validated registry-authored ``alignof(Class)`` value.

        Type attrs are extensible and may be absent, stale, or supplied by a
        foreign registry.  Accept only an exact integer power of two that is
        compatible with the native fixed size and every reflected own field.
        This keeps malformed metadata from becoming target-language ABI proof.
        """
        if type_info is None:
            return False, None
        raw_alignment = _lookup_type_attr(type_info.type_index, _NATIVE_ALIGNMENT_ATTR)
        if type(raw_alignment) is not int:
            return False, None
        alignment = int(raw_alignment)
        if alignment <= 0 or alignment & (alignment - 1):
            return False, None
        if native_total_size is not None and native_total_size % alignment != 0:
            return False, None
        field_alignments = [field.alignment for field in type_info.fields or []]
        if any(
            type(field_alignment) is not int
            or field_alignment <= 0
            or alignment < field_alignment
            or alignment % field_alignment != 0
            for field_alignment in field_alignments
        ):
            return False, None
        return True, alignment

    @staticmethod
    def _native_mutability(type_info: TypeInfo) -> tuple[bool, bool]:
        """Return an explicitly registered C++ mutability contract."""
        raw_mutable = _lookup_type_attr(type_info.type_index, _TYPE_MUTABLE_ATTR)
        if type(raw_mutable) is not bool:
            return False, False
        return True, raw_mutable

    def has_overloaded_methods(self) -> bool:
        """Return whether reflection exposed multiple signatures for a method."""
        seen: set[tuple[str, bool]] = set()
        for method in self.methods:
            key = (method.schema.name, method.is_member)
            if key in seen:
                return True
            seen.add(key)
        return False

    @staticmethod
    def from_type_info(type_info: TypeInfo) -> ObjectInfo:
        """Construct an `ObjectInfo` from a `TypeInfo` instance."""
        parent_type_info = type_info.parent_type_info
        parent_type_key: str | None = None
        if parent_type_info is not None:
            parent_type_key = parent_type_info.type_key

        has_native_layout_metadata, native_total_size = ObjectInfo._native_layout_size(type_info)
        parent_has_native_layout_metadata, parent_native_total_size = (
            ObjectInfo._native_layout_size(parent_type_info)
        )
        has_native_alignment_metadata, native_alignment = ObjectInfo._native_alignment(
            type_info, native_total_size
        )
        parent_has_native_alignment_metadata, parent_native_alignment = (
            ObjectInfo._native_alignment(
                parent_type_info,
                parent_native_total_size,
            )
        )
        if (
            has_native_alignment_metadata
            and parent_has_native_alignment_metadata
            and native_alignment is not None
            and parent_native_alignment is not None
            and (
                native_alignment < parent_native_alignment
                or native_alignment % parent_native_alignment != 0
            )
        ):
            has_native_alignment_metadata = False
            native_alignment = None
        # A native parent size cannot make a child with no native size metadata
        # safe to mirror.  Keep the independent parent trust bit for diagnostics,
        # but never expose a mixed native/fallback size pair to generators.
        if not has_native_layout_metadata:
            native_total_size = None
            parent_native_total_size = None

        has_mutability_metadata, mutable = ObjectInfo._native_mutability(type_info)

        # Detect __ffi_init__ from TypeMethod or TypeAttrColumn.
        has_init = any(m.name == "__ffi_init__" for m in type_info.methods)
        if not has_init:
            has_init = _lookup_type_attr(type_info.type_index, "__ffi_init__") is not None

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
                                alignment=field.alignment,
                                signed=_field_signedness(field.metadata),
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
                    size=field.size,
                    offset=field.offset,
                    alignment=field.alignment,
                    default=field.c_default,
                    default_is_factory=field.c_default_factory is not MISSING,
                    signed=_field_signedness(field.metadata),
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
            mutable=mutable,
            has_mutability_metadata=has_mutability_metadata,
            native_total_size=native_total_size,
            parent_native_total_size=parent_native_total_size,
            has_native_layout_metadata=has_native_layout_metadata,
            parent_has_native_layout_metadata=parent_has_native_layout_metadata,
            native_alignment=native_alignment,
            parent_native_alignment=parent_native_alignment,
            has_native_alignment_metadata=has_native_alignment_metadata,
            parent_has_native_alignment_metadata=parent_has_native_alignment_metadata,
        )
