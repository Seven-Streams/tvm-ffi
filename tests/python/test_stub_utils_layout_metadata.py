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
"""Tests that stubgen's language-neutral model preserves native layout metadata."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from tvm_ffi.core import MISSING, TypeInfo, TypeSchema
from tvm_ffi.stub import utils as stub_utils
from tvm_ffi.stub.utils import NamedTypeSchema, ObjectInfo
from tvm_ffi.testing import TestIntPair


def _fake_field(
    *,
    alignment: int = 8,
    type_is_signed: object = MISSING,
) -> SimpleNamespace:
    metadata: dict[str, object] = {"type_schema": {"type": "int"}}
    if type_is_signed is not MISSING:
        metadata["type_is_signed"] = type_is_signed
    return SimpleNamespace(
        name="value",
        metadata=metadata,
        size=8,
        alignment=alignment,
        offset=32,
        c_default=MISSING,
        c_default_factory=MISSING,
        c_init=True,
        c_kw_only=False,
        c_has_default=False,
    )


def _fake_type_info(
    type_key: str,
    *,
    type_index: int = 1000,
    parent: SimpleNamespace | None = None,
    fields: list[SimpleNamespace] | None = None,
    has_native_metadata: bool,
    total_size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        type_key=type_key,
        type_index=type_index,
        parent_type_info=parent,
        fields=[] if fields is None else fields,
        methods=[],
        _has_type_metadata=has_native_metadata,
        total_size=total_size,
    )


def _disable_registered_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stub_utils, "_lookup_type_attr", lambda *_args: None)


def _register_fake_attrs(
    monkeypatch: pytest.MonkeyPatch,
    attrs: dict[tuple[int, str], object],
) -> None:
    monkeypatch.setattr(
        stub_utils,
        "_lookup_type_attr",
        lambda type_index, attr_name: attrs.get((type_index, attr_name)),
    )


def test_synthetic_layout_metadata_defaults_to_unknown() -> None:
    field = NamedTypeSchema("value", TypeSchema("int"))
    info = ObjectInfo(fields=[field], methods=[])

    assert field.alignment is None
    assert field.signed is None
    assert info.native_total_size is None
    assert info.parent_native_total_size is None
    assert not info.has_native_layout_metadata
    assert not info.parent_has_native_layout_metadata
    assert info.native_alignment is None
    assert info.parent_native_alignment is None
    assert not info.has_native_alignment_metadata
    assert not info.parent_has_native_alignment_metadata
    assert not info.has_mutability_metadata


def test_from_type_info_preserves_native_size_parent_size_and_field_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _fake_type_info(
        "testing.Parent",
        type_index=1000,
        has_native_metadata=True,
        total_size=24,
    )
    source_field = _fake_field(alignment=16)
    child = _fake_type_info(
        "testing.Child",
        type_index=1001,
        parent=parent,
        fields=[source_field],
        has_native_metadata=True,
        total_size=64,
    )
    _register_fake_attrs(
        monkeypatch,
        {
            (1000, "__ffi_native_alignment__"): 8,
            (1001, "__ffi_native_alignment__"): 16,
            (1001, "__ffi_type_mutable__"): False,
        },
    )

    info = ObjectInfo.from_type_info(cast(TypeInfo, child))

    assert info.has_native_layout_metadata
    assert info.native_total_size == 64
    assert info.parent_has_native_layout_metadata
    assert info.parent_native_total_size == 24
    assert info.has_native_alignment_metadata
    assert info.native_alignment == 16
    assert info.parent_has_native_alignment_metadata
    assert info.parent_native_alignment == 8
    assert info.fields[0].alignment == source_field.alignment
    assert info.has_mutability_metadata
    assert not info.mutable


@pytest.mark.parametrize(
    (
        "child_is_native",
        "parent_is_native",
        "expected_child_size",
        "expected_parent_size",
    ),
    [
        (False, True, None, None),
        (True, False, 64, None),
    ],
)
def test_child_and_parent_native_layout_trust_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    child_is_native: bool,
    parent_is_native: bool,
    expected_child_size: int | None,
    expected_parent_size: int | None,
) -> None:
    _disable_registered_attrs(monkeypatch)
    parent = _fake_type_info(
        "testing.Parent",
        has_native_metadata=parent_is_native,
        total_size=24 if parent_is_native else 4096,
    )
    child = _fake_type_info(
        "testing.Child",
        parent=parent,
        has_native_metadata=child_is_native,
        total_size=64 if child_is_native else 8192,
    )

    info = ObjectInfo.from_type_info(cast(TypeInfo, child))

    assert info.has_native_layout_metadata is child_is_native
    assert info.parent_has_native_layout_metadata is parent_is_native
    assert info.native_total_size == expected_child_size
    assert info.parent_native_total_size == expected_parent_size


def test_zero_native_total_size_is_not_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_registered_attrs(monkeypatch)
    source = _fake_type_info("testing.Unsized", has_native_metadata=True, total_size=0)

    info = ObjectInfo.from_type_info(cast(TypeInfo, source))

    assert not info.has_native_layout_metadata
    assert info.native_total_size is None


@pytest.mark.parametrize("invalid_alignment", [0, -8, 3, True, "16", 8, 128])
def test_invalid_native_alignment_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
    invalid_alignment: object,
) -> None:
    source = _fake_type_info(
        "testing.InvalidAlignment",
        fields=[_fake_field(alignment=16)],
        has_native_metadata=True,
        total_size=64,
    )
    _register_fake_attrs(
        monkeypatch,
        {(1000, "__ffi_native_alignment__"): invalid_alignment},
    )

    info = ObjectInfo.from_type_info(cast(TypeInfo, source))

    assert info.has_native_layout_metadata
    assert info.native_total_size == 64
    assert not info.has_native_alignment_metadata
    assert info.native_alignment is None


@pytest.mark.parametrize("invalid_mutability", [None, 0, "false"])
def test_missing_or_invalid_mutability_is_not_implicit_false(
    monkeypatch: pytest.MonkeyPatch,
    invalid_mutability: object,
) -> None:
    source = _fake_type_info(
        "testing.InvalidMutability",
        has_native_metadata=True,
        total_size=64,
    )
    _register_fake_attrs(
        monkeypatch,
        {(1000, "__ffi_type_mutable__"): invalid_mutability},
    )

    info = ObjectInfo.from_type_info(cast(TypeInfo, source))

    assert not info.has_mutability_metadata
    assert not info.mutable


@pytest.mark.parametrize(
    ("raw_signedness", "expected"),
    [
        pytest.param(True, True, id="signed"),
        pytest.param(False, False, id="unsigned"),
        pytest.param(MISSING, None, id="missing"),
        pytest.param(None, None, id="none"),
        pytest.param(0, None, id="integer-zero"),
        pytest.param(1, None, id="integer-one"),
        pytest.param("false", None, id="string"),
    ],
)
def test_field_signedness_requires_exact_bool_for_fields_and_init_fields(
    monkeypatch: pytest.MonkeyPatch,
    raw_signedness: object,
    expected: bool | None,
) -> None:
    source = _fake_type_info(
        "testing.FieldSignedness",
        fields=[_fake_field(type_is_signed=raw_signedness)],
        has_native_metadata=True,
        total_size=64,
    )
    _register_fake_attrs(monkeypatch, {(1000, "__ffi_init__"): object()})

    info = ObjectInfo.from_type_info(cast(TypeInfo, source))

    assert info.fields[0].signed is expected
    assert len(info.init_fields) == 1
    assert info.init_fields[0].schema.signed is expected


def test_real_cpp_type_info_layout_metadata_is_not_dropped() -> None:
    type_info: TypeInfo = TestIntPair.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    info = ObjectInfo.from_type_info(type_info)

    assert type_info._has_type_metadata
    assert info.has_native_layout_metadata
    assert info.native_total_size == type_info.total_size
    assert info.parent_has_native_layout_metadata
    assert info.parent_native_total_size == type_info.parent_type_info.total_size
    assert info.has_native_alignment_metadata
    assert info.native_alignment is not None
    assert type_info.total_size % info.native_alignment == 0
    assert info.parent_has_native_alignment_metadata
    assert info.parent_native_alignment is not None
    assert info.has_mutability_metadata
    assert not info.mutable
    assert [field.signed for field in info.fields] == [True, True]
    assert [field.schema.signed for field in info.init_fields] == [True, True]
    assert [field.alignment for field in info.fields] == [
        field.alignment for field in type_info.fields
    ]


def test_root_cpp_object_has_explicit_alignment_and_false_mutability() -> None:
    type_info: TypeInfo = TestIntPair.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    root_type_info = type_info.parent_type_info
    assert root_type_info is not None

    info = ObjectInfo.from_type_info(root_type_info)

    assert info.type_key == "ffi.Object"
    assert info.has_native_alignment_metadata
    assert info.native_alignment is not None
    assert info.has_mutability_metadata
    assert not info.mutable
