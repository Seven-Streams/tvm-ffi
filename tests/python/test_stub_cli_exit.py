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

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub import cli as stub_cli
from tvm_ffi.stub.utils import (
    FuncInfo,
    InitConfig,
    NamedTypeSchema,
    ObjectInfo,
    Options,
    UnsupportedTypeError,
)


def _options(path: Path, *, dry_run: bool = True, init: InitConfig | None = None) -> Options:
    return Options(files=[str(path)], dry_run=dry_run, init=init, target="rust")


def _global_funcs() -> dict[str, list[FuncInfo]]:
    return {
        "testing": [
            FuncInfo.from_schema(
                "testing.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            )
        ]
    }


def test_cli_does_not_write_partial_file_when_an_object_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mod.rs"
    source.write_text(
        "\n".join(
            [
                "// tvm-ffi-stubgen(begin): global/testing",
                "// tvm-ffi-stubgen(end)",
                "// tvm-ffi-stubgen(begin): object/demo.Unsupported",
                "// tvm-ffi-stubgen(end)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    unsupported = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "items",
                TypeSchema("List", (TypeSchema("int"),)),
            )
        ],
        methods=[],
        type_key="demo.Unsupported",
        parent_type_key="ffi.Object",
    )
    generator = stub_cli.get_generator("rust")

    def skip_object(*_args: Any, **_kwargs: Any) -> None:
        raise UnsupportedTypeError("List")

    monkeypatch.setattr(stub_cli, "_parse_args", lambda: _options(source, dry_run=False))
    monkeypatch.setattr(stub_cli, "get_generator", lambda _target: generator)
    monkeypatch.setattr(stub_cli, "collect_global_funcs", _global_funcs)
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", lambda _key: unsupported)
    monkeypatch.setattr(generator, "generate_object_block", skip_object)

    assert stub_cli.__main__() == 1
    assert source.read_bytes() == original


def test_cli_does_not_write_partial_file_when_generation_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mod.rs"
    source.write_text(
        "\n".join(
            [
                "// tvm-ffi-stubgen(begin): global/testing",
                "// tvm-ffi-stubgen(end)",
                "// tvm-ffi-stubgen(begin): import-section",
                "// tvm-ffi-stubgen(end)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    generator = stub_cli.get_generator("rust")

    def fail_imports(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(stub_cli, "_parse_args", lambda: _options(source, dry_run=False))
    monkeypatch.setattr(stub_cli, "get_generator", lambda _target: generator)
    monkeypatch.setattr(stub_cli, "collect_global_funcs", _global_funcs)
    monkeypatch.setattr(generator, "generate_import_section_block", fail_imports)

    assert stub_cli.__main__() == 1
    assert source.read_bytes() == original


def test_cli_returns_failure_for_missing_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing.rs"
    monkeypatch.setattr(stub_cli, "_parse_args", lambda: _options(missing))

    assert stub_cli.__main__() == 1


def test_cli_returns_failure_for_parse_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "broken.rs"
    source.write_text(
        "// tvm-ffi-stubgen(begin): import-section\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    monkeypatch.setattr(stub_cli, "_parse_args", lambda: _options(source, dry_run=False))

    assert stub_cli.__main__() == 1
    assert source.read_bytes() == original


def test_cli_stops_after_stage_1_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mod.rs"
    source.write_text(
        "\n".join(
            [
                "// tvm-ffi-stubgen(ty-map): invalid",
                "// tvm-ffi-stubgen(begin): global/testing",
                "// tvm-ffi-stubgen(end)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    stage_3_called = False

    def stage_3(*_args: Any, **_kwargs: Any) -> bool:
        nonlocal stage_3_called
        stage_3_called = True
        return False

    monkeypatch.setattr(stub_cli, "_parse_args", lambda: _options(source, dry_run=False))
    monkeypatch.setattr(stub_cli, "collect_global_funcs", _global_funcs)
    monkeypatch.setattr(stub_cli, "_stage_3_transactional", stage_3)

    assert stub_cli.__main__() == 1
    assert not stage_3_called
    assert source.read_bytes() == original


def test_cli_rejects_dry_run_init_without_touching_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mod.rs"
    source.write_text("plain source\n", encoding="utf-8")
    original = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*")}
    init = InitConfig(pkg="demo", shared_target="demo", prefix="demo")
    stage_2_called = False

    def stage_2(*_args: Any, **_kwargs: Any) -> set[str]:
        nonlocal stage_2_called
        stage_2_called = True
        return set()

    monkeypatch.setattr(stub_cli, "_parse_args", lambda: _options(source, init=init))
    monkeypatch.setattr(stub_cli, "_stage_2", stage_2)

    assert stub_cli.__main__() == 1
    assert not stage_2_called
    assert {
        path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*")
    } == original


def test_cli_does_not_finalize_init_after_generation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mod.rs"
    source.write_text(
        "// tvm-ffi-stubgen(begin): import-section\n// tvm-ffi-stubgen(end)\n",
        encoding="utf-8",
    )
    init = InitConfig(pkg="demo", shared_target="demo", prefix="demo")
    generator = stub_cli.get_generator("rust")
    finalized = False

    def finalize(*_args: Any, **_kwargs: Any) -> None:
        nonlocal finalized
        finalized = True

    monkeypatch.setattr(stub_cli, "_parse_args", lambda: _options(source, dry_run=False, init=init))
    monkeypatch.setattr(stub_cli, "get_generator", lambda _target: generator)
    monkeypatch.setattr(stub_cli, "collect_global_funcs", lambda: {})
    monkeypatch.setattr(stub_cli, "_stage_2", lambda *_args, **_kwargs: {"demo"})
    monkeypatch.setattr(stub_cli, "_stage_3_transactional", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(generator, "finalize_init", finalize)

    assert stub_cli.__main__() == 1
    assert not finalized


def test_cli_returns_success_for_complete_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mod.rs"
    source.write_text(
        "// tvm-ffi-stubgen(begin): import-section\n// tvm-ffi-stubgen(end)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(stub_cli, "_parse_args", lambda: _options(source))
    monkeypatch.setattr(stub_cli, "collect_global_funcs", lambda: {})

    assert stub_cli.__main__() == 0
