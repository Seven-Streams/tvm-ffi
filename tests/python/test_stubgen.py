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

import pytest
import tvm_ffi.stub.cli as stub_cli
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.backend import Backend, get_backend
from tvm_ffi.stub.cli import _stage_2, _stage_3
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo
from tvm_ffi.stub.python_backend import consts as PC
from tvm_ffi.stub.python_backend.codegen import (
    generate_python_all,
    generate_python_export,
    generate_python_ffi_api,
    generate_python_global_funcs,
    generate_python_import_section,
    generate_python_init,
    generate_python_object,
    render_func_signature,
    render_object_fields,
    render_object_methods,
)
from tvm_ffi.stub.python_backend.imports import ImportItem
from tvm_ffi.stub.rust_backend import consts as RC
from tvm_ffi.stub.rust_backend.backend import RustBackend
from tvm_ffi.stub.rust_backend.codegen import (
    UnsupportedTypeError,
    build_ty_render,
    generate_rust_import_section,
    generate_rust_object,
    render_rust_type,
)
from tvm_ffi.stub.rust_backend.imports import RustImports, RustUse
from tvm_ffi.stub.utils import (
    FuncInfo,
    InitConfig,
    InitFieldInfo,
    NamedTypeSchema,
    ObjectInfo,
    Options,
)


def _identity_ty_map(name: str) -> str:
    return name


def _default_ty_map() -> dict[str, str]:
    return PC.TY_MAP_DEFAULTS.copy()


def _type_suffix(name: str) -> str:
    return PC.TY_MAP_DEFAULTS.get(name, name).rsplit(".", 1)[-1]


def test_codeblock_from_begin_line_variants() -> None:
    cases = [
        (f"{C.PYTHON_SYNTAX.begin} global/demo", "global", ("demo", "")),
        (f"{C.PYTHON_SYNTAX.begin} global/demo@.registry", "global", ("demo", ".registry")),
        (f"{C.PYTHON_SYNTAX.begin} object/demo.TypeBase", "object", "demo.TypeBase"),
        (f"{C.PYTHON_SYNTAX.begin} ty-map/custom", "ty-map", "custom"),
        (f"{C.PYTHON_SYNTAX.begin} import-section", "import-section", ""),
    ]
    for lineno, (line, kind, param) in enumerate(cases, start=1):
        block = CodeBlock.from_begin_line(lineno, line)
        assert block.kind == kind
        assert block.param == param
        assert block.lineno_start == lineno
        assert block.lineno_end is None
        assert block.lines == []


def test_codeblock_from_begin_line_ty_map_and_unknown() -> None:
    line = f"{C.PYTHON_SYNTAX.ty_map} custom -> mapped"
    block = CodeBlock.from_begin_line(5, line)
    assert block.kind == "ty-map"
    assert block.param == "custom -> mapped"
    assert block.lineno_start == 5
    assert block.lineno_end == 5

    with pytest.raises(ValueError):
        CodeBlock.from_begin_line(1, f"{C.PYTHON_SYNTAX.begin} unsupported/kind")


def test_fileinfo_from_file_skip_and_missing_markers(tmp_path: Path) -> None:
    skip = tmp_path / "skip.py"
    skip.write_text(f"print('hi')\n{C.PYTHON_SYNTAX.skip_file}\n", encoding="utf-8")
    assert FileInfo.from_file(skip) is None

    plain = tmp_path / "plain.py"
    plain.write_text("print('plain')\n", encoding="utf-8")
    assert FileInfo.from_file(plain) is None


def test_fileinfo_from_file_parses_blocks(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "first = 1",
            f"{C.PYTHON_SYNTAX.begin} global/demo.func",
            "in_stub = True",
            C.PYTHON_SYNTAX.end,
            f"{C.PYTHON_SYNTAX.ty_map} x -> y",
        ]
    )
    path = tmp_path / "demo.py"
    path.write_text(content, encoding="utf-8")

    info = FileInfo.from_file(path)
    assert info is not None
    assert info.path == path.resolve()
    assert len(info.code_blocks) == 3

    first, stub, ty_map = info.code_blocks
    assert first.kind is None and first.lines == ["first = 1"]

    assert stub.kind == "global"
    assert stub.param == ("demo.func", "")
    assert stub.lineno_start == 2
    assert stub.lineno_end == 4
    assert stub.lines == [
        f"{C.PYTHON_SYNTAX.begin} global/demo.func",
        "in_stub = True",
        C.PYTHON_SYNTAX.end,
    ]

    assert ty_map.kind == "ty-map"
    assert ty_map.param == "x -> y"
    assert ty_map.lineno_start == ty_map.lineno_end == 5
    assert ty_map.lines == [f"{C.PYTHON_SYNTAX.ty_map} x -> y"]


def test_fileinfo_from_file_error_paths(tmp_path: Path) -> None:
    nested = tmp_path / "nested.py"
    nested.write_text(
        "\n".join(
            [
                f"{C.PYTHON_SYNTAX.begin} global/outer",
                f"{C.PYTHON_SYNTAX.begin} global/inner",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Nested stub not permitted"):
        FileInfo.from_file(nested)

    unmatched_end = tmp_path / "unmatched.py"
    unmatched_end.write_text(C.PYTHON_SYNTAX.end + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unmatched"):
        FileInfo.from_file(unmatched_end)

    unclosed = tmp_path / "unclosed.py"
    unclosed.write_text(f"{C.PYTHON_SYNTAX.begin} global/method\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unclosed stub block"):
        FileInfo.from_file(unclosed)


def test_funcinfo_gen_variants() -> None:
    called: list[str] = []

    def ty_map(name: str) -> str:
        called.append(name)
        return name

    schema_no_args = NamedTypeSchema("demo.no_args", TypeSchema("Callable", ()))
    func = FuncInfo(schema=schema_no_args, is_member=False)
    assert render_func_signature(func, ty_map, indent=2) == "  def no_args(*args: Any) -> Any: ..."
    assert called == ["Any"]

    schema_member = NamedTypeSchema(
        "pkg.Class.method",
        TypeSchema(
            "Callable",
            (
                TypeSchema("str"),
                TypeSchema("int"),
                TypeSchema("float"),
            ),
        ),
    )
    member_func = FuncInfo(schema=schema_member, is_member=True)
    assert (
        render_func_signature(member_func, _identity_ty_map, indent=0)
        == "def method(self, _1: float, /) -> str: ..."
    )

    schema_bad = NamedTypeSchema("bad", TypeSchema("int"))
    with pytest.raises(ValueError):
        render_func_signature(
            FuncInfo(schema=schema_bad, is_member=False), _identity_ty_map, indent=0
        )


def test_objectinfo_gen_fields_and_methods() -> None:
    ty_calls: list[str] = []

    def ty_map(name: str) -> str:
        ty_calls.append(name)
        return {"list": "Sequence", "dict": "Mapping"}.get(name, name)

    info = ObjectInfo(
        fields=[
            NamedTypeSchema("field_a", TypeSchema("list", (TypeSchema("int"),))),
            NamedTypeSchema(
                "field_b", TypeSchema("dict", (TypeSchema("str"), TypeSchema("float")))
            ),
        ],
        methods=[
            FuncInfo(
                schema=NamedTypeSchema("demo.static", TypeSchema("Callable", (TypeSchema("int"),))),
                is_member=False,
            ),
            FuncInfo(
                schema=NamedTypeSchema(
                    "demo.member",
                    TypeSchema("Callable", (TypeSchema("str"), TypeSchema("bytes"))),
                ),
                is_member=True,
            ),
        ],
    )

    assert render_object_fields(info, ty_map, indent=2) == [
        "  field_a: Sequence[int]",
        "  field_b: Mapping[str, float]",
    ]
    assert ty_calls.count("list") == 1 and ty_calls.count("dict") == 1

    methods = render_object_methods(info, _identity_ty_map, indent=2)
    assert methods == [
        "  @staticmethod",
        "  def static() -> int: ...",
        "  def member(self, /) -> str: ...",
    ]


def test_type_schema_container_origins() -> None:
    """Test that Array/List/Map/Dict origins are distinct and validated correctly."""
    # Array and List: 0 or 1 arg, default to (Any,)
    for origin in ("Array", "List"):
        s = TypeSchema(origin)
        assert s.args == (TypeSchema("Any"),), f"{origin} should default to (Any,)"
        s = TypeSchema(origin, (TypeSchema("int"),))
        assert s.repr() == f"{origin}[int]"

    # Map and Dict: 0 or 2 args, default to (Any, Any)
    for origin in ("Map", "Dict"):
        s = TypeSchema(origin)
        assert s.args == (TypeSchema("Any"), TypeSchema("Any")), (
            f"{origin} should default to (Any, Any)"
        )
        s = TypeSchema(origin, (TypeSchema("str"), TypeSchema("float")))
        assert s.repr() == f"{origin}[str, float]"

    # from_json_str round-trip through _TYPE_SCHEMA_ORIGIN_CONVERTER
    s = TypeSchema.from_json_str('{"type":"ffi.Array","args":[{"type":"int"}]}')
    assert s.origin == "Array"
    assert s.repr() == "Array[int]"

    s = TypeSchema.from_json_str('{"type":"ffi.List","args":[{"type":"str"}]}')
    assert s.origin == "List"
    assert s.repr() == "List[str]"

    s = TypeSchema.from_json_str('{"type":"ffi.Map","args":[{"type":"str"},{"type":"int"}]}')
    assert s.origin == "Map"
    assert s.repr() == "Map[str, int]"

    s = TypeSchema.from_json_str('{"type":"ffi.Dict","args":[{"type":"str"},{"type":"float"}]}')
    assert s.origin == "Dict"
    assert s.repr() == "Dict[str, float]"

    # Backward compat: "list" and "dict" origins still work
    s = TypeSchema("list", (TypeSchema("int"),))
    assert s.repr() == "list[int]"
    s = TypeSchema("dict", (TypeSchema("str"), TypeSchema("int")))
    assert s.repr() == "dict[str, int]"


def test_objectinfo_gen_fields_container_types() -> None:
    """Test that ObjectInfo fields render distinct container annotations."""
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("arr", TypeSchema("Array", (TypeSchema("int"),))),
            NamedTypeSchema("lst", TypeSchema("List", (TypeSchema("str"),))),
            NamedTypeSchema("mp", TypeSchema("Map", (TypeSchema("str"), TypeSchema("int")))),
            NamedTypeSchema("dt", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("float")))),
        ],
        methods=[],
    )
    assert render_object_fields(info, _type_suffix, indent=0) == [
        "arr: Sequence[int]",
        "lst: MutableSequence[str]",
        "mp: Mapping[str, int]",
        "dt: MutableMapping[str, float]",
    ]


def test_generate_global_funcs_updates_block() -> None:
    code = CodeBlock(
        kind="global",
        param=("demo", "mockpkg"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg", C.PYTHON_SYNTAX.end],
    )
    funcs = [
        FuncInfo(
            schema=NamedTypeSchema(
                "demo.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            ),
            is_member=False,
        )
    ]
    opts = Options(indent=2)
    imports: list[ImportItem] = []
    generate_python_global_funcs(code, funcs, _default_ty_map(), imports, opts)
    assert imports == [
        ImportItem("mockpkg.init_ffi_api", alias="_FFI_INIT_FUNC"),
        ImportItem("typing.TYPE_CHECKING"),
    ]
    assert code.lines == [
        f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg",
        "# fmt: off",
        '_FFI_INIT_FUNC("demo", __name__)',
        "if TYPE_CHECKING:",
        "  def add_one(_0: int, /) -> int: ...",
        "# fmt: on",
        C.PYTHON_SYNTAX.end,
    ]


def test_generate_global_funcs_noop_on_empty_list() -> None:
    code = CodeBlock(
        kind="global",
        param=("empty", ""),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/empty", C.PYTHON_SYNTAX.end],
    )
    imports: list[ImportItem] = []
    generate_python_global_funcs(code, [], _default_ty_map(), imports, Options())
    assert code.lines == [f"{C.PYTHON_SYNTAX.begin} global/empty", C.PYTHON_SYNTAX.end]
    assert imports == []


def test_generate_global_funcs_respects_custom_import_from() -> None:
    code = CodeBlock(
        kind="global",
        param=("demo", "custom.mod"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/demo@custom.mod", C.PYTHON_SYNTAX.end],
    )
    funcs = [
        FuncInfo(
            schema=NamedTypeSchema(
                "demo.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            ),
            is_member=False,
        )
    ]
    imports: list[ImportItem] = []
    generate_python_global_funcs(code, funcs, _default_ty_map(), imports, Options(indent=0))
    assert ImportItem("custom.mod.init_ffi_api", alias="_FFI_INIT_FUNC") in imports


def test_generate_global_funcs_aliases_colliding_type() -> None:
    """When a function name matches a type name, the type import gets an alias."""
    code = CodeBlock(
        kind="global",
        param=("demo", "mockpkg"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg", C.PYTHON_SYNTAX.end],
    )
    # Function "demo.Foo" returns type "demo.Foo" — name collision
    funcs = [
        FuncInfo(
            schema=NamedTypeSchema(
                "demo.Foo",
                TypeSchema("Callable", (TypeSchema("demo.Foo"), TypeSchema("Any"))),
            ),
            is_member=False,
        )
    ]
    ty_map = _default_ty_map()
    ty_map["demo.Foo"] = "somepkg.Foo"
    imports: list[ImportItem] = []
    generate_python_global_funcs(code, funcs, ty_map, imports, Options(indent=4))
    # The type import should use an alias to avoid shadowing the function
    assert ImportItem("somepkg.Foo", type_checking_only=True, alias="_Foo") in imports
    # The function annotation should use the alias
    assert any("-> _Foo:" in line for line in code.lines)


def test_generate_object_fields_only_block() -> None:
    code = CodeBlock(
        kind="object",
        param="demo.TypeDerived",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} object/demo.TypeDerived", C.PYTHON_SYNTAX.end],
    )
    opts = Options(indent=4)
    imports: list[ImportItem] = []
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("field_a", TypeSchema("int")),
            NamedTypeSchema("field_b", TypeSchema("float")),
        ],
        methods=[],
        type_key="demo.TypeDerived",
        parent_type_key="demo.Parent",
    )
    generate_python_object(
        code,
        _default_ty_map(),
        imports,
        opts,
        info,
    )
    assert imports == []

    expected = [
        f"{C.PYTHON_SYNTAX.begin} object/demo.TypeDerived",
        " " * code.indent + "# fmt: off",
        *[
            (" " * code.indent) + line
            for line in render_object_fields(info, _type_suffix, indent=0)
        ],
        " " * code.indent + "# fmt: on",
        C.PYTHON_SYNTAX.end,
    ]
    assert code.lines == expected


def test_generate_object_with_methods() -> None:
    code = CodeBlock(
        kind="object",
        param="demo.IntPair",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} object/demo.IntPair", C.PYTHON_SYNTAX.end],
    )
    opts = Options(indent=4)
    imports: list[ImportItem] = []
    info = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo.from_schema(
                "demo.IntPair.__ffi_init__",
                TypeSchema("Callable", (TypeSchema("None"), TypeSchema("int"), TypeSchema("int"))),
                is_member=True,
            ),
            FuncInfo.from_schema(
                "demo.IntPair.sum",
                TypeSchema("Callable", (TypeSchema("int"),)),
                is_member=True,
            ),
        ],
        type_key="demo.IntPair",
        parent_type_key="demo.Parent",
    )
    generate_python_object(code, _default_ty_map(), imports, opts, info)
    assert set(imports) == {ImportItem("typing.TYPE_CHECKING")}

    assert code.lines[0] == f"{C.PYTHON_SYNTAX.begin} object/demo.IntPair"
    assert code.lines[-1] == C.PYTHON_SYNTAX.end
    assert "# fmt: off" in code.lines[1]
    assert any("if TYPE_CHECKING:" in line for line in code.lines)
    method_lines = [line for line in code.lines if "def __ffi_init__" in line or "def sum" in line]
    # __ffi_init__ from TypeMethod is rendered as an instance method (self, ...) -> None
    assert any(line.strip().startswith("def __ffi_init__(self") for line in method_lines)
    assert any(line.strip().startswith("def sum") for line in method_lines)


def test_generate_import_section_groups_modules() -> None:
    code = CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} import", C.PYTHON_SYNTAX.end],
    )
    imports = [
        ImportItem("typing.Any", type_checking_only=True),
        ImportItem("demo_pkg.Tensor", type_checking_only=True),
        ImportItem("demo.TestObjectBase", type_checking_only=True),
        ImportItem("custom.mod.Type", type_checking_only=True),
    ]
    opts = Options(indent=4)
    generate_python_import_section(code, imports, opts)

    expected_prefix = [
        f"{C.PYTHON_SYNTAX.begin} import",
        "# fmt: off",
        "# isort: off",
        "from __future__ import annotations",
        "from typing import TYPE_CHECKING",
        "if TYPE_CHECKING:",
    ]
    assert code.lines[: len(expected_prefix)] == expected_prefix
    assert "    from demo import TestObjectBase" in code.lines
    assert "    from demo_pkg import Tensor" in code.lines
    assert "    from custom.mod import Type" in code.lines
    assert "    from typing import Any" in code.lines
    assert code.lines[-2:] == ["# fmt: on", C.PYTHON_SYNTAX.end]


def test_generate_import_section_no_imports_noop() -> None:
    code = CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} import", C.PYTHON_SYNTAX.end],
    )
    before = list(code.lines)
    generate_python_import_section(code, [], Options())
    assert code.lines == before


def test_generate_all_builds_sorted_and_deduped_list() -> None:
    code = CodeBlock(
        kind="global",
        param="all",
        lineno_start=1,
        lineno_end=2,
        lines=["    " + C.PYTHON_SYNTAX.begin + " global/all", C.PYTHON_SYNTAX.end],
    )
    generate_python_all(
        code,
        names={"tvm_ffi.foo", "bar", "pkg.baz", "bar"},  # duplicates stripped
        opt=Options(indent=2),
    )
    assert code.lines == [
        "    " + C.PYTHON_SYNTAX.begin + " global/all",
        '    "bar",',
        '    "baz",',
        '    "foo",',
        C.PYTHON_SYNTAX.end,
    ]


def test_generate_all_noop_on_empty_names() -> None:
    code = CodeBlock(
        kind="global",
        param="all-empty",
        lineno_start=1,
        lineno_end=2,
        lines=[C.PYTHON_SYNTAX.begin + " global/all-empty", C.PYTHON_SYNTAX.end],
    )
    before = list(code.lines)
    generate_python_all(code, names=set(), opt=Options())
    assert code.lines == before


def test_generate_all_uses_isort_style_ordering() -> None:
    code = CodeBlock(
        kind="global",
        param="all-mixed",
        lineno_start=1,
        lineno_end=2,
        lines=[C.PYTHON_SYNTAX.begin + " global/all-mixed", C.PYTHON_SYNTAX.end],
    )
    names = {"foo", "Bar", "LIB", "baz", "Alpha", "CONST"}
    generate_python_all(code, names=names, opt=Options(indent=0))
    assert code.lines == [
        C.PYTHON_SYNTAX.begin + " global/all-mixed",
        '"CONST",',
        '"LIB",',
        '"Alpha",',
        '"Bar",',
        '"baz",',
        '"foo",',
        C.PYTHON_SYNTAX.end,
    ]


def test_stage_3_adds_LIB_when_load_lib_imported(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    global_block = CodeBlock(
        kind="global",
        param=("testing", ""),
        lineno_start=2,
        lineno_end=3,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/testing", C.PYTHON_SYNTAX.end],
    )
    import_obj_block = CodeBlock(
        kind="import-object",
        param=("tvm_ffi.libinfo.load_lib_module", "False", "_FFI_LOAD_LIB"),
        lineno_start=1,
        lineno_end=1,
        lines=[
            f"{C.PYTHON_SYNTAX.import_object} tvm_ffi.libinfo.load_lib_module;False;_FFI_LOAD_LIB"
        ],
    )
    all_block = CodeBlock(
        kind="__all__",
        param="",
        lineno_start=4,
        lineno_end=5,
        lines=[f"{C.PYTHON_SYNTAX.begin} __all__", C.PYTHON_SYNTAX.end],
    )
    file_info = FileInfo(
        path=path,
        lines=tuple(
            line for block in (import_obj_block, global_block, all_block) for line in block.lines
        ),
        code_blocks=[import_obj_block, global_block, all_block],
    )
    funcs = [
        FuncInfo.from_schema(
            "testing.add_one",
            TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
        )
    ]
    _stage_3(
        file_info,
        Options(dry_run=True),
        _default_ty_map(),
        {"testing": funcs},
    )
    lib_lines = [line for line in all_block.lines if "LIB" in line]
    assert any("LIB" in line for line in lib_lines)


def test_generate_export_builds_all_extension() -> None:
    code = CodeBlock(
        kind="export",
        param="ffi_api",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} export/ffi_api", C.PYTHON_SYNTAX.end],
    )
    generate_python_export(code)
    full_text = "\n".join(code.lines)
    assert "from .ffi_api import *" in full_text
    assert "ffi_api__all__" in full_text


def test_generate_init_with_and_without_existing_export_block() -> None:
    code_no_blocks = generate_python_init([], "demo")
    assert "Package demo." in code_no_blocks
    assert f"{C.PYTHON_SYNTAX.begin} export/_ffi_api" in code_no_blocks

    code_with_export = generate_python_init(
        [
            CodeBlock(
                kind="export",
                param="_ffi_api",
                lineno_start=1,
                lineno_end=2,
                lines=["", ""],
            )
        ],
        "demo",
    )
    assert code_with_export == ""


def test_generate_ffi_api_without_objects_includes_sections() -> None:
    init_cfg = InitConfig(pkg="pkg", shared_target="pkg_shared", prefix="pkg.")
    code = generate_python_ffi_api(
        [],
        _default_ty_map(),
        "demo.mod",
        [],
        init_cfg,
        is_root=False,
    )
    assert f"{C.PYTHON_SYNTAX.begin} import-section" in code
    assert f"{C.PYTHON_SYNTAX.begin} global/demo.mod" in code
    assert C.PYTHON_SYNTAX.begin + " __all__" in code
    assert "LIB =" not in code


def test_generate_ffi_api_with_objects_imports_parents() -> None:
    init_cfg = InitConfig(pkg="pkg", shared_target="pkg_shared", prefix="pkg.")
    obj_info = ObjectInfo(
        fields=[],
        methods=[],
        type_key="demo.TypeDerived",
        parent_type_key="demo.Parent",
    )
    parent_key = obj_info.parent_type_key
    code = generate_python_ffi_api(
        [],
        _default_ty_map(),
        "demo",
        [obj_info],
        init_cfg,
        is_root=False,
    )
    assert C.PYTHON_SYNTAX.import_object in code  # register_object prompt
    assert f"{C.PYTHON_SYNTAX.begin} object/{obj_info.type_key}" in code
    assert parent_key is not None
    parent_import_prompt = (
        f"{C.PYTHON_SYNTAX.import_object} {parent_key};False;_{parent_key.replace('.', '_')}"
    )
    assert parent_import_prompt in code


def test_stage_2_filters_prefix_and_marks_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefixes: dict[str, list[FuncInfo]] = {"demo.sub": [], "demo": [], "other": []}
    monkeypatch.setattr(stub_cli, "collect_type_keys", lambda: prefixes)
    monkeypatch.setattr(stub_cli, "toposort_objects", lambda objs: [])

    global_funcs = {
        "demo.sub": [
            FuncInfo.from_schema(
                "demo.sub.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            )
        ],
        "demo": [
            FuncInfo.from_schema(
                "demo.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            )
        ],
        "other": [
            FuncInfo.from_schema(
                "other.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            )
        ],
    }
    _stage_2(
        files=[],
        ty_map=_default_ty_map(),
        init_cfg=InitConfig(pkg="demo-pkg", shared_target="demo_shared", prefix="demo."),
        init_path=tmp_path,
        global_funcs=global_funcs,
    )

    root_api = tmp_path / "demo" / "_ffi_api.py"
    sub_api = tmp_path / "demo" / "sub" / "_ffi_api.py"
    other_api = tmp_path / "other" / "_ffi_api.py"
    assert root_api.exists()
    assert sub_api.exists()
    assert not other_api.exists()
    root_text = root_api.read_text(encoding="utf-8")
    sub_text = sub_api.read_text(encoding="utf-8")
    assert 'LIB = _FFI_LOAD_LIB("demo-pkg", "demo_shared")' in root_text
    assert "LIB =" not in sub_text


# ---------------------------------------------------------------------------
# Rust backend: constant tables (rust_backend/consts.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "rust"),
    [
        # scalars / primitives (type_traits.rs) -- bare, no import
        ("int", "i64"),
        ("float", "f64"),
        ("bool", "bool"),
        ("None", "()"),
        ("Optional", "Option"),  # std prelude, no import
        # core / containers -- fully qualified so `use` can be derived
        ("str", "tvm_ffi::String"),  # NOT std::string::String
        ("Any", "tvm_ffi::Any"),
        ("Callable", "tvm_ffi::Function"),
        ("Array", "tvm_ffi::Array"),  # crate's own Array<T>, NOT Vec
        ("Object", "tvm_ffi::Object"),
        ("Tensor", "tvm_ffi::Tensor"),
        ("Shape", "tvm_ffi::Shape"),
        ("Device", "tvm_ffi::DLDevice"),  # dlpack DLDevice, NOT a `Device` type
        ("dtype", "tvm_ffi::DLDataType"),  # dlpack DLDataType, NOT a `DataType` type
        ("DataType", "tvm_ffi::DLDataType"),
        # builtin object type keys
        ("ffi.String", "tvm_ffi::String"),
        ("ffi.Bytes", "tvm_ffi::Bytes"),
        ("ffi.Module", "tvm_ffi::Module"),
        ("ffi.Error", "tvm_ffi::Error"),
        ("ffi.Function", "tvm_ffi::Function"),
    ],
)
def test_rust_ty_map_defaults(origin: str, rust: str) -> None:
    assert RC.RUST_TY_MAP_DEFAULTS[origin] == rust


def test_rust_array_is_not_vec() -> None:
    # Guard against regressing to the (wrong) Vec/HashMap assumption.
    assert not any("Vec" in v for v in RC.RUST_TY_MAP_DEFAULTS.values())
    assert not any("HashMap" in v for v in RC.RUST_TY_MAP_DEFAULTS.values())


def test_rust_crate_types_are_fully_qualified() -> None:
    # Non-primitive crate types must carry the `tvm_ffi::` path so the import
    # collector can emit `use tvm_ffi::...`. Primitives/prelude stay bare.
    bare_ok = {"i64", "f64", "bool", "()", "Option"}
    for origin, rust in RC.RUST_TY_MAP_DEFAULTS.items():
        if rust in bare_ok:
            assert "::" not in rust, origin
        else:
            assert rust.startswith("tvm_ffi::"), (origin, rust)


@pytest.mark.parametrize("origin", ["Map", "Dict", "List", "Union"])
def test_rust_unsupported_origins(origin: str) -> None:
    assert origin in RC.RUST_UNSUPPORTED_ORIGINS
    # An unsupported origin must never also have a (bogus) default mapping.
    assert origin not in RC.RUST_TY_MAP_DEFAULTS


def test_rust_mod_map_ffi_to_crate_root() -> None:
    assert RC.RUST_MOD_MAP["ffi"] == "tvm_ffi"


# ---------------------------------------------------------------------------
# Rust backend: use modelling (rust_backend/imports.py)
# ---------------------------------------------------------------------------


def test_rustuse_keeps_qualified_path() -> None:
    u = RustUse("tvm_ffi::Array")
    assert u.path == "tvm_ffi::Array"
    assert u.leaf == "Array"
    assert u.full_name == "tvm_ffi::Array"
    assert u.name_in_scope == "Array"
    assert u.as_use_line() == "use tvm_ffi::Array;"


def test_rustuse_normalizes_dotted_ffi_name() -> None:
    # leading `ffi` segment rewritten via RUST_MOD_MAP, dots -> ::
    assert RustUse("ffi.String").path == "tvm_ffi::String"
    # unmapped crate prefix is preserved, dots still -> ::
    u = RustUse("my_pkg.sub.Foo")
    assert u.path == "my_pkg::sub::Foo"
    assert u.leaf == "Foo"
    assert u.as_use_line() == "use my_pkg::sub::Foo;"


def test_rustuse_alias() -> None:
    u = RustUse("tvm_ffi::Array", alias="_Array")
    assert u.name_in_scope == "_Array"
    assert u.as_use_line() == "use tvm_ffi::Array as _Array;"


@pytest.mark.parametrize("bare", ["i64", "f64", "bool", "()", "Option"])
def test_rustuse_bare_types_need_no_use(bare: str) -> None:
    u = RustUse(bare)
    assert u.path == bare
    assert u.leaf == bare
    assert u.as_use_line() == ""


def test_rustuse_is_hashable_and_dedups() -> None:
    a = RustUse("tvm_ffi::Array")
    b = RustUse("tvm_ffi::Array")
    assert a == b
    assert len({a, b}) == 1


def test_rustimports_default_empty() -> None:
    imp = RustImports()
    assert imp.items == []
    imp.items.append(RustUse("tvm_ffi::Tensor"))
    assert imp.items[0].leaf == "Tensor"


# ---------------------------------------------------------------------------
# Rust backend: type renderer (rust_backend/codegen.py)
# ---------------------------------------------------------------------------


def _rust_render(schema: TypeSchema) -> tuple[str, RustImports]:
    """Render `schema` with a fresh collector; return (text, imports)."""
    imports = RustImports()
    ty_render = build_ty_render(RC.RUST_TY_MAP_DEFAULTS, imports)
    return render_rust_type(schema, ty_render), imports


def test_render_primitive_no_import() -> None:
    text, imports = _rust_render(TypeSchema("int"))
    assert text == "i64"
    assert imports.items == []  # primitives need no `use`


def test_render_optional() -> None:
    text, _ = _rust_render(TypeSchema("Optional", (TypeSchema("int"),)))
    assert text == "Option<i64>"


def test_render_array_records_use() -> None:
    text, imports = _rust_render(TypeSchema("Array", (TypeSchema("int"),)))
    assert text == "Array<i64>"
    assert RustUse("tvm_ffi::Array") in imports.items


def test_render_callable_is_function() -> None:
    text, imports = _rust_render(TypeSchema("Callable", (TypeSchema("int"),)))
    assert text == "Function"
    assert RustUse("tvm_ffi::Function") in imports.items


def test_render_tuple() -> None:
    assert _rust_render(TypeSchema("tuple"))[0] == "()"
    text, _ = _rust_render(TypeSchema("tuple", (TypeSchema("int"), TypeSchema("float"))))
    assert text == "(i64, f64)"


def test_render_object_leaf_records_use() -> None:
    text, imports = _rust_render(TypeSchema("ffi.String"))
    assert text == "String"
    assert RustUse("tvm_ffi::String") in imports.items


def test_render_nested() -> None:
    schema = TypeSchema("Optional", (TypeSchema("Array", (TypeSchema("int"),)),))
    text, imports = _rust_render(schema)
    assert text == "Option<Array<i64>>"
    assert RustUse("tvm_ffi::Array") in imports.items


@pytest.mark.parametrize(
    "schema",
    [
        TypeSchema("Union", (TypeSchema("int"), TypeSchema("str"))),
        TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))),
        TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))),
        TypeSchema("List", (TypeSchema("int"),)),
    ],
)
def test_render_unsupported_raises(schema: TypeSchema) -> None:
    with pytest.raises(UnsupportedTypeError) as exc:
        _rust_render(schema)
    assert exc.value.origin == schema.origin


def test_render_unsupported_nested_raises() -> None:
    # Map buried inside an Array still bubbles up.
    schema = TypeSchema("Array", (TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))),))
    with pytest.raises(UnsupportedTypeError) as exc:
        _rust_render(schema)
    assert exc.value.origin == "Map"


def test_rust_backend_render_type_delegates() -> None:
    imports = RustImports()
    ty_render = build_ty_render(RC.RUST_TY_MAP_DEFAULTS, imports)
    out = RustBackend().render_type(TypeSchema("Optional", (TypeSchema("int"),)), ty_render)
    assert out == "Option<i64>"


def test_ty_render_dedups_same_path() -> None:
    imports = RustImports()
    tr = build_ty_render(RC.RUST_TY_MAP_DEFAULTS, imports)
    assert tr("Array") == "Array"
    assert tr("Array") == "Array"  # same path again -> reuse binding
    assert imports.items == [RustUse("tvm_ffi::Array")]  # recorded exactly once


def test_ty_render_aliases_same_leaf_different_path() -> None:
    imports = RustImports()
    tr = build_ty_render({"A": "crate_a::Foo", "B": "crate_b::Foo", "C": "crate_c::Foo"}, imports)
    assert tr("A") == "Foo"  # first claims the bare leaf
    assert tr("B") == "Foo2"  # collision -> aliased
    assert tr("C") == "Foo3"  # next collision
    lines = [u.as_use_line() for u in imports.items]
    assert lines == [
        "use crate_a::Foo;",
        "use crate_b::Foo as Foo2;",
        "use crate_c::Foo as Foo3;",
    ]


def test_ty_render_bare_types_not_tracked() -> None:
    imports = RustImports()
    tr = build_ty_render(RC.RUST_TY_MAP_DEFAULTS, imports)
    assert tr("int") == "i64"
    assert tr("Optional") == "Option"  # prelude, bare
    assert imports.items == []


def test_ty_render_seeds_from_existing_imports() -> None:
    # A pre-seeded `use` (e.g. from an import-object directive) must be respected:
    # a different path with the same leaf gets aliased rather than colliding.
    imports = RustImports(items=[RustUse("crate_a::Foo")])
    tr = build_ty_render({"B": "crate_b::Foo"}, imports)
    assert tr("B") == "Foo2"


# ---------------------------------------------------------------------------
# Rust backend: object generation (rust_backend/codegen.py)
# ---------------------------------------------------------------------------


def _rust_object_block(key: str) -> CodeBlock:
    return CodeBlock(
        kind="object",
        param=key,
        lineno_start=1,
        lineno_end=2,
        lines=[f"// tvm-ffi-stubgen(begin): object/{key}", "// tvm-ffi-stubgen(end)"],
    )


def _gen_rust_object(info: ObjectInfo) -> tuple[str, RustImports]:
    block = _rust_object_block(info.type_key or "x")
    imports = RustImports()
    generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), info)
    return "\n".join(block.lines), imports


def _expr_info(*, value_frozen: bool = False) -> ObjectInfo:
    """Root `Expr`: field `value: i64`, static `test() -> i64`, init(i64)."""
    return ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("int"), frozen=value_frozen)],
        methods=[
            FuncInfo(
                NamedTypeSchema("test", TypeSchema("Callable", (TypeSchema("int"),))),
                is_member=False,
            )
        ],
        type_key="cpp_rust_test.Expr",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo("value", NamedTypeSchema("value", TypeSchema("int")), False, False)
        ],
        has_init=True,
    )


def _add_info() -> ObjectInfo:
    """Return derived `Add` info with fields, method, and constructor metadata."""
    return ObjectInfo(
        fields=[
            NamedTypeSchema("a", TypeSchema("cpp_rust_test.Expr")),
            NamedTypeSchema("b", TypeSchema("cpp_rust_test.Expr")),
        ],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    "update",
                    TypeSchema("Callable", (TypeSchema("None"), TypeSchema("cpp_rust_test.Add"))),
                ),
                is_member=True,
            )
        ],
        type_key="cpp_rust_test.Add",
        parent_type_key="cpp_rust_test.Expr",
        init_fields=[
            InitFieldInfo(
                "a", NamedTypeSchema("a", TypeSchema("cpp_rust_test.Expr")), False, False
            ),
            InitFieldInfo(
                "b", NamedTypeSchema("b", TypeSchema("cpp_rust_test.Expr")), False, False
            ),
            InitFieldInfo("value", NamedTypeSchema("value", TypeSchema("int")), False, False),
        ],
        has_init=True,
    )


def test_rust_object_root_struct_and_impl() -> None:
    text, imports = _gen_rust_object(_expr_info())
    # data struct embeds the root Object as `base`
    assert "#[repr(C)]" in text
    assert "struct ExprObj {" in text
    assert "    base: Object," in text
    assert "    pub value: i64," in text
    # ObjectCore impl
    assert 'const TYPE_KEY: &\'static str = "cpp_rust_test.Expr";' in text
    assert "        lookup_type_index(Self::TYPE_KEY)" in text
    assert "        Object::object_header_mut(&mut this.base)" in text
    # ref + Deref/DerefMut (value is def_rw -> mutable class)
    assert "#[derive(DeriveObjectRef, Clone)]" in text
    assert "struct Expr {" in text
    assert "    data: ObjectArc<ExprObj>," in text
    assert "impl Deref for Expr {" in text
    assert "impl DerefMut for Expr {" in text
    # new via __ffi_init__
    assert "fn new(value: i64) -> Result<Self> {" in text
    assert 'let ctor = get_type_method(ExprObj::TYPE_KEY, "__ffi_init__")?;' in text
    assert "let call = into_typed_fn!(ctor, Fn(i64) -> Result<Expr>);" in text
    # static method: no self
    assert "fn test() -> Result<i64> {" in text
    assert 'let f = get_type_method(ExprObj::TYPE_KEY, "test")?;' in text
    assert "let call = into_typed_fn!(f, Fn() -> Result<i64>);" in text
    uses = {u.as_use_line() for u in imports.items}
    assert "use tvm_ffi::object::Object;" in uses
    assert "use std::ops::DerefMut;" in uses


def test_rust_object_derived_embeds_parent() -> None:
    text, _ = _gen_rust_object(_add_info())
    assert "struct AddObj {" in text
    assert "    base: ExprObj," in text  # parent Obj embedded, not Object
    assert "    pub a: Expr," in text
    assert "        ExprObj::object_header_mut(&mut this.base)" in text
    # derived Obj also derefs to its embedded base
    assert "impl Deref for AddObj {" in text
    assert "    type Target = ExprObj;" in text
    # instance method: &mut self receiver (mutable class) but shared `&Add` in typed fn
    assert "fn update(&mut self) -> Result<()> {" in text
    assert "let call = into_typed_fn!(f, Fn(&Add) -> Result<()>);" in text
    assert "        call(self)" in text
    assert "fn new(a: Expr, b: Expr, value: i64) -> Result<Self> {" in text


def test_rust_object_immutable_has_no_derefmut() -> None:
    text, _ = _gen_rust_object(_expr_info(value_frozen=True))  # def_ro -> immutable
    assert "impl Deref for Expr {" in text
    assert "DerefMut" not in text
    assert "fn test() -> Result<i64> {" in text  # static unaffected


def test_rust_object_mixed_fields_warns_and_immutable(capsys: pytest.CaptureFixture[str]) -> None:
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("ro", TypeSchema("int"), frozen=True),
            NamedTypeSchema("rw", TypeSchema("int"), frozen=False),
        ],
        methods=[],
        type_key="demo.Mixed",
        parent_type_key="ffi.Object",
    )
    text, _ = _gen_rust_object(info)
    out = capsys.readouterr().out
    assert "mixed read-only/read-write" in out
    assert "DerefMut" not in text  # treated as immutable


def test_rust_object_skipped_on_unsupported(capsys: pytest.CaptureFixture[str]) -> None:
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("cfg", TypeSchema("Map", (TypeSchema("str"), TypeSchema("int")))),
        ],
        methods=[],
        type_key="demo.HasMap",
        parent_type_key="ffi.Object",
    )
    block = _rust_object_block("demo.HasMap")
    imports = RustImports(items=[RustUse("tvm_ffi::Tensor")])
    generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), info)
    # block emptied to just the markers; imports untouched (no partial leakage)
    assert block.lines == [
        "// tvm-ffi-stubgen(begin): object/demo.HasMap",
        "// tvm-ffi-stubgen(end)",
    ]
    assert imports.items == [RustUse("tvm_ffi::Tensor")]
    assert "[Skipped] object demo.HasMap" in capsys.readouterr().out


def _rust_import_block() -> CodeBlock:
    return CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=["// tvm-ffi-stubgen(begin): import-section", "// tvm-ffi-stubgen(end)"],
    )


def test_rust_import_section_renders_dedups_sorts() -> None:
    block = _rust_import_block()
    imports = RustImports(
        items=[
            RustUse("tvm_ffi::Tensor"),
            RustUse("tvm_ffi::object::ObjectArc"),
            RustUse("tvm_ffi::Tensor"),  # duplicate -> collapsed
            RustUse("crate_b::Foo", alias="Foo2"),
        ]
    )
    generate_rust_import_section(block, imports, Options(), defined_types=set())
    assert block.lines == [
        "// tvm-ffi-stubgen(begin): import-section",
        "use crate_b::Foo as Foo2;",
        "use tvm_ffi::Tensor;",
        "use tvm_ffi::object::ObjectArc;",
        "// tvm-ffi-stubgen(end)",
    ]


def test_rust_import_section_filters_defined_types() -> None:
    block = _rust_import_block()
    imports = RustImports(items=[RustUse("cpp_rust_test::Expr"), RustUse("tvm_ffi::Tensor")])
    # Expr is defined in this file -> its `use` must be dropped.
    generate_rust_import_section(block, imports, Options(), defined_types={"cpp_rust_test::Expr"})
    assert block.lines == [
        "// tvm-ffi-stubgen(begin): import-section",
        "use tvm_ffi::Tensor;",
        "// tvm-ffi-stubgen(end)",
    ]


def test_rust_import_section_empty() -> None:
    block = _rust_import_block()
    generate_rust_import_section(block, RustImports(), Options(), defined_types=set())
    assert block.lines == [
        "// tvm-ffi-stubgen(begin): import-section",
        "// tvm-ffi-stubgen(end)",
    ]


def test_rust_backend_wired() -> None:
    be = get_backend("rust")
    assert isinstance(be, Backend)
    imp = be.new_imports()
    assert isinstance(imp, RustImports)
    be.add_imported_object(imp, "cpp_rust_test.Expr", "False", "")
    assert imp.items == [RustUse("cpp_rust_test::Expr")]
    assert be.canonical_type_name("cpp_rust_test.Expr") == "cpp_rust_test::Expr"
    assert be.extra_export_names(imp) == set()
    # object block delegates to generate_rust_object
    block = _rust_object_block("cpp_rust_test.Expr")
    be.generate_object_block(
        block, RC.RUST_TY_MAP_DEFAULTS.copy(), be.new_imports(), Options(), _expr_info()
    )
    assert "struct ExprObj {" in "\n".join(block.lines)
    # all/export blocks are no-ops (deferred); must not raise
    be.generate_all_block(_rust_object_block("x"), {"Foo"}, Options())
    be.generate_export_block(_rust_object_block("x"))


def test_rust_stage3_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rs = tmp_path / "demo.rs"
    rs.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} object/cpp_rust_test.Expr",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} import-section",
                C.RUST_SYNTAX.end,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    info = FileInfo.from_file(rs)
    assert info is not None
    # Avoid needing a loaded shared library: feed a constructed ObjectInfo.
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", lambda key: _expr_info())

    _stage_3(
        info,
        Options(dry_run=True),
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        {},
        backend=RustBackend(),
    )
    text = "\n".join(info.lines)
    # object block filled
    assert "struct ExprObj {" in text
    assert "impl Expr {" in text
    assert 'get_type_method(ExprObj::TYPE_KEY, "__ffi_init__")' in text
    # import-section filled with the machinery `use`s
    assert "use tvm_ffi::object::ObjectArc;" in text
    assert "use tvm_ffi::object::ObjectCore;" in text
    # Expr defines itself -> no self `use`
    assert "use cpp_rust_test::Expr;" not in text


def test_rust_global_funcs_block_is_noop() -> None:
    # Decision 5: Rust does not generate global functions; the block is untouched.
    lines = ["// tvm-ffi-stubgen(begin): global/demo", "// tvm-ffi-stubgen(end)"]
    block = CodeBlock(
        kind="global", param=("demo", ""), lineno_start=1, lineno_end=2, lines=list(lines)
    )
    funcs = [
        FuncInfo(
            NamedTypeSchema("demo.f", TypeSchema("Callable", (TypeSchema("int"),))), is_member=False
        )
    ]
    imports = RustImports()
    RustBackend().generate_global_funcs_block(
        block, funcs, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options()
    )
    assert block.lines == lines
    assert imports.items == []


def test_rust_object_no_init_no_methods_has_no_impl() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("int"))],
        methods=[],
        type_key="demo.Plain",
        parent_type_key="ffi.Object",
        has_init=False,
    )
    text, _ = _gen_rust_object(info)
    assert "struct PlainObj {" in text
    assert "impl Plain {" not in text  # no new, no methods -> no impl block
    assert "fn new" not in text
