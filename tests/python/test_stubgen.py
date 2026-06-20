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
from tvm_ffi.core import MISSING, TypeSchema
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.cli import _stage_2, _stage_3
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo
from tvm_ffi.stub.generator import get_generator
from tvm_ffi.stub.python_generator import consts as PC
from tvm_ffi.stub.python_generator.codegen import (
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
from tvm_ffi.stub.python_generator.utils import ImportItem
from tvm_ffi.stub.rust_generator import codegen as rust_codegen
from tvm_ffi.stub.rust_generator import consts as RC
from tvm_ffi.stub.rust_generator.codegen import (
    UnsupportedTypeError,
    finalize_rust_module_tree,
    generate_rust_import_section,
    generate_rust_object,
    render_rust_type,
)
from tvm_ffi.stub.rust_generator.generator import RustGenerator
from tvm_ffi.stub.rust_generator.utils import RustImports, RustUse
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
        block = CodeBlock.from_begin_line(lineno, line, C.PYTHON_SYNTAX)
        assert block.kind == kind
        assert block.param == param
        assert block.lineno_start == lineno
        assert block.lineno_end is None
        assert block.lines == []

    with pytest.raises(ValueError):
        CodeBlock.from_begin_line(1, f"{C.PYTHON_SYNTAX.begin} unsupported/kind", C.PYTHON_SYNTAX)


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


@pytest.mark.parametrize("from_mod", ["mockpkg", "custom.mod"])
def test_generate_global_funcs_updates_block(from_mod: str) -> None:
    code = CodeBlock(
        kind="global",
        param=("demo", from_mod),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/demo@{from_mod}", C.PYTHON_SYNTAX.end],
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
        ImportItem(f"{from_mod}.init_ffi_api", alias="_FFI_INIT_FUNC"),
        ImportItem("typing.TYPE_CHECKING"),
    ]
    assert code.lines == [
        f"{C.PYTHON_SYNTAX.begin} global/demo@{from_mod}",
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


def test_import_item_mod_map_prefix_rewrite() -> None:
    # MOD_MAP rewrites must respect module-path boundaries.
    assert ImportItem("ffi.Object").mod == "tvm_ffi"
    assert ImportItem("testing.TestIntPair").mod == "tvm_ffi.testing"
    assert ImportItem("testing.sub.Thing").mod == "tvm_ffi.testing.sub"
    # A module that merely starts with a mapped prefix is NOT rewritten.
    assert ImportItem("testingfoo.Thing").mod == "testingfoo"
    assert ImportItem("ffi2.Thing").mod == "ffi2"


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
        syntax=C.PYTHON_SYNTAX,
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
        get_generator("python"),
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
    code_no_blocks = generate_python_init([], "demo", "_ffi_api", C.PYTHON_SYNTAX)
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
        "_ffi_api",
        C.PYTHON_SYNTAX,
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
        syntax=C.PYTHON_SYNTAX,
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
        syntax=C.PYTHON_SYNTAX,
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
        generator=get_generator("python"),
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
# Rust backend: use modelling (rust_generator/imports.py)
# ---------------------------------------------------------------------------


def test_rustuse_keeps_qualified_path() -> None:
    u = RustUse("tvm_ffi::Array")
    assert u.path == "tvm_ffi::Array"
    assert u.leaf == "Array"
    assert u.as_use_line() == "use tvm_ffi::Array;"


def test_rustuse_normalizes_dotted_ffi_name() -> None:
    # leading `ffi` segment rewritten via RUST_MOD_MAP to the external crate
    assert RustUse("ffi.String").path == "tvm_ffi::String"
    # a generated-tree type key is rooted at `crate` (mounted at the crate root)
    u = RustUse("my_pkg.sub.Foo")
    assert u.path == "crate::my_pkg::sub::Foo"
    assert u.leaf == "Foo"
    assert u.as_use_line() == "use crate::my_pkg::sub::Foo;"


@pytest.mark.parametrize("bare", ["i64", "bool"])
def test_rustuse_bare_types_need_no_use(bare: str) -> None:
    u = RustUse(bare)
    assert u.path == bare
    assert u.leaf == bare
    assert u.as_use_line() == ""


# ---------------------------------------------------------------------------
# Rust backend: type renderer (rust_generator/codegen.py)
# ---------------------------------------------------------------------------


def _rust_render(schema: TypeSchema) -> tuple[str, RustImports]:
    """Render `schema` with a fresh collector; return (text, imports)."""
    imports = RustImports()
    ty_map = RC.RUST_TY_MAP_DEFAULTS

    def ty_render(origin: str) -> str:
        return imports.record(ty_map.get(origin, origin))

    return render_rust_type(schema, ty_render), imports


def test_render_primitive_no_import() -> None:
    text, imports = _rust_render(TypeSchema("int"))
    assert text == "i64"
    assert imports.items == []  # primitives need no `use`


def test_render_array_records_use() -> None:
    text, imports = _rust_render(TypeSchema("Array", (TypeSchema("int"),)))
    assert text == "Array<i64>"
    assert RustUse("tvm_ffi::Array") in imports.items


def test_render_callable_is_function() -> None:
    text, imports = _rust_render(TypeSchema("Callable", (TypeSchema("int"),)))
    assert text == "Function"
    assert RustUse("tvm_ffi::Function") in imports.items


def test_render_object_leaf_records_use() -> None:
    # Importing `tvm_ffi::String` shadows the prelude `String` in the generated
    # module; that is safe because the derive macros expand with fully
    # qualified `::std::string::String`.
    text, imports = _rust_render(TypeSchema("ffi.String"))
    assert text == "String"
    assert RustUse("tvm_ffi::String") in imports.items


def test_render_nested() -> None:
    schema = TypeSchema("Array", (TypeSchema("Array", (TypeSchema("int"),)),))
    text, imports = _rust_render(schema)
    assert text == "Array<Array<i64>>"
    assert RustUse("tvm_ffi::Array") in imports.items


@pytest.mark.parametrize(
    "schema",
    [
        TypeSchema("Union", (TypeSchema("int"), TypeSchema("str"))),
        TypeSchema("Map"),  # untyped map -> Map<Any, Any>, which is not AnyCompatible
        TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))),
        TypeSchema("List", (TypeSchema("int"),)),
        TypeSchema("tuple", (TypeSchema("int"), TypeSchema("float"))),
        TypeSchema("tuple"),
    ],
)
def test_render_unsupported_raises(schema: TypeSchema) -> None:
    with pytest.raises(UnsupportedTypeError) as exc:
        _rust_render(schema)
    assert exc.value.origin == schema.origin


@pytest.mark.parametrize(
    "inner",
    [
        TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))),
    ],
)
def test_render_unsupported_nested_raises(inner: TypeSchema) -> None:
    # An unsupported origin buried inside an Array still bubbles up and skips the
    # enclosing object.
    schema = TypeSchema("Array", (inner,))
    with pytest.raises(UnsupportedTypeError) as exc:
        _rust_render(schema)
    assert exc.value.origin == inner.origin


def test_render_optional_is_native_option() -> None:
    # `Optional<T>` renders as the native `Option<T>` at function boundaries and
    # when nested (the layout-mirror is only for direct struct fields).
    text, imports = _rust_render(TypeSchema("Optional", (TypeSchema("int"),)))
    assert text == "Option<i64>"
    assert imports.items == []  # `Option` is prelude; no `use`

    text, imports = _rust_render(
        TypeSchema("Array", (TypeSchema("Optional", (TypeSchema("ffi.String"),)),))
    )
    assert text == "Array<Option<String>>"  # nested Optional inside Array


def test_render_map_typed() -> None:
    # A typed map renders as the crate's read-only `Map<K, V>` (NOT std HashMap)
    # and records its `use`, plus the `use` of each element type.
    text, imports = _rust_render(TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))))
    assert text == "Map<String, i64>"
    assert RustUse("tvm_ffi::Map") in imports.items
    assert RustUse("tvm_ffi::String") in imports.items

    # Nested inside an Array round-trips through the same recursive path.
    text, _ = _rust_render(
        TypeSchema("Array", (TypeSchema("Map", (TypeSchema("ffi.String"), TypeSchema("int"))),))
    )
    assert text == "Array<Map<String, i64>>"


@pytest.mark.parametrize(
    "schema",
    [
        TypeSchema("Map"),  # 0 args -> Map<Any, Any>
        TypeSchema("Map", (TypeSchema("Any"), TypeSchema("int"))),  # Any key
        TypeSchema("Map", (TypeSchema("str"), TypeSchema("Object"))),  # bare Object value
    ],
)
def test_render_map_non_any_compatible_inner_is_unsupported(schema: TypeSchema) -> None:
    # `Map<K, V>` needs both K and V AnyCompatible to cross the Any boundary; an
    # `Any` / bare-`Object` element (including the untyped `Map<Any, Any>` default)
    # skips the enclosing object, exactly as before Map was supported.
    with pytest.raises(UnsupportedTypeError) as exc:
        _rust_render(schema)
    assert exc.value.origin == "Map"


@pytest.mark.parametrize(
    "schema",
    [
        TypeSchema("Array", (TypeSchema("Any"),)),  # Array<Any>
        TypeSchema("Array", (TypeSchema("Object"),)),  # Array<bare Object>
        TypeSchema("Array", (TypeSchema("Array", (TypeSchema("Any"),)),)),  # Array<Array<Any>>
    ],
)
def test_render_array_non_any_compatible_element_is_unsupported(schema: TypeSchema) -> None:
    # C4: `Array<T>` requires `T: AnyCompatible`; an `Any` / bare-`Object` element
    # (at any depth) has no rendering (`Array<Any>` would not compile) and skips
    # the enclosing object -- symmetric with the `Map` / `Optional` guards.
    with pytest.raises(UnsupportedTypeError) as exc:
        _rust_render(schema)
    assert exc.value.origin == "Array"


@pytest.mark.parametrize(
    "type_key",
    ["ffi.reflection.AccessPath", "ffi.reflection.AccessStep", "ffi.SomethingUnexposed"],
)
def test_rust_unmapped_ffi_builtin_field_is_unsupported(type_key: str) -> None:
    # C1: an `ffi.*` builtin absent from the ty_map is a crate type the Rust crate
    # does not expose (e.g. `ffi.reflection.AccessPath`). Referencing it must skip
    # the enclosing object, NOT emit a `use tvm_ffi::reflection::AccessPath;` for a
    # nonexistent crate path (the old passthrough did exactly that -> E0432).
    info = ObjectInfo(
        fields=[NamedTypeSchema("p", TypeSchema(type_key))],
        methods=[],
        type_key="cpp_rust_test.Refs",
        parent_type_key="ffi.Object",
    )
    with pytest.raises(UnsupportedTypeError) as exc:
        _gen_rust_object(info)
    assert exc.value.origin == type_key


def test_ty_render_dedups_same_path() -> None:
    imports = RustImports()
    ty_map = RC.RUST_TY_MAP_DEFAULTS

    def tr(origin: str) -> str:
        return imports.record(ty_map.get(origin, origin))

    assert tr("Array") == "Array"
    assert tr("Array") == "Array"  # same path again -> reuse binding
    assert imports.items == [RustUse("tvm_ffi::Array")]  # recorded exactly once


def test_ty_render_same_leaf_different_path_full_qualifies() -> None:
    # No auto-aliasing: two different paths wanting the same in-scope name are
    # disambiguated by full-qualifying the loser inline (it records no `use`).
    imports = RustImports()
    assert imports.record("crate_a::Foo") == "Foo"  # first claims the bare leaf
    assert imports.record("crate_b::Foo") == "crate_b::Foo"  # loser uses full path
    assert imports.items == [RustUse("crate_a::Foo")]  # only the winner is imported
    # The winner still dedups by path; the loser stays full-qualified, no `use`.
    assert imports.record("crate_a::Foo") == "Foo"
    assert imports.record("crate_b::Foo") == "crate_b::Foo"


def test_rust_imports_seed_local_types_records_paths_and_leaves() -> None:
    imports = RustImports()
    imports.seed_local_types(["tirx.Call"])
    # both the ref name and its `<Name>Obj` twin are claimed, crate::-rooted
    assert imports.local_paths == {"crate::tirx::Call", "crate::tirx::CallObj"}
    assert imports.local_leaves == {"Call", "CallObj"}


def test_rust_imports_cross_module_collides_with_local_full_qualifies() -> None:
    # A4: a cross-module import whose leaf collides with a locally-defined type is
    # referenced by full path inline (no `use`, which would be an E0255 clash with
    # the local `pub struct Call`).
    imports = RustImports()
    imports.seed_local_types(["tirx.Call"])  # this file defines `Call` / `CallObj`
    assert imports.record("relax.expr.Call") == "crate::relax::expr::Call"
    assert imports.items == []  # no colliding `use` emitted
    # the `<Name>Obj` twin collides too (a cross-module base `CallObj`)
    assert imports.record("relax.expr.CallObj") == "crate::relax::expr::CallObj"
    assert imports.items == []


def test_rust_imports_same_file_reference_uses_bare_leaf() -> None:
    # A4: a reference to a type the file itself defines is an in-scope sibling --
    # bare leaf, never full-qualified (its `use` is dropped by the import section).
    imports = RustImports()
    imports.seed_local_types(["tirx.Call"])
    assert imports.record("tirx.Call") == "Call"
    assert imports.record("tirx.CallObj") == "CallObj"


# ---------------------------------------------------------------------------
# Rust backend: object generation (rust_generator/codegen.py)
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


def _expr_info(*, mutable: bool = True) -> ObjectInfo:
    """Root `Expr`: field `value: i64`, static `test() -> i64`, init(i64).

    Native-eligible (root, field-binding init), so its ``ffi_new`` is the native
    struct-literal form. The blocked-constructor path is covered by the derived
    fixtures (non-resolvable parent); ``Optional`` fields make the whole type
    unsupported (see ``test_rust_optional_field_is_unsupported``).
    """
    return ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("int"))],
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
        mutable=mutable,
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
        mutable=True,
    )


def _native_point_info() -> ObjectInfo:
    """Root auto-init `Point`: init fields x, y -> native `ObjectArc::new`."""
    return ObjectInfo(
        fields=[
            NamedTypeSchema("x", TypeSchema("int")),
            NamedTypeSchema("y", TypeSchema("int")),
        ],
        methods=[],
        type_key="cpp_rust_test.Point",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo("x", NamedTypeSchema("x", TypeSchema("int")), False, False),
            InitFieldInfo("y", NamedTypeSchema("y", TypeSchema("int")), False, False),
        ],
        has_init=True,
    )


def test_rust_native_root_construction() -> None:
    text, _ = _gen_rust_object(_native_point_info())
    # Auto-init root -> native: `ffi_new()` opens the builder (base prefilled
    # with the root header, fields unset) and `build` allocates via
    # `ObjectArc::new` -- no `__ffi_init__` round-trip. Every field is a
    # setter; the root header is prefilled, so there is no `base` setter.
    assert "pub fn ffi_new() -> PointBuilder {" in text
    assert "base: Object::new()," in text
    assert "pub struct PointBuilder {" in text
    assert "    x: Option<i64>," in text
    assert "pub fn x(mut self, x: i64) -> Self {" in text
    assert "self.x = Some(x);" in text
    assert "pub fn build(self) -> Result<Point> {" in text
    assert "data: ObjectArc::new(self.build_obj()?)," in text
    assert "base: self.base," in text
    # `build_obj` (the bare struct value a derived type's `base` setter takes)
    # ships unconditionally -- even on a root with no child in this DLL -- and
    # holds the missing-field checks that `build` delegates to.
    assert "pub fn build_obj(self) -> Result<PointObj> {" in text
    assert text.count("self.x.ok_or_else") == 1
    assert "pub fn base(" not in text
    assert "impl PointObj {" not in text
    assert "__ffi_init__" not in text
    assert "from_type_method" not in text


def _builder_knobs_info() -> ObjectInfo:
    """Root auto-init `Knobs`: one required field + a default of every renderable kind."""
    return ObjectInfo(
        fields=[
            NamedTypeSchema("scale", TypeSchema("int")),
            NamedTypeSchema("offset", TypeSchema("int"), default=2),
            NamedTypeSchema("verbose", TypeSchema("bool"), default=True),
            NamedTypeSchema("ratio", TypeSchema("float"), default=0.5),
            NamedTypeSchema("label", TypeSchema("ffi.String"), default='he"llo\n'),
        ],
        methods=[],
        type_key="cpp_rust_test.Knobs",
        parent_type_key="ffi.Object",
        has_init=True,
    )


def test_rust_builder_defaulted_fields_prefilled() -> None:
    text, _ = _gen_rust_object(_builder_knobs_info())
    # `ffi_new()` takes no field parameters: the builder API is uniform.
    assert "pub fn ffi_new() -> KnobsBuilder {" in text
    # Defaulted fields are prefilled with their rendered literal (strings are
    # escaped Rust-style: `\"` for the quote, `\u{..}` for non-printables) ...
    assert "offset: 2," in text
    assert "verbose: true," in text
    assert "ratio: 0.5," in text
    assert 'label: tvm_ffi::String::from("he\\"llo\\u{a}"),' in text
    # ... while the field without a default starts unset.
    assert "scale: None," in text
    assert "scale: Option<i64>," in text
    # Every field gets a like-named consuming setter.
    assert "pub fn scale(mut self, scale: i64) -> Self {" in text
    assert "self.scale = Some(scale);" in text
    assert "pub fn offset(mut self, offset: i64) -> Self {" in text
    assert "self.offset = offset;" in text
    assert "pub fn verbose(mut self, verbose: bool) -> Self {" in text
    assert "pub fn label(mut self, label: String) -> Self {" in text
    # `build_obj` checks only the unset-able field and moves the rest.
    assert "pub fn build(self) -> Result<Knobs> {" in text
    assert (
        "let scale = self.scale.ok_or_else(|| tvm_ffi::Error::new("
        'tvm_ffi::VALUE_ERROR, "field `scale` is not set", ""))?;' in text
    )
    assert "offset: self.offset," in text
    assert "scale: self.scale," not in text  # bound via the checked local


@pytest.mark.parametrize(
    ("default", "is_factory"),
    [
        pytest.param([1, 2], False, id="container"),
        pytest.param(float("inf"), False, id="non-finite-float"),
        pytest.param(MISSING, True, id="default-factory"),
    ],
)
def test_rust_unrenderable_default_blocks_native(
    default: object, is_factory: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    # A default stubgen cannot spell as a Rust literal -- or one that only exists
    # by calling an FFI factory -- blocks native construction; with no FFI
    # fallback the constructor is skipped with a warning.
    info = _native_point_info()
    info.fields = [
        NamedTypeSchema("x", TypeSchema("int")),
        NamedTypeSchema("y", TypeSchema("int"), default=default, default_is_factory=is_factory),
    ]
    text, _ = _gen_rust_object(info)
    assert "ffi_new" not in text
    assert "PointBuilder" not in text
    out = capsys.readouterr().out
    assert "[Warning] object cpp_rust_test.Point: skipping `ffi_new`" in out
    assert "'y'" in out


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (NamedTypeSchema("a", TypeSchema("int"), default=2), "2"),
        (NamedTypeSchema("a", TypeSchema("bool"), default=True), "true"),
        (NamedTypeSchema("a", TypeSchema("float"), default=0.5), "0.5"),
        (NamedTypeSchema("a", TypeSchema("float"), default=5), "5.0"),  # int coerces to float
        (NamedTypeSchema("a", TypeSchema("str"), default="x"), 'tvm_ffi::String::from("x")'),
        (NamedTypeSchema("a", TypeSchema("ffi.String"), default="x"), 'tvm_ffi::String::from("x")'),
        # C2: type mismatches have no Rust literal -> None (not a wrong-typed value)
        (NamedTypeSchema("a", TypeSchema("dtype"), default=""), None),
        (NamedTypeSchema("a", TypeSchema("DataType"), default="float32"), None),
        (NamedTypeSchema("a", TypeSchema("Device"), default=""), None),
        (NamedTypeSchema("a", TypeSchema("int"), default="5"), None),  # str on int field
        (NamedTypeSchema("a", TypeSchema("bool"), default=1), None),  # int on bool field
        (NamedTypeSchema("a", TypeSchema("float"), default=float("inf")), None),  # non-finite
    ],
)
def test_rust_default_expr_matches_field_type(field: NamedTypeSchema, expected: str | None) -> None:
    assert rust_codegen._default_expr(field) == expected


def _dtype_default_info() -> ObjectInfo:
    """Build a native root type with a DLDataType field carrying a string default."""
    return ObjectInfo(
        fields=[NamedTypeSchema("dtype", TypeSchema("dtype"), default="")],
        methods=[],
        type_key="cpp_rust_test.Typed",
        parent_type_key="ffi.Object",
        has_init=True,
    )


def test_rust_dtype_string_default_blocks_native(capsys: pytest.CaptureFixture[str]) -> None:
    # C2: a `DLDataType` field with a registered string default must NOT render as
    # `tvm_ffi::String::from(...)` (an E0308 mismatch). The default has no Rust
    # literal, so it blocks native construction with a clear warning instead.
    text, _ = _gen_rust_object(_dtype_default_info())
    assert "pub dtype: DLDataType," in text  # the field type still renders
    assert "tvm_ffi::String::from" not in text  # the broken String default is gone
    assert "ffi_new" not in text  # native construction blocked
    assert "TypedBuilder" not in text
    out = capsys.readouterr().out
    assert "[Warning] object cpp_rust_test.Typed: skipping `ffi_new`" in out
    assert "'dtype'" in out


def _native_narrow_info() -> ObjectInfo:
    """Root auto-init `Pixel`: narrow scalar fields (int32/int8/float) + an int method.

    Field schemas carry reflection's ``sizeof(T)`` so the renderer can emit the
    width-correct ``#[repr(C)]`` field types; the method's ``int`` stays
    schema-erased (no size) and must keep the packed-``Any`` default ``i64``.
    """
    return ObjectInfo(
        fields=[
            NamedTypeSchema("x", TypeSchema("int"), size=4),
            NamedTypeSchema("flag", TypeSchema("int"), size=1),
            NamedTypeSchema("weight", TypeSchema("float"), size=4),
            NamedTypeSchema("big", TypeSchema("int"), size=8),
            NamedTypeSchema("ratio", TypeSchema("float"), size=4),
        ],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    "get_x",
                    TypeSchema("Callable", (TypeSchema("int"), TypeSchema("cpp_rust_test.Pixel"))),
                ),
                is_member=True,
            )
        ],
        type_key="cpp_rust_test.Pixel",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo("x", NamedTypeSchema("x", TypeSchema("int"), size=4), False, False),
        ],
        has_init=True,
        mutable=True,
    )


def test_rust_scalar_fields_width_narrowed() -> None:
    text, _ = _gen_rust_object(_native_narrow_info())
    # Struct fields are laid out directly -> width-correct primitives by `size`.
    assert "pub x: i32," in text
    assert "pub flag: i8," in text
    assert "pub weight: f32," in text
    assert "pub big: i64," in text
    # The builder setters bind straight into the struct -> same widths.
    assert "pub fn ffi_new() -> PixelBuilder {" in text
    assert "pub fn x(mut self, x: i32) -> Self {" in text
    assert "pub fn flag(mut self, flag: i8) -> Self {" in text
    assert "pub fn weight(mut self, weight: f32) -> Self {" in text
    assert "pub fn big(mut self, big: i64) -> Self {" in text
    assert "    ratio: Option<f32>," in text
    # Method args/returns travel as packed Any (v_int64) -> stay i64.
    assert "pub fn get_x(&mut self) -> Result<i64> {" in text


def _scrambled_layout_info(*, gap: bool = False) -> ObjectInfo:
    """Fields REGISTERED out of memory order: beta@24, gamma@32, alpha@16.

    Declaration (memory) order is ``alpha: i32 @16, beta: i64 @24 (4 bytes of
    padding), gamma: i32 @32`` -- ``#[repr(C)]`` reproduces exactly this layout
    when the fields are emitted by offset. With ``gap=True``, ``gamma`` moves to
    offset 40 (as if an unregistered C++ member sat at 32..40), which no
    ``#[repr(C)]`` ordering can reproduce -> the offset warning must fire.
    """
    return ObjectInfo(
        fields=[
            NamedTypeSchema("beta", TypeSchema("int"), size=8, offset=24),
            NamedTypeSchema("gamma", TypeSchema("int"), size=4, offset=40 if gap else 32),
            NamedTypeSchema("alpha", TypeSchema("int"), size=4, offset=16),
        ],
        methods=[],
        type_key="cpp_rust_test.Scrambled",
        parent_type_key="ffi.Object",
    )


def test_rust_struct_fields_sorted_by_offset(capsys: pytest.CaptureFixture[str]) -> None:
    text, _ = _gen_rust_object(_scrambled_layout_info())
    # The struct lays fields out positionally -> memory (offset) order, not
    # registration order.
    alpha, beta, gamma = (text.index(f"pub {n}:") for n in ("alpha", "beta", "gamma"))
    assert alpha < beta < gamma
    # The repr(C) layout (with its natural alignment padding after `alpha`)
    # matches the recorded offsets -> no warning.
    assert "[Warning]" not in capsys.readouterr().out


def test_rust_struct_offset_gap_warns(capsys: pytest.CaptureFixture[str]) -> None:
    text, _ = _gen_rust_object(_scrambled_layout_info(gap=True))
    # The binding is still emitted (warning, not an error) ...
    assert "pub struct ScrambledObj {" in text
    # ... but the unreproducible hole at 32..40 is reported: repr(C) places
    # `gamma` right after `beta` (offset 32), reflection says 40.
    out = capsys.readouterr().out
    assert "[Warning] object cpp_rust_test.Scrambled" in out
    assert "'gamma' is at C++ offset 40" in out
    assert "places it at offset 32" in out


def test_rust_offset_check_resumes_after_unverifiable_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A field without size metadata is skipped (not an early bail-out): the field
    # right after it has no known predecessor end, but checking resumes one field
    # later -- the hole before `d` must still be reported.
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("a", TypeSchema("int"), size=4, offset=16),
            NamedTypeSchema("b", TypeSchema("int"), offset=20),  # no size -> unverifiable
            NamedTypeSchema("c", TypeSchema("int"), size=4, offset=24),
            NamedTypeSchema("d", TypeSchema("int"), size=4, offset=48),  # repr(C) says 28
        ],
        methods=[],
        type_key="cpp_rust_test.Holey",
        parent_type_key="ffi.Object",
    )
    _gen_rust_object(info)
    out = capsys.readouterr().out
    assert "'d' is at C++ offset 48" in out
    assert "places it at offset 28" in out


@pytest.mark.parametrize("init_arity", [2, 1])
def test_rust_native_explicit_init_stays_native(init_arity: int) -> None:
    # Native eligibility ignores the explicit `refl::init<...>` method entirely:
    # whether its arity matches the field count (2) or not (1, the
    # `Circle(radius)` derive shape), `ffi_new` binds the own fields with no FFI
    # `__ffi_init__` dispatch. A user who needs the faithful C++ ctor semantics
    # hand-writes a `new` (outside the markers) over the builder.
    info = _native_point_info()
    args = (TypeSchema("cpp_rust_test.Point"),) + (TypeSchema("int"),) * init_arity
    info.methods = [
        FuncInfo(
            NamedTypeSchema("__ffi_init__", TypeSchema("Callable", args)),
            is_member=False,
        )
    ]
    text, _ = _gen_rust_object(info)
    assert "pub fn ffi_new() -> PointBuilder {" in text
    assert "data: ObjectArc::new(self.build_obj()?)," in text
    assert "__ffi_init__" not in text


def test_rust_optional_method_arg_renders_native_option() -> None:
    # `Optional<T>` in a method signature is the native `Option<T>` (the
    # layout-mirror is only for direct struct fields).
    info = _native_point_info()
    info.methods = [
        FuncInfo(
            NamedTypeSchema(
                "lookup",
                TypeSchema(
                    "Callable",
                    (TypeSchema("int"), TypeSchema("Optional", (TypeSchema("int"),))),
                ),
            ),
            is_member=False,
        )
    ]
    text, _ = _gen_rust_object(info)
    assert "pub fn lookup(_0: Option<i64>) -> Result<i64> {" in text


def _optional_holder_info(*, mutable: bool = True) -> ObjectInfo:
    """Root `Holder` with one Optional field per layout category.

    Fields carry the reflected ``size``/``alignment`` the layout-mirror needs
    (the ``ffi.Object`` header is 24 bytes; offsets are kept self-consistent so
    the offset check stays quiet).
    """
    return ObjectInfo(
        fields=[
            # Category C (std::optional<scalar>): align comes from reflection.
            NamedTypeSchema(
                "a", TypeSchema("Optional", (TypeSchema("int"),)), size=16, alignment=8, offset=24
            ),
            # Category B (String): a nullable embedded Any.
            NamedTypeSchema(
                "s", TypeSchema("Optional", (TypeSchema("str"),)), size=16, alignment=8, offset=40
            ),
            # Category C with a narrower scalar -> alignment 4 (NOT 8).
            NamedTypeSchema(
                "n", TypeSchema("Optional", (TypeSchema("int"),)), size=8, alignment=4, offset=56
            ),
            # Category A but Any-bearing inner -> () marker, no typed accessor.
            NamedTypeSchema(
                "arr",
                TypeSchema("Optional", (TypeSchema("Array", (TypeSchema("Any"),)),)),
                size=8,
                alignment=8,
                offset=64,
            ),
        ],
        methods=[],
        type_key="cpp_rust_test.Holder",
        parent_type_key="ffi.Object",
        has_init=True,
        mutable=mutable,
    )


def test_rust_optional_field_renders_layout_mirror(capsys: pytest.CaptureFixture[str]) -> None:
    text, _ = _gen_rust_object(_optional_holder_info())
    # size/alignment come from reflection -> Optional<T, AlignK, N>.
    assert "pub a: Optional<i64, Align8, 16>," in text
    assert "pub s: Optional<String, Align8, 16>," in text
    assert "pub n: Optional<i64, Align4, 8>," in text  # narrower scalar -> Align4
    # Any-bearing inner is not a valid standalone type -> () marker.
    assert "pub arr: Optional<(), Align8, 8>," in text
    # get/set accessors marshal a native Option<V> (mutable class -> both), with
    # a per-call-site OnceLock so the field lookup is cached, not re-scanned.
    assert "pub fn a(&self) -> Result<Option<i64>> {" in text
    assert (
        "static CELL: std::sync::OnceLock<tvm_ffi::FieldAccess> = std::sync::OnceLock::new();"
        in text
    )
    assert 'self.a.read_cached::<i64>(&CELL, HolderObj::type_index(), "a")' in text
    assert "pub fn set_a(&self, value: Option<i64>) -> Result<()> {" in text
    assert 'self.a.write_cached::<i64>(&CELL, HolderObj::type_index(), "a", value)' in text
    # Any inner -> field emitted but NO typed accessor (would need AnyCompatible).
    assert "pub fn arr(" not in text
    # A direct Optional field is view-only -> the type is not natively built.
    assert "pub fn ffi_new()" not in text
    out = capsys.readouterr().out
    assert "is an ffi::Optional" in out
    # The Any-inner field is skipped with a diagnostic, not silently.
    assert "optional field 'arr' has no typed accessor" in out


def test_rust_optional_field_immutable_getter_only() -> None:
    text, _ = _gen_rust_object(_optional_holder_info(mutable=False))
    assert "pub fn a(&self) -> Result<Option<i64>> {" in text
    assert "pub fn set_a(" not in text  # immutable class -> read-only accessor
    assert "impl DerefMut" not in text


def test_rust_optional_object_inner_field_skips_accessor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # `Optional<ObjectRef>` (bare base object, origin `ffi.Object` -> tvm_ffi::Object)
    # is NOT AnyCompatible, so a `read::<Object>` accessor would not compile. The
    # field is still emitted (with a `()` marker) but the accessor is skipped + warned.
    info = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "obj",
                TypeSchema("Optional", (TypeSchema("ffi.Object"),)),
                size=8,
                alignment=8,
                offset=24,
            ),
        ],
        methods=[],
        type_key="cpp_rust_test.OptObjHolder",
        parent_type_key="ffi.Object",
        has_init=True,
        mutable=True,
    )
    text, _ = _gen_rust_object(info)
    assert "pub obj: Optional<(), Align8, 8>," in text  # () marker, NOT Object
    assert "pub fn obj(" not in text  # no accessor -> no `read::<Object>` compile break
    assert "optional field 'obj' has no typed accessor" in capsys.readouterr().out


def test_rust_optional_object_method_arg_is_unsupported() -> None:
    # `Optional<ObjectRef>` in a method signature would render `Option<Object>`,
    # which is not AnyCompatible; render must raise so the object is skipped rather
    # than emit non-compiling bindings.
    info = _native_point_info()
    info.methods = [
        FuncInfo(
            NamedTypeSchema(
                "f",
                TypeSchema(
                    "Callable",
                    (TypeSchema("int"), TypeSchema("Optional", (TypeSchema("ffi.Object"),))),
                ),
            ),
            is_member=False,
        )
    ]
    with pytest.raises(UnsupportedTypeError) as exc:
        _gen_rust_object(info)
    assert exc.value.origin == "Optional"


def test_rust_optional_field_unsupported_alignment_skips() -> None:
    # An ffi::Optional alignment with no marker (not in {1,2,4,8,16}) is skipped.
    info = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "a", TypeSchema("Optional", (TypeSchema("int"),)), size=32, alignment=32, offset=24
            )
        ],
        methods=[],
        type_key="cpp_rust_test.Weird",
        parent_type_key="ffi.Object",
        has_init=True,
        mutable=True,
    )
    with pytest.raises(UnsupportedTypeError) as exc:
        _gen_rust_object(info)
    assert exc.value.origin == "Optional"


def _native_point3d_info() -> ObjectInfo:
    """Build the derived `Point3D : Point` fixture: own init field `z` (x / y on the parent)."""
    return ObjectInfo(
        fields=[NamedTypeSchema("z", TypeSchema("int"))],
        methods=[],
        type_key="cpp_rust_test.Point3D",
        parent_type_key="cpp_rust_test.Point",
        init_fields=[
            InitFieldInfo("x", NamedTypeSchema("x", TypeSchema("int")), False, False),
            InitFieldInfo("y", NamedTypeSchema("y", TypeSchema("int")), False, False),
            InitFieldInfo("z", NamedTypeSchema("z", TypeSchema("int")), False, False),
        ],
        has_init=True,
    )


def _patch_native_point_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for type-key resolution: just the Point / Point3D fixture pair."""
    fixtures = {
        "cpp_rust_test.Point": _native_point_info,
        "cpp_rust_test.Point3D": _native_point3d_info,
    }
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key]())


def _native_attrs_info() -> ObjectInfo:
    """Build a native root parent in module `ir` (the cross-module base for the child)."""
    return ObjectInfo(
        fields=[NamedTypeSchema("span", TypeSchema("int"))],
        methods=[],
        type_key="ir.Attrs",
        has_init=True,
    )


def _native_relax_foo_info() -> ObjectInfo:
    """Build a native derived `relax.Foo` whose parent `ir.Attrs` is cross-module."""
    return ObjectInfo(
        fields=[NamedTypeSchema("k", TypeSchema("int"))],
        methods=[],
        type_key="relax.Foo",
        parent_type_key="ir.Attrs",
        has_init=True,
    )


def test_rust_cross_module_base_obj_is_imported() -> None:
    # A2: a derived type embeds its parent as `base: <Parent>Obj`. When the parent
    # lives in another generated module, the value type `<Parent>Obj` must be
    # imported (crate::-rooted), else the struct / Deref / builder won't compile.
    info = ObjectInfo(
        fields=[],
        methods=[],
        type_key="relax.Foo",
        parent_type_key="ir.Attrs",
    )
    text, imports = _gen_rust_object(info)
    assert "    base: AttrsObj," in text  # parent Obj embedded by leaf name
    assert "    type Target = AttrsObj;" in text  # Deref target is the base Obj
    uses = {u.as_use_line() for u in imports.items}
    assert "use crate::ir::AttrsObj;" in uses
    assert "use ir::AttrsObj;" not in uses  # never the bare (extern-prelude) form


def test_rust_native_derived_cross_module_imports_base_and_parent_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A2: a native derived type whose parent lives in another module must import
    # BOTH the embedded `<Parent>Obj` (struct field / builder store + setter) AND
    # the parent ref type `<Parent>` (used by the unset-base default-construction
    # fallback in `build_obj`). Both must be crate::-rooted.
    fixtures = {"ir.Attrs": _native_attrs_info, "relax.Foo": _native_relax_foo_info}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key]())
    text, imports = _gen_rust_object(_native_relax_foo_info())
    assert "    base: Option<AttrsObj>," in text  # builder stores the parent Obj
    assert "pub fn base(mut self, base: AttrsObj) -> Self {" in text
    assert "None => Attrs::ffi_new().build_obj()" in text  # parent ref in the fallback
    uses = {u.as_use_line() for u in imports.items}
    assert "use crate::ir::AttrsObj;" in uses  # embedded base value type
    assert "use crate::ir::Attrs;" in uses  # parent ref type


def test_rust_same_module_base_obj_filtered_from_imports() -> None:
    # A2: a parent in the SAME file is embedded by leaf name; its recorded
    # `<Parent>Obj` import must be filtered out alongside the ref type (the import
    # section drops both `crate::..::Expr` and its `..ExprObj` twin), so neither
    # collides with the local `Expr` / `ExprObj` definitions.
    block = _rust_import_block()
    imports = RustImports(items=[RustUse("cpp_rust_test.ExprObj"), RustUse("cpp_rust_test.Expr")])
    generate_rust_import_section(
        block, imports, Options(), defined_types={"crate::cpp_rust_test::Expr"}
    )
    assert block.lines == [
        "// tvm-ffi-stubgen(begin): import-section",
        "// tvm-ffi-stubgen(end)",
    ]


def test_rust_native_derived_base_setter(monkeypatch: pytest.MonkeyPatch) -> None:
    # A derived native type does NOT flatten ancestor fields, and `ffi_new` is
    # nullary like everywhere else: `base` is a consuming setter (uniform API)
    # taking the parent's bare struct value from its builder's `build_obj`.
    _patch_native_point_registry(monkeypatch)
    text, _ = _gen_rust_object(_native_point3d_info())
    assert "pub fn ffi_new() -> Point3DBuilder {" in text
    assert "base: None," in text  # the builder opens with base unset
    assert "base: Option<PointObj>," in text
    assert "pub fn base(mut self, base: PointObj) -> Self {" in text
    assert "pub fn z(mut self, z: i64) -> Self {" in text
    assert "pub fn build(self) -> Result<Point3D> {" in text
    # An unset base default-constructs the parent through its builder, with a
    # re-contextualized error (the parent's bare message names a foreign field).
    assert "None => Point::ffi_new().build_obj().map_err(|e| tvm_ffi::Error::new(" in text
    assert "field `base` is not set and default `Point` construction failed: {}" in text
    assert "data: ObjectArc::new(self.build_obj()?)," in text
    assert "Ok(Point3DObj {" in text
    # `build_obj` ships on every builder (a grandchild's `base` source).
    assert "pub fn build_obj(self) -> Result<Point3DObj> {" in text
    # No flattened ancestor setters, no FFI dispatch.
    assert "pub fn x(" not in text
    assert "pub fn y(" not in text
    assert "__ffi_init__" not in text


def test_rust_field_named_base_renames_synthetic_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Y: a derived type with its OWN reflected field `base` (e.g. tirx.Ramp,
    # arith.IterSumExpr) must not collide with the synthetic parent-embed (E0124).
    # The embed renames to `_base` (derive(Object) reads the first field
    # positionally); the real `base` field keeps its name. Builder setters/inits,
    # the Deref target, the base-resolve fallback, and the obj-literal all follow.
    parent = ObjectInfo(
        fields=[NamedTypeSchema("v", TypeSchema("int"))],
        methods=[],
        type_key="cpp_rust_test.PExpr",
        has_init=True,
    )
    child = ObjectInfo(
        fields=[NamedTypeSchema("base", TypeSchema("int"))],  # real C++ field named `base`
        methods=[],
        type_key="cpp_rust_test.Ramp",
        parent_type_key="cpp_rust_test.PExpr",
        has_init=True,
    )
    fixtures = {"cpp_rust_test.PExpr": parent, "cpp_rust_test.Ramp": child}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    text, _ = _gen_rust_object(child)
    # struct: synthetic embed renamed (first field), real `base` field kept
    assert "struct RampObj {" in text
    assert "    _base: PExprObj," in text  # synthetic parent embed renamed
    assert "    pub base: i64," in text  # the real C++ field keeps `base`
    # Deref targets the renamed embed
    assert "&self._base" in text
    # builder: two distinct stores + setters, no `base`/`base` collision
    assert "    _base: Option<PExprObj>," in text
    assert "    base: Option<i64>," in text
    assert "pub fn _base(mut self, _base: PExprObj) -> Self {" in text  # parent setter
    assert "pub fn base(mut self, base: i64) -> Self {" in text  # real-field setter
    assert "self._base = Some(_base);" in text
    assert "self.base = Some(base);" in text
    # build_obj: base fallback + obj-literal use the renamed embed
    assert "let _base = match self._base {" in text
    assert "None => PExpr::ffi_new().build_obj()" in text
    assert "let base = self.base.ok_or_else(" in text  # the real field, unwrapped
    assert "Ok(RampObj {" in text
    assert "        _base," in text  # obj-literal shorthand: the embed local
    assert "        base," in text  # obj-literal shorthand: the real field local


def _three_level_fixtures() -> dict[str, ObjectInfo]:
    """Build a native 3-level chain GP -> P -> C, each with one distinct own field."""
    return {
        "cpp_rust_test.GP": ObjectInfo(
            fields=[NamedTypeSchema("a", TypeSchema("int"))],
            methods=[],
            type_key="cpp_rust_test.GP",
            has_init=True,
        ),
        "cpp_rust_test.P": ObjectInfo(
            fields=[NamedTypeSchema("b", TypeSchema("int"))],
            methods=[],
            type_key="cpp_rust_test.P",
            parent_type_key="cpp_rust_test.GP",
            has_init=True,
        ),
        "cpp_rust_test.C": ObjectInfo(
            fields=[NamedTypeSchema("c", TypeSchema("int"))],
            methods=[],
            type_key="cpp_rust_test.C",
            parent_type_key="cpp_rust_test.P",
            has_init=True,
        ),
    }


def test_rust_multilevel_inheritance_no_field_flattening(monkeypatch: pytest.MonkeyPatch) -> None:
    # C3: each level's `Obj` struct and its `build_obj` literal use ONLY that
    # level's own fields (ancestor fields live in the embedded `base`), so a child
    # never assigns an ancestor field into a parent `Obj`. The E0560
    # (`RelaxFrameObj has no field is_func`) the feedback saw is an A4 same-leaf
    # name-collision artifact, not a flattening bug in the generator.
    fixtures = _three_level_fixtures()
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])

    text_c, _ = _gen_rust_object(fixtures["cpp_rust_test.C"])
    assert "struct CObj {" in text_c
    assert "    base: PObj," in text_c  # parent embedded, not flattened
    assert "    pub c: i64," in text_c
    assert "pub a:" not in text_c and "pub b:" not in text_c  # no ancestor fields
    assert "None => P::ffi_new().build_obj()" in text_c  # base via the parent's builder
    assert "Ok(CObj {" in text_c  # the literal targets C's own Obj ...
    assert "let c = self.c.ok_or_else(" in text_c  # ... binding only its own field
    assert "let a " not in text_c and "let b " not in text_c

    text_p, _ = _gen_rust_object(fixtures["cpp_rust_test.P"])
    assert "    base: GPObj," in text_p
    assert "    pub b: i64," in text_p
    assert "pub a:" not in text_p  # grandparent field stays in the embedded base
    assert "Ok(PObj {" in text_p
    assert "let b = self.b.ok_or_else(" in text_p


def test_rust_skipped_base_skips_descendant_transitively(monkeypatch: pytest.MonkeyPatch) -> None:
    # D: a base with an unsupported field (here a `Dict`) is itself skipped, so its
    # `<Base>Obj` is never emitted. A subclass embedding `base: <Base>Obj` would not
    # compile, so it is skipped transitively naming the unbuildable base.
    base = ObjectInfo(
        fields=[NamedTypeSchema("m", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))))],
        methods=[],
        type_key="cpp_rust_test.Base",
        parent_type_key="ffi.Object",
    )
    child = ObjectInfo(
        fields=[NamedTypeSchema("x", TypeSchema("int"))],
        methods=[],
        type_key="cpp_rust_test.Child",
        parent_type_key="cpp_rust_test.Base",
    )
    fixtures = {"cpp_rust_test.Base": base, "cpp_rust_test.Child": child}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    # The base itself is unsupported (the `Dict` field has no Rust rendering).
    with pytest.raises(UnsupportedTypeError) as base_exc:
        _gen_rust_object(base)
    assert base_exc.value.origin == "Dict"
    # The child is skipped transitively, the error naming the unbuildable base.
    with pytest.raises(UnsupportedTypeError) as child_exc:
        _gen_rust_object(child)
    assert child_exc.value.origin == "cpp_rust_test.Base"


def test_rust_skipped_grandparent_skips_whole_subtree(monkeypatch: pytest.MonkeyPatch) -> None:
    # D: an unbuildable ancestor propagates down the whole chain. Root has an
    # unsupported field; Mid renders on its own but embeds RootObj; Leaf embeds
    # MidObj. Both Mid and Leaf are skipped (each naming its immediate parent).
    root = ObjectInfo(
        fields=[NamedTypeSchema("t", TypeSchema("tuple", (TypeSchema("int"), TypeSchema("int"))))],
        methods=[],
        type_key="cpp_rust_test.Root",
        parent_type_key="ffi.Object",
    )
    mid = ObjectInfo(
        fields=[NamedTypeSchema("b", TypeSchema("int"))],
        methods=[],
        type_key="cpp_rust_test.Mid",
        parent_type_key="cpp_rust_test.Root",
    )
    leaf = ObjectInfo(
        fields=[NamedTypeSchema("c", TypeSchema("int"))],
        methods=[],
        type_key="cpp_rust_test.Leaf",
        parent_type_key="cpp_rust_test.Mid",
    )
    fixtures = {k.type_key: k for k in (root, mid, leaf)}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    with pytest.raises(UnsupportedTypeError) as mid_exc:  # Mid skipped: Root unrenderable
        _gen_rust_object(mid)
    assert mid_exc.value.origin == "cpp_rust_test.Root"
    with pytest.raises(UnsupportedTypeError) as leaf_exc:  # Leaf skipped: Root up the chain
        _gen_rust_object(leaf)
    assert leaf_exc.value.origin == "cpp_rust_test.Mid"


def test_rust_unresolvable_base_is_not_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # D (lenient): a base that cannot be RESOLVED from the registry is not treated
    # as skipped -- a descendant is dropped only when an ancestor is *provably*
    # unrenderable. The derived type still generates (non-native, since the parent
    # chain also cannot be proven constructible).
    def boom(key: str) -> ObjectInfo:
        raise KeyError(key)

    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", boom)
    info = ObjectInfo(
        fields=[NamedTypeSchema("x", TypeSchema("int"))],
        methods=[],
        type_key="cpp_rust_test.Child",
        parent_type_key="cpp_rust_test.Missing",
    )
    text, _ = _gen_rust_object(info)
    assert "struct ChildObj {" in text  # still generated, not skipped
    assert "base: MissingObj," in text  # references the parent by name
    assert "ffi_new" not in text  # non-native: parent not provably constructible


def test_rust_unmapped_ffi_base_is_skipped() -> None:
    # X: a type whose parent is an unmapped `ffi.*` builtin (e.g. `ffi.Enum`) has
    # no crate `<X>Obj` to embed, so it is skipped -- not emitted with a dangling
    # `use tvm_ffi::EnumObj;` base import.
    info = ObjectInfo(
        fields=[NamedTypeSchema("x", TypeSchema("int"))],
        methods=[],
        type_key="testing.MyEnum",
        parent_type_key="ffi.Enum",
    )
    with pytest.raises(UnsupportedTypeError) as exc:
        _gen_rust_object(info)
    assert exc.value.origin == "ffi.Enum"


def test_rust_field_referencing_skipped_type_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # X: a field whose type was itself skipped (here `S` has a `Dict` field) must
    # not leave a dangling `use`/field type. The referencing type is skipped, and
    # no import for the skipped type is recorded (the raise precedes the record).
    skipped = ObjectInfo(
        fields=[NamedTypeSchema("m", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))))],
        methods=[],
        type_key="ir.S",
        parent_type_key="ffi.Object",
    )
    user = ObjectInfo(
        fields=[NamedTypeSchema("s", TypeSchema("ir.S"))],
        methods=[],
        type_key="relax.User",
        parent_type_key="ffi.Object",
    )
    fixtures = {"ir.S": skipped, "relax.User": user}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    imports = RustImports()
    block = _rust_object_block("relax.User")
    with pytest.raises(UnsupportedTypeError) as exc:
        generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), user)
    assert exc.value.origin == "ir.S"
    # the skipped type's import was never recorded (raise before record)
    assert not any(u.path == "crate::ir::S" for u in imports.items)


def test_rust_cyclic_type_references_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    # X: two mutually-referencing types must not loop in the buildability probe
    # (`check_refs` is disabled during the dry-render, breaking the recursion).
    # Both are buildable, so both generate, each referencing the other by name.
    a = ObjectInfo(
        fields=[NamedTypeSchema("b", TypeSchema("m.B"))],
        methods=[],
        type_key="m.A",
        parent_type_key="ffi.Object",
    )
    b = ObjectInfo(
        fields=[NamedTypeSchema("a", TypeSchema("m.A"))],
        methods=[],
        type_key="m.B",
        parent_type_key="ffi.Object",
    )
    fixtures = {"m.A": a, "m.B": b}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    text_a, imports_a = _gen_rust_object(a)
    assert "    pub b: B," in text_a
    assert "use crate::m::B;" in {u.as_use_line() for u in imports_a.items}


def test_rust_multi_hop_skip_propagates_to_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Round 3: `Mid` own-renders but references the SKIPPED `Leaf` (Dict field), so
    # `Mid` is skipped (1-hop). Reference-aware buildability propagates that: a
    # `Consumer` whose field is `Mid` is skipped too -- it must NOT emit a dangling
    # `use crate::ir::Mid;` for a type the crate never generates (the round-3 bug).
    leaf = ObjectInfo(
        fields=[NamedTypeSchema("m", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))))],
        methods=[],
        type_key="ir.Leaf",
        parent_type_key="ffi.Object",
    )
    mid = ObjectInfo(
        fields=[NamedTypeSchema("leaf", TypeSchema("ir.Leaf"))],  # references the skipped Leaf
        methods=[],
        type_key="ir.Mid",
        parent_type_key="ffi.Object",
    )
    consumer = ObjectInfo(
        fields=[NamedTypeSchema("mid", TypeSchema("ir.Mid"))],  # 2 hops from Leaf
        methods=[],
        type_key="relax.Consumer",
        parent_type_key="ffi.Object",
    )
    fixtures = {"ir.Leaf": leaf, "ir.Mid": mid, "relax.Consumer": consumer}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    with pytest.raises(UnsupportedTypeError):  # Leaf: own unsupported field
        _gen_rust_object(leaf)
    with pytest.raises(UnsupportedTypeError) as mid_exc:  # Mid: 1-hop ref to skipped Leaf
        _gen_rust_object(mid)
    assert mid_exc.value.origin == "ir.Leaf"
    imports = RustImports()
    block = _rust_object_block("relax.Consumer")
    with pytest.raises(UnsupportedTypeError) as cons_exc:  # Consumer: 2-hop, reference-aware
        generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), consumer)
    assert cons_exc.value.origin == "ir.Mid"
    assert not any(u.path == "crate::ir::Mid" for u in imports.items)  # no dangling import


def test_rust_cycle_referencing_skipped_type_skips_all(monkeypatch: pytest.MonkeyPatch) -> None:
    # Round 3 (greatest-fixpoint): a reference cycle A<->B where A *also* references
    # a SKIPPED `Leaf` must mark BOTH A and B unbuildable -- the cycle must not
    # rescue them. (Naive memoization of the verdict would wrongly cache B=True.)
    leaf = ObjectInfo(
        fields=[NamedTypeSchema("m", TypeSchema("List", (TypeSchema("int"),)))],
        methods=[],
        type_key="m.Leaf",
        parent_type_key="ffi.Object",
    )
    a = ObjectInfo(
        fields=[
            NamedTypeSchema("b", TypeSchema("m.B")),
            NamedTypeSchema("l", TypeSchema("m.Leaf")),
        ],
        methods=[],
        type_key="m.A",
        parent_type_key="ffi.Object",
    )
    b = ObjectInfo(
        fields=[NamedTypeSchema("a", TypeSchema("m.A"))],
        methods=[],
        type_key="m.B",
        parent_type_key="ffi.Object",
    )
    fixtures = {"m.Leaf": leaf, "m.A": a, "m.B": b}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    ty = RC.RUST_TY_MAP_DEFAULTS.copy()
    assert rust_codegen._base_buildable("m.A", ty) is False
    assert rust_codegen._base_buildable("m.B", ty) is False  # the cycle does not rescue B


def test_rust_callable_field_referencing_skipped_type_does_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Round 3 (false-skip guard): `Callable` is type-erased -- it renders as the
    # opaque `tvm_ffi::Function`, ignoring its signature -- so a field typed
    # `Callable[[Skipped]]` references nothing. The owner must STILL generate
    # (buildability must not over-collect Callable args).
    leaf = ObjectInfo(
        fields=[NamedTypeSchema("m", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))))],
        methods=[],
        type_key="ir.Leaf",
        parent_type_key="ffi.Object",
    )
    owner = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "cb", TypeSchema("Callable", (TypeSchema("None"), TypeSchema("ir.Leaf")))
            ),
        ],
        methods=[],
        type_key="relax.Owner",
        parent_type_key="ffi.Object",
    )
    fixtures = {"ir.Leaf": leaf, "relax.Owner": owner}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    assert rust_codegen._base_buildable("relax.Owner", RC.RUST_TY_MAP_DEFAULTS.copy()) is True
    text, _ = _gen_rust_object(owner)
    assert "struct OwnerObj {" in text
    assert "    pub cb: Function," in text  # type-erased; no reference to Leaf


def test_rust_method_arg_referencing_skipped_type_skips_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Round 3 (companion to the Callable guard): a reflected method's signature IS
    # rendered by `_method_fn`, so a method param referencing a SKIPPED type must
    # still skip the owner -- and reference-aware buildability must agree, so the
    # owner's consumers cascade-skip. Confirms method refs survive the Callable fix.
    leaf = ObjectInfo(
        fields=[NamedTypeSchema("m", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))))],
        methods=[],
        type_key="ir.Leaf",
        parent_type_key="ffi.Object",
    )
    owner = ObjectInfo(
        fields=[NamedTypeSchema("x", TypeSchema("int"))],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    "use_leaf", TypeSchema("Callable", (TypeSchema("None"), TypeSchema("ir.Leaf")))
                ),
                is_member=False,
            )
        ],
        type_key="relax.Owner2",
        parent_type_key="ffi.Object",
    )
    fixtures = {"ir.Leaf": leaf, "relax.Owner2": owner}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    with pytest.raises(UnsupportedTypeError):  # method param renders the skipped Leaf
        _gen_rust_object(owner)
    assert rust_codegen._base_buildable("relax.Owner2", RC.RUST_TY_MAP_DEFAULTS.copy()) is False


def test_rust_object_root_struct_and_impl() -> None:
    text, imports = _gen_rust_object(_expr_info())
    # data struct embeds the root Object as `base`
    assert "#[repr(C)]" in text
    assert "struct ExprObj {" in text
    assert "    base: Object," in text
    assert "    pub value: i64," in text
    # ObjectCore impl is folded into the `#[derive(Object)]` proc macro: the stub
    # only emits the derive + `#[type_key]` attr, not a hand-written impl.
    assert "#[derive(tvm_ffi::derive::Object)]" in text
    assert '#[type_key = "cpp_rust_test.Expr"]' in text
    assert "unsafe impl ObjectCore" not in text
    assert "lookup_type_index" not in text
    assert "object_header_mut" not in text
    # ref + Deref/DerefMut (value is def_rw -> mutable class)
    assert "#[derive(tvm_ffi::derive::ObjectRef, Clone)]" in text
    assert "struct Expr {" in text
    assert "    data: ObjectArc<ExprObj>," in text
    assert "impl Deref for Expr {" in text
    assert "impl DerefMut for Expr {" in text
    # native ffi_new (root, field-binding init): opens the builder; `build`
    # allocates. generated types/functions are `pub` (decision Q2)
    assert "pub struct ExprObj {" in text
    assert "pub struct Expr {" in text
    assert "pub fn ffi_new() -> ExprBuilder {" in text
    assert "pub fn value(mut self, value: i64) -> Self {" in text
    assert "pub struct ExprBuilder {" in text
    assert "pub fn build(self) -> Result<Expr> {" in text
    assert "pub fn test() -> Result<i64> {" in text
    assert "data: ObjectArc::new(self.build_obj()?)," in text
    assert "Ok(ExprObj {" in text
    assert "base: Object::new()," in text
    assert "__ffi_init__" not in text
    # static method: no self; uniform packed-call convention with cached getter
    assert "thread_local!(static F: std::cell::OnceCell<tvm_ffi::Function>" in text
    assert (
        "let f = tvm_ffi::Function::from_type_method_cached(&F, "
        'ExprObj::type_index(), "test")?;' in text
    )
    assert "Ok(f.call_packed(&[])?.try_into()?)" in text
    uses = {u.as_use_line() for u in imports.items}
    assert "use tvm_ffi::Object;" in uses
    assert "use std::ops::DerefMut;" in uses


def test_rust_object_derived_embeds_parent() -> None:
    text, _ = _gen_rust_object(_add_info())
    assert "struct AddObj {" in text
    assert "    base: ExprObj," in text  # parent Obj embedded, not Object
    assert "    pub a: Expr," in text
    # object_header_mut is derived by the `#[derive(Object)]` macro from the
    # first field (`base: ExprObj`), so the stub no longer hand-writes it.
    assert "object_header_mut" not in text
    # derived Obj also derefs to its embedded base
    assert "impl Deref for AddObj {" in text
    assert "    type Target = ExprObj;" in text
    # instance method: &mut self receiver (mutable class); self is packed as `&*self`
    assert "fn update(&mut self) -> Result<()> {" in text
    assert "Ok(f.call_packed(&[AnyView::from(&*self)])?.try_into()?)" in text
    # The parent type key is not resolvable from the live registry -> the chain
    # cannot be proven native and there is no FFI fallback: no ctor at all.
    assert "ffi_new" not in text
    assert "AddBuilder" not in text


def test_rust_object_immutable_has_no_derefmut() -> None:
    text, _ = _gen_rust_object(_expr_info(mutable=False))  # _type_mutable=false
    assert "impl Deref for Expr {" in text
    assert "DerefMut" not in text
    assert "fn test() -> Result<i64> {" in text  # static unaffected


def test_rust_object_field_of_type_object_shares_boilerplate_use() -> None:
    # Regression for an E0252 "Object defined multiple times" collision: a root
    # object's boilerplate `Object` (the struct `base`) and a field whose type
    # is itself `ffi.Object` both record the crate-root re-export path
    # `tvm_ffi::Object`, so they dedup to a single `use` -- no second
    # `use ...::Object;` that would fail to compile.
    info = ObjectInfo(
        fields=[NamedTypeSchema("child", TypeSchema("ffi.Object"))],
        methods=[],
        type_key="demo.Holder",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)
    assert "    base: Object," in text  # boilerplate Object as the struct base
    assert "    pub child: Object," in text  # the field binds the same type
    uses = [u.as_use_line() for u in imports.items]
    assert uses.count("use tvm_ffi::Object;") == 1


def test_rust_cross_module_type_use_is_crate_rooted() -> None:
    # A1: a field whose type lives in another generated module must be imported
    # via a `crate::`-rooted path (the tree is mounted at the crate root). A bare
    # `use ir::Attrs;` would resolve against the extern prelude and fail to
    # compile; the body still references the type by its leaf name.
    info = ObjectInfo(
        fields=[NamedTypeSchema("attrs", TypeSchema("ir.Attrs"))],
        methods=[],
        type_key="tirx.Foo",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)
    assert "    pub attrs: Attrs," in text  # referenced by leaf in the body
    uses = {u.as_use_line() for u in imports.items}
    assert "use crate::ir::Attrs;" in uses
    assert "use ir::Attrs;" not in uses  # never the bare (extern-prelude) form


def test_rust_method_any_return_stays_any_not_anyview() -> None:
    # Q5: a top-level `Any` *return* stays owning `Any` (a borrow has no lifetime
    # source coming back out of an FFI call); only top-level `Any` *params* become
    # the non-owning `AnyView`. Regression for return type being rendered as AnyView.
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("int"))],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    # Callable(return=Any, self=Self, param=Any)
                    "probe",
                    TypeSchema(
                        "Callable",
                        (TypeSchema("Any"), TypeSchema("demo.Boxed"), TypeSchema("Any")),
                    ),
                ),
                is_member=True,
            )
        ],
        type_key="demo.Boxed",
        parent_type_key="ffi.Object",
        mutable=True,
    )
    text, imports = _gen_rust_object(info)
    # return -> owning Any; param -> non-owning AnyView
    assert "pub fn probe(&mut self, _0: AnyView) -> Result<Any> {" in text
    assert "Result<AnyView>" not in text  # the bug would have produced this
    # All methods use the uniform `call_packed` convention (which natively speaks
    # `AnyView` args and an `Any` return -- the only convention that can). An
    # `Any` return is forwarded directly, with no trailing `try_into`.
    assert "into_typed_fn!" not in text
    assert "f.call_packed(&[AnyView::from(&*self), _0])" in text
    # owning Any return must record its `use`
    assert RustUse("tvm_ffi::Any") in imports.items
    assert RustUse("tvm_ffi::AnyView") in imports.items


def _has_dict_info() -> ObjectInfo:
    # A `Dict` field is genuinely unsupported (no Rust type at all), so this stays
    # a skip fixture for the generic skip-handling tests below. (An untyped
    # `Map<Any, Any>` field now renders -- see ``test_rust_map_any_field_renders``
    # -- because as a field a `Map` is a single pointer with phantom params.)
    return ObjectInfo(
        fields=[
            NamedTypeSchema("cfg", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int")))),
        ],
        methods=[],
        type_key="demo.HasDict",
        parent_type_key="ffi.Object",
    )


def test_rust_object_unsupported_raises() -> None:
    # `generate_rust_object` propagates UnsupportedTypeError (cli catches it and
    # resets the block). Boilerplate `use`s recorded before the raise may stay
    # behind in the collector -- harmless, generated files open with
    # `#![allow(unused_imports)]`.
    block = _rust_object_block("demo.HasDict")
    imports = RustImports(items=[RustUse("tvm_ffi::Tensor")])
    with pytest.raises(UnsupportedTypeError) as exc:
        generate_rust_object(
            block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), _has_dict_info()
        )
    assert exc.value.origin == "Dict"
    assert RustUse("tvm_ffi::Tensor") in imports.items  # pre-seeded use kept


def test_rust_stage3_skipped_type_not_counted_as_defined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A skipped object must not enter `defined_types`: another object referencing
    # it keeps its `use` in the import section (previously dropped -> E0412).
    rs = tmp_path / "demo.rs"
    rs.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} import-section",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/demo.HasDict",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/demo.Holder",
                C.RUST_SYNTAX.end,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    infos = {
        "demo.HasDict": _has_dict_info(),
        "demo.Holder": ObjectInfo(
            fields=[NamedTypeSchema("child", TypeSchema("demo.HasDict"))],
            methods=[],
            type_key="demo.Holder",
            parent_type_key="ffi.Object",
        ),
    }
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", lambda key: infos[key])
    info = FileInfo.from_file(rs)
    assert info is not None
    _stage_3(
        info,
        Options(dry_run=True),
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        {},
        generator=RustGenerator(),
    )
    text = "\n".join(info.lines)
    assert "[Skipped] object demo.HasDict" in capsys.readouterr().out
    assert "struct HasDictObj" not in text  # skipped block reset to bare markers
    assert "    pub child: HasDict," in text  # the referencing object still renders
    assert "use crate::demo::HasDict;" in text  # ... and keeps its (crate::-rooted) import


def test_rust_stage3_skipped_base_skips_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # D end-to-end: a base with an unsupported field (`List`) is skipped, and its
    # subclass is skipped transitively -- NOT emitted referencing a missing
    # `BaseObj` (which would be E0412). Neither struct lands in the output.
    rs = tmp_path / "demo.rs"
    rs.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} import-section",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/demo.Base",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/demo.Child",
                C.RUST_SYNTAX.end,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    infos = {
        "demo.Base": ObjectInfo(
            fields=[NamedTypeSchema("m", TypeSchema("List", (TypeSchema("int"),)))],
            methods=[],
            type_key="demo.Base",
            parent_type_key="ffi.Object",
        ),
        "demo.Child": ObjectInfo(
            fields=[NamedTypeSchema("x", TypeSchema("int"))],
            methods=[],
            type_key="demo.Child",
            parent_type_key="demo.Base",
        ),
    }
    # `_stage_3` resolves via stub_cli; `_base_buildable`/`_native_eligible` via rust_codegen.
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", lambda key: infos[key])
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: infos[key])
    info = FileInfo.from_file(rs)
    assert info is not None
    _stage_3(
        info,
        Options(dry_run=True),
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        {},
        generator=RustGenerator(),
    )
    text = "\n".join(info.lines)
    out = capsys.readouterr().out
    assert "[Skipped] object demo.Base" in out
    assert "[Skipped] object demo.Child" in out  # transitive skip
    assert "struct BaseObj" not in text
    assert "struct ChildObj" not in text  # not emitted referencing a missing BaseObj


def test_rust_stage3_field_ref_to_skipped_type_drops_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # X end-to-end: `S` is skipped (Dict field); `User` has a field of type `S`.
    # `User` is skipped too -- NOT emitted with a dangling `pub s: S` and a
    # `use crate::demo::S;` import that points at a type the crate never defines.
    rs = tmp_path / "demo.rs"
    rs.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} import-section",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/demo.S",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/demo.User",
                C.RUST_SYNTAX.end,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    infos = {
        "demo.S": ObjectInfo(
            fields=[
                NamedTypeSchema("m", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))))
            ],
            methods=[],
            type_key="demo.S",
            parent_type_key="ffi.Object",
        ),
        "demo.User": ObjectInfo(
            fields=[NamedTypeSchema("s", TypeSchema("demo.S"))],
            methods=[],
            type_key="demo.User",
            parent_type_key="ffi.Object",
        ),
    }
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", lambda key: infos[key])
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: infos[key])
    info = FileInfo.from_file(rs)
    assert info is not None
    _stage_3(
        info,
        Options(dry_run=True),
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        {},
        generator=RustGenerator(),
    )
    text = "\n".join(info.lines)
    out = capsys.readouterr().out
    assert "[Skipped] object demo.S" in out
    assert "[Skipped] object demo.User" in out  # referencing type skipped too
    assert "struct UserObj" not in text
    assert "use crate::demo::S;" not in text  # no dangling import for the skipped type


def test_rust_stage3_multi_hop_skip_drops_consumer_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Round 3 end-to-end (the reported dangling-import bug across modules):
    # `Consumer` (relax) has a field of cross-module `ir.Mid`; `Mid` references the
    # skipped `ir.Leaf` (Dict field). Reference-aware buildability skips `Consumer`
    # and drops the `use crate::ir::Mid;` import for a type the crate never emits.
    rs = tmp_path / "relax.rs"
    rs.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} import-section",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/relax.Consumer",
                C.RUST_SYNTAX.end,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    infos = {
        "relax.Consumer": ObjectInfo(
            fields=[NamedTypeSchema("mid", TypeSchema("ir.Mid"))],
            methods=[],
            type_key="relax.Consumer",
            parent_type_key="ffi.Object",
        ),
        "ir.Mid": ObjectInfo(
            fields=[NamedTypeSchema("leaf", TypeSchema("ir.Leaf"))],
            methods=[],
            type_key="ir.Mid",
            parent_type_key="ffi.Object",
        ),
        "ir.Leaf": ObjectInfo(
            fields=[
                NamedTypeSchema("m", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))))
            ],
            methods=[],
            type_key="ir.Leaf",
            parent_type_key="ffi.Object",
        ),
    }
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", lambda key: infos[key])
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: infos[key])
    info = FileInfo.from_file(rs)
    assert info is not None
    _stage_3(
        info,
        Options(dry_run=True),
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        {},
        generator=RustGenerator(),
    )
    text = "\n".join(info.lines)
    out = capsys.readouterr().out
    assert "[Skipped] object relax.Consumer" in out  # skipped via the 2-hop chain
    assert "struct ConsumerObj" not in text
    assert "use crate::ir::Mid;" not in text  # no dangling import for the skipped Mid


def test_rust_bytes_field_maps_to_crate_bytes() -> None:
    # C++ `Bytes` fields carry the schema origin "bytes" (string.h TypeStr).
    info = ObjectInfo(
        fields=[NamedTypeSchema("payload", TypeSchema("bytes"))],
        methods=[],
        type_key="demo.Blob",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)
    assert "    pub payload: Bytes," in text
    assert RustUse("tvm_ffi::Bytes") in imports.items


def test_rust_map_field_renders() -> None:
    # A typed `Map<K, V>` field renders as the crate's read-only `Map` (a single
    # pointer, layout-compatible with C++ `Map : ObjectRef`).
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("cfg", TypeSchema("Map", (TypeSchema("str"), TypeSchema("int")))),
        ],
        methods=[],
        type_key="demo.Config",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)
    assert "    pub cfg: Map<String, i64>," in text
    assert RustUse("tvm_ffi::Map") in imports.items


def test_rust_map_any_field_renders() -> None:
    # Keystone recovery: a `Map<str, Any>` (or untyped `Map<Any, Any>`) FIELD is a
    # single pointer with phantom params, so it renders even though `Any` is not
    # AnyCompatible -- only a typed accessor (not emitted for Map) would need it.
    # In arg/return/nested position the guard still rejects it (those must marshal).
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("d", TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any")))),
            NamedTypeSchema("u", TypeSchema("Map")),  # untyped -> Map<Any, Any>
        ],
        methods=[],
        type_key="ir.DictAttrs",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)
    assert "    pub d: Map<String, Any>," in text
    assert "    pub u: Map<Any, Any>," in text
    assert RustUse("tvm_ffi::Map") in imports.items
    assert RustUse("tvm_ffi::Any") in imports.items
    assert "[Skipped]" not in text  # the object is NOT skipped


def test_rust_map_value_genuinely_unsupported_still_skips() -> None:
    # The relaxation is only for phantom-OK params: a `Map` value that is not a
    # valid Rust type at all (here a `Dict`) still skips the object.
    info = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "bad",
                TypeSchema(
                    "Map",
                    (TypeSchema("str"), TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int")))),
                ),
            ),
        ],
        methods=[],
        type_key="ir.Bad",
        parent_type_key="ffi.Object",
    )
    with pytest.raises(UnsupportedTypeError) as exc:
        _gen_rust_object(info)
    assert exc.value.origin == "Dict"


def test_rust_map_any_field_unskips_cascade(monkeypatch: pytest.MonkeyPatch) -> None:
    # The keystone cascade: `DictAttrs.__dict__ : Map<str, Any>` no longer skips
    # `DictAttrs`, so reference-aware buildability marks it (and `BaseFunc`, which
    # references it) buildable -- the consumers that round 3 cascade-skipped now
    # come back.
    dict_attrs = ObjectInfo(
        fields=[NamedTypeSchema("d", TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any"))))],
        methods=[],
        type_key="ir.DictAttrs",
        parent_type_key="ffi.Object",
    )
    base_func = ObjectInfo(
        fields=[NamedTypeSchema("attrs", TypeSchema("ir.DictAttrs"))],
        methods=[],
        type_key="ir.BaseFunc",
        parent_type_key="ffi.Object",
    )
    fixtures = {"ir.DictAttrs": dict_attrs, "ir.BaseFunc": base_func}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: fixtures[key])
    ty = RC.RUST_TY_MAP_DEFAULTS.copy()
    assert rust_codegen._base_buildable("ir.DictAttrs", ty) is True
    assert rust_codegen._base_buildable("ir.BaseFunc", ty) is True  # cascade recovered


def test_rust_unknown_bare_origin_skips_object() -> None:
    # An unmapped bare origin (no `.`) has no Rust rendering; emitting it
    # verbatim would be invalid source, so the object is skipped instead.
    info = ObjectInfo(
        fields=[NamedTypeSchema("name", TypeSchema("const char*"))],
        methods=[],
        type_key="demo.Raw",
        parent_type_key="ffi.Object",
    )
    with pytest.raises(UnsupportedTypeError) as exc:
        _gen_rust_object(info)
    assert exc.value.origin == "const char*"


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
            RustUse("crate_b::Foo"),
        ]
    )
    generate_rust_import_section(block, imports, Options(), defined_types=set())
    assert block.lines == [
        "// tvm-ffi-stubgen(begin): import-section",
        "use crate_b::Foo;",
        "use tvm_ffi::Tensor;",
        "use tvm_ffi::object::ObjectArc;",
        "// tvm-ffi-stubgen(end)",
    ]


def test_rust_import_section_filters_defined_types() -> None:
    block = _rust_import_block()
    imports = RustImports(items=[RustUse("cpp_rust_test.Expr"), RustUse("tvm_ffi::Tensor")])
    # Expr is defined in this file -> its (crate::-rooted) `use` must be dropped.
    generate_rust_import_section(
        block, imports, Options(), defined_types={"crate::cpp_rust_test::Expr"}
    )
    assert block.lines == [
        "// tvm-ffi-stubgen(begin): import-section",
        "use tvm_ffi::Tensor;",
        "// tvm-ffi-stubgen(end)",
    ]


def test_rust_generator_wired() -> None:
    gen = get_generator("rust")
    assert isinstance(gen, RustGenerator)
    imp = gen.new_imports()
    assert isinstance(imp, RustImports)
    gen.add_imported_object(imp, "cpp_rust_test.Expr", "False", "")
    assert imp.items == [RustUse("cpp_rust_test.Expr")]  # normalized to crate::-rooted
    assert gen.canonical_type_name("cpp_rust_test.Expr") == "crate::cpp_rust_test::Expr"
    assert gen.extra_export_names(imp) == set()
    # object block delegates to generate_rust_object
    block = _rust_object_block("cpp_rust_test.Expr")
    gen.generate_object_block(
        block, RC.RUST_TY_MAP_DEFAULTS.copy(), gen.new_imports(), Options(), _expr_info()
    )
    assert "struct ExprObj {" in "\n".join(block.lines)
    # all/export blocks are no-ops (deferred); must not raise
    gen.generate_all_block(_rust_object_block("x"), {"Foo"}, Options())
    gen.generate_export_block(_rust_object_block("x"))


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
        generator=RustGenerator(),
    )
    text = "\n".join(info.lines)
    # object block filled (native ffi_new: root field-binding fixture)
    assert "struct ExprObj {" in text
    assert "impl Expr {" in text
    assert "data: ObjectArc::new(self.build_obj()?)," in text
    # import-section filled with the machinery `use`s
    assert "use tvm_ffi::ObjectArc;" in text
    assert "use tvm_ffi::ObjectCore;" in text
    # Expr defines itself -> no self `use`
    assert "use cpp_rust_test::Expr;" not in text


def test_rust_stage3_cross_module_name_collision_full_qualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A4 end-to-end: a file defining `tirx.Call` and a sibling `tirx.Holder` whose
    # field type is the cross-module `relax.expr.Call` (same leaf `Call`). The
    # collision is resolved by full-qualifying the cross-module reference inline;
    # no conflicting `use` is emitted next to the local `Call` definition.
    rs = tmp_path / "tirx.rs"
    rs.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} import-section",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/tirx.Call",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/tirx.Holder",
                C.RUST_SYNTAX.end,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    infos = {
        "tirx.Call": ObjectInfo(
            fields=[], methods=[], type_key="tirx.Call", parent_type_key="ffi.Object"
        ),
        "tirx.Holder": ObjectInfo(
            fields=[NamedTypeSchema("call", TypeSchema("relax.expr.Call"))],
            methods=[],
            type_key="tirx.Holder",
            parent_type_key="ffi.Object",
        ),
    }
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", lambda key: infos[key])
    info = FileInfo.from_file(rs)
    assert info is not None
    _stage_3(
        info,
        Options(dry_run=True),
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        {},
        generator=RustGenerator(),
    )
    text = "\n".join(info.lines)
    assert "struct CallObj {" in text  # the local `Call` is defined here
    # the cross-module `Call` is referenced by full path, not imported under `Call`
    assert "pub call: crate::relax::expr::Call," in text
    assert "use crate::relax::expr::Call;" not in text


def _keyword_field_info() -> ObjectInfo:
    """Build a native root type with a Rust-keyword field name `mod` (+ a normal field)."""
    return ObjectInfo(
        fields=[
            NamedTypeSchema("mod", TypeSchema("int")),
            NamedTypeSchema("value", TypeSchema("int")),
        ],
        methods=[],
        type_key="cpp_rust_test.KwHolder",
        has_init=True,
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mod", "r#mod"),
        ("impl", "r#impl"),
        ("type", "r#type"),
        ("async", "r#async"),  # reserved keyword, still rejected as a bare ident
        ("self", "self_"),  # `r#` cannot spell it -> suffix rename
        ("Self", "Self_"),
        ("crate", "crate_"),
        ("super", "super_"),
        ("value", "value"),  # ordinary name unchanged
        ("set_mod", "set_mod"),  # not a keyword
    ],
)
def test_rust_ident_escapes_keywords(name: str, expected: str) -> None:
    assert rust_codegen._rust_ident(name) == expected


def test_rust_keyword_field_name_is_raw_identifier() -> None:
    # B: a C++ field named `mod` (a Rust keyword) is emitted as `r#mod` in every
    # bare-identifier position (struct, builder store/setter, ffi_new, unwrap,
    # obj-literal), while the FFI field-name string and error text keep `mod`.
    text, _ = _gen_rust_object(_keyword_field_info())
    assert "pub r#mod: i64," in text  # struct field
    assert "pub value: i64," in text  # ordinary field unescaped
    assert "r#mod: None," in text  # ffi_new builder-literal init
    assert "r#mod: Option<i64>," in text  # builder store
    assert "pub fn r#mod(mut self, r#mod: i64) -> Self {" in text  # builder setter
    assert "self.r#mod = Some(r#mod);" in text
    assert "let r#mod = self.r#mod.ok_or_else(" in text  # build_obj unwrap
    assert "`mod` is not set" in text  # error text keeps the C++ name
    assert "    r#mod," in text  # obj-literal shorthand binds the local
    # the ordinary `value` field still gets a plain (unescaped) setter
    assert "pub fn value(mut self, value: i64) -> Self {" in text


def test_rust_keyword_method_name_is_raw_identifier() -> None:
    # B: a reflected method named `move` (a keyword) becomes `pub fn r#move`, but
    # the FFI lookup string stays `move`.
    info = _native_point_info()
    info.methods = [
        FuncInfo(
            NamedTypeSchema("move", TypeSchema("Callable", (TypeSchema("int"),))),
            is_member=False,
        )
    ]
    text, _ = _gen_rust_object(info)
    assert "pub fn r#move(" in text
    assert '"move")?;' in text  # the FFI method-name string is unescaped


def test_rust_keyword_optional_accessor_is_raw_identifier() -> None:
    # B: an Optional field named `impl` yields `pub fn r#impl` / `pub fn set_impl`
    # accessing `self.r#impl`, while the reflection field name stays `impl`.
    info = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "impl",
                TypeSchema("Optional", (TypeSchema("int"),)),
                size=16,
                alignment=8,
                offset=24,
            )
        ],
        methods=[],
        type_key="cpp_rust_test.KwOpt",
        parent_type_key="ffi.Object",
        has_init=True,
        mutable=True,
    )
    text, _ = _gen_rust_object(info)
    assert "pub r#impl: " in text  # struct field is escaped
    assert "pub fn r#impl(&self) -> Result<Option<i64>> {" in text
    assert "self.r#impl.read_cached::<i64>(&CELL, " in text
    assert '"impl") }' in text  # FFI field name unescaped in the getter
    assert "pub fn set_impl(&self, value: Option<i64>) -> Result<()> {" in text
    assert "self.r#impl.write_cached::<i64>(&CELL, " in text


def test_rust_default_ty_map_is_real() -> None:
    # Regression: default_ty_map must be the real table, not an empty placeholder.
    m = RustGenerator().default_ty_map()
    assert m["int"] == "i64"
    assert m["None"] == "()"


def test_rust_api_filenames() -> None:
    gen = RustGenerator()
    assert gen.api_filename() == "mod.rs"
    assert gen.init_filename() == "mod.rs"
    assert gen.generate_init_file([], "demo", "mod") == ""


def test_rust_api_file_scaffold() -> None:
    text = RustGenerator().generate_api_file(
        [],
        {},
        "demo",
        [_expr_info()],
        InitConfig("p", "l", "demo."),
        is_root=True,
    )
    assert "#![allow(dead_code, unused_imports)]" in text
    assert f"{C.RUST_SYNTAX.begin} import-section" in text
    assert f"{C.RUST_SYNTAX.begin} object/cpp_rust_test.Expr" in text
    # method lookup lives in the crate (`Function::from_type_method_cached`);
    # the scaffold carries no per-file helper block or support code.
    assert "helpers" not in text
    assert "fn get_type_method" not in text
    # no global / __all__ / export markers for Rust
    assert "global/" not in text
    assert "__all__" not in text
    assert "export/" not in text


def test_rust_finalize_module_tree(tmp_path: Path) -> None:
    # Two sibling binding modules under `a`, plus an intermediate `a` with no types.
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "mod.rs").write_text("// bindings b\n", encoding="utf-8")
    (tmp_path / "a" / "c").mkdir(parents=True)
    (tmp_path / "a" / "c" / "mod.rs").write_text("// bindings c\n", encoding="utf-8")

    finalize_rust_module_tree(tmp_path, {"a.b", "a.c"})

    # root declares the top-level module; `a/mod.rs` (created) declares its children
    assert "pub mod a;" in (tmp_path / "mod.rs").read_text(encoding="utf-8")
    a_mod = (tmp_path / "a" / "mod.rs").read_text(encoding="utf-8")
    assert "pub mod b;" in a_mod and "pub mod c;" in a_mod
    # leaf binding files are untouched
    assert "// bindings b" in (tmp_path / "a" / "b" / "mod.rs").read_text(encoding="utf-8")

    # idempotent: re-running adds no duplicates
    finalize_rust_module_tree(tmp_path, {"a.b", "a.c"})
    assert (tmp_path / "a" / "mod.rs").read_text(encoding="utf-8").count("pub mod b;") == 1


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
    RustGenerator().generate_global_funcs_block(
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
