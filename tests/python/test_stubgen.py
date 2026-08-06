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

import itertools
import os
import shutil
import subprocess
import typing
from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
from tvm_ffi import Object, method
from tvm_ffi.core import TypeSchema
from tvm_ffi.dataclasses import py_class
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.cli import _stage_2, _stage_3
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo, collect_files
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
    render_object_ffi_init,
    render_object_fields,
    render_object_init,
    render_object_methods,
)
from tvm_ffi.stub.python_generator.utils import ImportItem
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

_counter = itertools.count()


def _identity_ty_map(name: str) -> str:
    return name


def _unique_type_key(base: str) -> str:
    return f"testing.stubgen.{base}_{next(_counter)}"


def _default_ty_map() -> dict[str, str]:
    return PC.TY_MAP_DEFAULTS.copy()


def _type_suffix(name: str) -> str:
    return PC.TY_MAP_DEFAULTS.get(name, name).rsplit(".", 1)[-1]


def _input_type_suffix(name: str) -> str:
    return PC.TY_MAP_INPUT_DEFAULTS.get(name, PC.TY_MAP_DEFAULTS.get(name, name)).rsplit(".", 1)[-1]


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


def test_objectinfo_gen_overloaded_static_methods() -> None:
    info = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo.from_schema(
                "demo.Factory.create",
                TypeSchema("Callable", (TypeSchema("demo.Factory"), TypeSchema("int"))),
                is_member=False,
            ),
            FuncInfo.from_schema(
                "demo.Factory.reset",
                TypeSchema("Callable", (TypeSchema("None"),)),
                is_member=False,
            ),
            FuncInfo.from_schema(
                "demo.Factory.create",
                TypeSchema("Callable", (TypeSchema("demo.Factory"), TypeSchema("str"))),
                is_member=False,
            ),
        ],
    )
    assert info.has_overloaded_methods()
    assert render_object_methods(info, _identity_ty_map, indent=2) == [
        "  @overload",
        "  @staticmethod",
        "  def create(_0: int, /) -> demo.Factory: ...",
        "  @overload",
        "  @staticmethod",
        "  def create(_0: str, /) -> demo.Factory: ...",
        "  @staticmethod",
        "  def reset() -> None: ...",
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


def test_funcinfo_gen_uses_input_annotations_for_parameters() -> None:
    info = FuncInfo(
        schema=NamedTypeSchema(
            "demo.echo_list",
            TypeSchema(
                "Callable",
                (
                    TypeSchema("List", (TypeSchema("int"),)),
                    TypeSchema("List", (TypeSchema("int"),)),
                ),
            ),
        ),
        is_member=False,
    )

    assert (
        render_func_signature(info, _type_suffix, indent=0, input_ty_map=_input_type_suffix)
        == "def echo_list(_0: Sequence[int], /) -> MutableSequence[int]: ..."
    )


def test_generate_global_funcs_populates_input_defaults_for_partial_ty_map() -> None:
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
                "demo.echo_list",
                TypeSchema(
                    "Callable",
                    (
                        TypeSchema("List", (TypeSchema("int"),)),
                        TypeSchema("List", (TypeSchema("int"),)),
                    ),
                ),
            ),
            is_member=False,
        )
    ]
    imports: list[ImportItem] = []

    generate_python_global_funcs(
        code, funcs, {"List": "collections.abc.MutableSequence"}, imports, Options()
    )

    assert code.lines == [
        f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg",
        "# fmt: off",
        '_FFI_INIT_FUNC("demo", __name__)',
        "if TYPE_CHECKING:",
        "    def echo_list(_0: Sequence[int], /) -> MutableSequence[int]: ...",
        "# fmt: on",
        C.PYTHON_SYNTAX.end,
    ]


def test_objectinfo_gen_init_uses_input_annotations() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("items", TypeSchema("List", (TypeSchema("int"),)))],
        methods=[],
        init_fields=[
            InitFieldInfo(
                name="items",
                schema=NamedTypeSchema("items", TypeSchema("List", (TypeSchema("int"),))),
                kw_only=False,
                has_default=False,
            )
        ],
        has_init=True,
    )

    assert render_object_fields(info, _type_suffix, indent=0) == ["items: MutableSequence[int]"]
    assert render_object_init(info, _type_suffix, indent=0, input_ty_map=_input_type_suffix) == [
        "def __init__(self, items: Sequence[int]) -> None: ..."
    ]
    assert render_object_ffi_init(
        info, _type_suffix, indent=0, input_ty_map=_input_type_suffix
    ) == ["def __ffi_init__(self, items: Sequence[int]) -> None: ..."]


def test_py_class_method_metadata_renders_stub_signature() -> None:
    @py_class(_unique_type_key("MethodMetadata"))
    class MethodMetadata(Object):
        value: int

        @method
        def describe(self, values: typing.List[int], prefix: str) -> str:  # noqa: UP006
            return f"{prefix}:{self.value}:{len(values)}"

        @method
        @staticmethod
        def normalize(values: typing.List[int]) -> typing.List[int]:  # noqa: UP006
            return values

    info = ObjectInfo.from_type_info(MethodMetadata.__tvm_ffi_type_info__)  # ty: ignore[unresolved-attribute]
    methods = {method.schema.name: method for method in info.methods}
    describe_schema = methods["describe"].schema

    assert describe_schema.origin == "Callable"
    assert [arg.origin for arg in describe_schema.args] == [
        "str",
        MethodMetadata.__tvm_ffi_type_info__.type_key,  # ty: ignore[unresolved-attribute]
        "List",
        "str",
    ]
    assert render_object_methods(info, _type_suffix, indent=0, input_ty_map=_input_type_suffix) == [
        "def describe(self, _1: Sequence[int], _2: str, /) -> str: ...",
        "@staticmethod",
        "def normalize(_0: Sequence[int], /) -> MutableSequence[int]: ...",
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


def test_generate_global_funcs_imports_enum_from_dataclasses() -> None:
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
                "demo.get_enum",
                TypeSchema("Callable", (TypeSchema("ffi.Enum"),)),
            ),
            is_member=False,
        )
    ]
    imports: list[ImportItem] = []

    generate_python_global_funcs(code, funcs, _default_ty_map(), imports, Options())

    assert ImportItem("tvm_ffi.dataclasses.Enum", type_checking_only=True) in imports
    assert "    def get_enum() -> Enum: ..." in code.lines


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
    # leading `ffi` segment rewritten via RUST_MOD_MAP, dots -> ::
    assert RustUse("ffi.String").path == "tvm_ffi::String"
    # unmapped crate prefix is preserved, dots still -> ::
    u = RustUse("my_pkg.sub.Foo")
    assert u.path == "my_pkg::sub::Foo"
    assert u.leaf == "Foo"
    assert u.as_use_line() == "use my_pkg::sub::Foo;"


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
    assert RustUse("::tvm_ffi::Array") in imports.items


def test_render_callable_is_function() -> None:
    text, imports = _rust_render(TypeSchema("Callable", (TypeSchema("int"),)))
    assert text == "Function"
    assert RustUse("::tvm_ffi::Function") in imports.items


def test_render_object_leaf_records_use() -> None:
    # Importing `tvm_ffi::String` shadows the prelude `String` in the generated
    # module; that is safe because the derive macros expand with fully
    # qualified `::std::string::String`.
    text, imports = _rust_render(TypeSchema("ffi.String"))
    assert text == "String"
    assert RustUse("::tvm_ffi::String") in imports.items


def test_render_nested() -> None:
    schema = TypeSchema("Array", (TypeSchema("Array", (TypeSchema("int"),)),))
    text, imports = _rust_render(schema)
    assert text == "Array<Array<i64>>"
    assert RustUse("::tvm_ffi::Array") in imports.items


@pytest.mark.parametrize(
    "schema",
    [
        TypeSchema("Union", (TypeSchema("int"), TypeSchema("str"))),
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


def test_render_map_typed() -> None:
    schema = TypeSchema("Map", (TypeSchema("str"), TypeSchema("int")))
    text, imports = _rust_render(schema)
    assert text == "Map<String, i64>"
    assert RustUse("::tvm_ffi::Map") in imports.items
    assert RustUse("::tvm_ffi::String") in imports.items


def test_render_optional_value_positions() -> None:
    # Reflected values render plain `Option<T>`; generated fields are accessed
    # through owning getter results rather than mirrored in-place layouts.
    assert (
        _rust_render(TypeSchema("Optional", (TypeSchema("int"),)))[0]
        == "::core::option::Option<i64>"
    )
    assert (
        _rust_render(TypeSchema("Optional", (TypeSchema("str"),)))[0]
        == "::core::option::Option<String>"
    )
    assert (
        _rust_render(TypeSchema("Optional", (TypeSchema("bytes"),)))[0]
        == "::core::option::Option<Bytes>"
    )
    text, imports = _rust_render(
        TypeSchema("Optional", (TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))),))
    )
    assert text == "::core::option::Option<Map<String, i64>>"
    assert RustUse("::tvm_ffi::Map") in imports.items
    # Nested inside an Array (elements are Any-encoded, so `Option<T>` is fine).
    text, _ = _rust_render(TypeSchema("Array", (TypeSchema("Optional", (TypeSchema("int"),)),)))
    assert text == "Array<::core::option::Option<i64>>"


@pytest.mark.parametrize(
    ("schema", "origin"),
    [
        # A genuinely unsupported origin buried inside a container still
        # bubbles up. The object renderer catches it and uses Any/AnyView.
        pytest.param(
            TypeSchema("Array", (TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))),)),
            "Dict",
            id="array-of-dict",
        ),
        pytest.param(
            TypeSchema("Map", (TypeSchema("str"), TypeSchema("List", (TypeSchema("int"),)))),
            "List",
            id="map-of-list",
        ),
        # NB: `void*` (`ctypes.c_void_p`) is rejected at leaf resolution
        # (`_ObjectRenderer._ty_render`), not by `render_rust_type` itself, so it
        # is covered by `test_rust_void_ptr_unsupported` (which uses the real
        # renderer), not this `_rust_render` double.
    ],
)
def test_render_unsupported_nested_raises(schema: TypeSchema, origin: str) -> None:
    with pytest.raises(UnsupportedTypeError) as exc:
        _rust_render(schema)
    assert exc.value.origin == origin


def test_ty_render_dedups_same_path() -> None:
    imports = RustImports()
    ty_map = RC.RUST_TY_MAP_DEFAULTS

    def tr(origin: str) -> str:
        return imports.record(ty_map.get(origin, origin))

    assert tr("Array") == "Array"
    assert tr("Array") == "Array"  # same path again -> reuse binding
    assert imports.items == [RustUse("::tvm_ffi::Array")]  # recorded exactly once


def test_ty_render_same_leaf_uses_qualified_fallback() -> None:
    imports = RustImports()
    assert imports.record("crate_a::Foo") == "Foo"  # first claims the bare leaf
    assert imports.record("crate_b::Foo") == "crate_b::Foo"
    assert imports.items == [RustUse("crate_a::Foo")]


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


def _expr_info() -> ObjectInfo:
    return ObjectInfo(
        fields=[
            NamedTypeSchema("value", TypeSchema("int")),
            NamedTypeSchema(
                "dynamic",
                TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any"))),
            ),
        ],
        methods=[
            FuncInfo(
                NamedTypeSchema("test", TypeSchema("Callable", (TypeSchema("int"),))),
                is_member=False,
            ),
            FuncInfo(
                NamedTypeSchema(
                    "probe",
                    TypeSchema(
                        "Callable",
                        (
                            TypeSchema("Any"),
                            TypeSchema("cpp_rust_test.Expr"),
                            TypeSchema("Any"),
                        ),
                    ),
                ),
                is_member=True,
            ),
        ],
        type_key="cpp_rust_test.Expr",
        parent_type_key="ffi.Object",
        has_init=True,
        mutable=True,
    )


def _add_info() -> ObjectInfo:
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
        has_init=True,
        mutable=True,
    )


def test_rust_object_is_opaque_and_uses_owning_getters() -> None:
    text, imports = _gen_rust_object(_expr_info())

    assert "pub struct ExprObj {\n    base: ::tvm_ffi::Object," in text
    assert "pub value:" not in text
    assert "impl ExprObj {" in text
    assert "pub fn value(&self) -> ::tvm_ffi::Result<i64> {" in text
    assert 'get_reflected_field_unchecked(self, "value")' in text
    assert "pub fn dynamic(&self) -> ::tvm_ffi::Result<::tvm_ffi::Any> {" in text
    assert 'get_reflected_field_unchecked(self, "dynamic")' in text
    assert "Map<String, ObjectRef>" not in text

    assert "pub fn test() -> ::tvm_ffi::Result<i64> {" in text
    assert (
        "pub fn probe(&self, _0: ::tvm_ffi::AnyView) -> ::tvm_ffi::Result<::tvm_ffi::Any> {" in text
    )
    assert "object_core_as_any_view(self)" in text
    assert "pub fn same_as<" not in text

    assert "pub fn ffi_new() -> ::tvm_ffi::Result<Self>" in text
    assert "cached_type_attr!" in text
    assert "Builder" not in text
    assert "ObjectArc::new" not in text
    assert "DerefMut" not in text
    assert "downcast" not in text
    assert imports.items == []


def test_rust_derived_object_inherits_parent_getters() -> None:
    text, _ = _gen_rust_object(_add_info())

    assert "pub struct AddObj {\n    base: ExprObj," in text
    assert "impl ::std::ops::Deref for AddObj {" in text
    assert "type Target = ExprObj;" in text
    assert "impl AddObj {" in text
    assert "pub fn a(&self) -> ::tvm_ffi::Result<Expr> {" in text
    assert "pub fn b(&self) -> ::tvm_ffi::Result<Expr> {" in text
    assert "pub fn update(&self) -> ::tvm_ffi::Result<()> {" in text
    assert "&mut self" not in text
    assert "impl ::core::convert::From<Add> for Expr {" in text
    assert (
        "::tvm_ffi::ObjectArc::from_raw("
        "::tvm_ffi::ObjectArc::into_raw(arc) as *const ExprObj)" in text
    )


def test_rust_cross_module_parent_and_keyword_field() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("impl", TypeSchema("int"))],
        methods=[],
        type_key="type.impl",
        parent_type_key="ir.Expr",
    )
    text, imports = _gen_rust_object(info)

    assert "pub struct r#impl {" in text
    assert "pub struct implObj {" in text
    assert "base: ExprObj," in text
    assert "pub fn r#impl(&self) -> ::tvm_ffi::Result<i64>" in text
    assert 'get_reflected_field_unchecked(self, "impl")' in text
    assert RustUse("super::ir::ExprObj") in imports.items
    assert RustUse("super::ir::Expr") in imports.items


def test_rust_typed_field_getters_use_value_types_not_layout_mirrors() -> None:
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("maybe", TypeSchema("Optional", (TypeSchema("str"),)), size=40),
            NamedTypeSchema("items", TypeSchema("Array", (TypeSchema("int"),))),
            NamedTypeSchema(
                "lookup",
                TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))),
            ),
            NamedTypeSchema("payload", TypeSchema("bytes")),
            NamedTypeSchema("child", TypeSchema("ffi.Object")),
        ],
        methods=[],
        type_key="demo.Fields",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)

    assert "pub fn maybe(&self) -> ::tvm_ffi::Result<::core::option::Option<String>> {" in text
    assert "pub fn items(&self) -> ::tvm_ffi::Result<Array<i64>> {" in text
    assert "pub fn lookup(&self) -> ::tvm_ffi::Result<Map<String, i64>> {" in text
    assert "pub fn payload(&self) -> ::tvm_ffi::Result<Bytes> {" in text
    assert "pub fn child(&self) -> ::tvm_ffi::Result<ObjectRef> {" in text
    assert "tvm_ffi::Optional" not in text
    assert RustUse("::tvm_ffi::Bytes") in imports.items
    assert RustUse("::tvm_ffi::object::ObjectRef") in imports.items


@pytest.mark.parametrize(
    "schema",
    [
        TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any"))),
        TypeSchema("Array", (TypeSchema("Any"),)),
        TypeSchema("Optional", (TypeSchema("Any"),)),
        TypeSchema("List", (TypeSchema("int"),)),
        TypeSchema("ctypes.c_void_p"),
        TypeSchema("const char*"),
    ],
)
def test_rust_dynamic_field_falls_back_to_any(schema: TypeSchema) -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", schema)],
        methods=[],
        type_key="demo.Dynamic",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)

    assert "pub struct DynamicObj {" in text
    assert "pub fn value(&self) -> ::tvm_ffi::Result<::tvm_ffi::Any> {" in text
    assert "try_into()" not in text
    assert imports.items == []


def test_rust_dynamic_method_uses_any_and_anyview() -> None:
    dynamic_map = TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any")))
    info = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    "rewrite",
                    TypeSchema(
                        "Callable",
                        (
                            dynamic_map,
                            TypeSchema("demo.Dynamic"),
                            TypeSchema("Array", (TypeSchema("Any"),)),
                        ),
                    ),
                ),
                is_member=True,
            )
        ],
        type_key="demo.Dynamic",
        parent_type_key="ffi.Object",
        mutable=True,
    )
    text, _ = _gen_rust_object(info)

    assert (
        "pub fn rewrite(&self, _0: ::tvm_ffi::AnyView) "
        "-> ::tvm_ffi::Result<::tvm_ffi::Any> {" in text
    )
    assert (
        "__tvm_ffi_func.call_packed("
        "&[::tvm_ffi::object::object_core_as_any_view(self), _0])" in text
    )
    assert "&mut self" not in text


def test_rust_explicit_and_auto_constructors_call_reflection() -> None:
    explicit = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    "__ffi_init__",
                    TypeSchema(
                        "Callable",
                        (TypeSchema("ffi.Object"), TypeSchema("int"), TypeSchema("str")),
                    ),
                ),
                is_member=False,
            )
        ],
        type_key="demo.Explicit",
        parent_type_key="ffi.Object",
        has_init=True,
    )
    text, _ = _gen_rust_object(explicit)
    assert "pub fn ffi_new(_0: i64, _1: String) -> ::tvm_ffi::Result<Self>" in text
    assert (
        "cached_type_method!(<ExplicitObj as ::tvm_ffi::ObjectCore>::type_index(), "
        '"__ffi_init__")' in text
    )

    auto = ObjectInfo(
        fields=[],
        methods=[],
        type_key="demo.Auto",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo(
                "required",
                NamedTypeSchema("required", TypeSchema("int")),
                kw_only=False,
                has_default=False,
            ),
            InitFieldInfo(
                "optional",
                NamedTypeSchema("optional", TypeSchema("str")),
                kw_only=False,
                has_default=True,
            ),
            InitFieldInfo(
                "keyword",
                NamedTypeSchema("keyword", TypeSchema("bool")),
                kw_only=True,
                has_default=False,
            ),
        ],
        has_init=True,
    )
    text, _ = _gen_rust_object(auto)
    assert (
        "pub fn ffi_new(required: i64, optional: String, keyword: bool) "
        "-> ::tvm_ffi::Result<Self>" in text
    )
    assert (
        "cached_type_attr!(<AutoObj as ::tvm_ffi::ObjectCore>::type_index(), "
        '"__ffi_init__")' in text
    )
    assert "__tvm_ffi_func.call_packed_with_kwargs" in text
    assert '("optional", ::tvm_ffi::AnyView::from(&optional))' in text
    assert '("keyword", ::tvm_ffi::AnyView::from(&keyword))' in text


def test_rust_auto_constructor_names_do_not_shadow_fields() -> None:
    names = ["f", "bad-name", "_1", "__tvm_ffi_func"]
    info = ObjectInfo(
        fields=[],
        methods=[],
        type_key="demo.Names",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo(name, NamedTypeSchema(name, TypeSchema("int")), False, False)
            for name in names
        ],
        has_init=True,
    )
    text, _ = _gen_rust_object(info)

    # The valid `_1` field keeps its name; the invalid field gets a collision-free fallback.
    assert "ffi_new(f: i64, __1: i64, _1: i64, __tvm_ffi_func: i64)" in text
    assert "let ___tvm_ffi_func =" in text
    for field, param in zip(names, ("f", "__1", "_1", "__tvm_ffi_func")):
        assert f'("{field}", ::tvm_ffi::AnyView::from(&{param}))' in text


def test_rust_duplicate_inherited_init_name_skips_constructor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    field = NamedTypeSchema("value", TypeSchema("int"))
    info = ObjectInfo(
        fields=[],
        methods=[],
        type_key="demo.Duplicate",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo("value", field, False, False),
            InitFieldInfo("value", field, True, True),
        ],
        has_init=True,
    )
    text, _ = _gen_rust_object(info)
    assert "fn ffi_new" not in text
    assert "duplicate names" in capsys.readouterr().out


def test_rust_overload_is_skipped_instead_of_emitting_duplicate_methods(
    capsys: pytest.CaptureFixture[str],
) -> None:
    methods = [
        FuncInfo(
            NamedTypeSchema("probe", TypeSchema("Callable", (TypeSchema("int"), argument))),
            is_member=False,
        )
        for argument in (TypeSchema("int"), TypeSchema("str"))
    ]
    info = ObjectInfo(
        fields=[],
        methods=methods,
        type_key="demo.Overloaded",
        parent_type_key="ffi.Object",
    )
    text, _ = _gen_rust_object(info)
    assert "pub fn probe" not in text
    assert "skipping overloaded Rust method" in capsys.readouterr().out


def test_rust_static_and_instance_methods_may_share_a_name() -> None:
    methods = [
        FuncInfo(
            NamedTypeSchema("probe", TypeSchema("Callable", (TypeSchema("int"),))),
            is_member=False,
        ),
        FuncInfo(
            NamedTypeSchema(
                "probe",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("demo.Both"))),
            ),
            is_member=True,
        ),
    ]
    text, _ = _gen_rust_object(
        ObjectInfo(
            fields=[],
            methods=methods,
            type_key="demo.Both",
            parent_type_key="ffi.Object",
        )
    )
    assert text.count("pub fn probe(") == 2


def test_rust_named_any_and_import_collisions_remain_typed() -> None:
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("named_any", TypeSchema("a.Any")),
            NamedTypeSchema("left", TypeSchema("a.Foo")),
            NamedTypeSchema("right", TypeSchema("b.Foo")),
        ],
        methods=[],
        type_key="demo.Holder",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)

    assert "pub fn named_any(&self) -> ::tvm_ffi::Result<Any>" in text
    assert 'get_reflected_field_unchecked(self, "named_any") }?.try_into()' in text
    assert "pub fn left(&self) -> ::tvm_ffi::Result<Foo>" in text
    assert "pub fn right(&self) -> ::tvm_ffi::Result<super::b::Foo>" in text
    assert RustUse("super::a::Any") in imports.items
    assert RustUse("super::a::Foo") in imports.items


def test_rust_invalid_accessor_does_not_drop_object(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("_", TypeSchema("int"))],
        methods=[],
        type_key="demo.Invalid",
        parent_type_key="ffi.Object",
    )
    text, _ = _gen_rust_object(info)
    assert "pub struct InvalidObj" in text
    assert "pub fn _(" not in text
    assert "skipping Rust field accessor" in capsys.readouterr().out


def test_rust_local_type_name_reserves_import_leaf() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("str"))],
        methods=[],
        type_key="demo.String",
        parent_type_key="ffi.Object",
    )
    block = _rust_object_block("demo.String")
    imports = RustImports()
    RustGenerator().reserve_defined_types(imports, {"demo::String"})
    generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), info)
    text = "\n".join(block.lines)
    assert "pub struct String" in text
    assert "::tvm_ffi::Result<::tvm_ffi::String>" in text
    assert RustUse("::tvm_ffi::String") not in imports.items


def test_rust_type_map_of_local_type_keeps_external_reference() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("other", TypeSchema("demo.Foo"))],
        methods=[],
        type_key="demo.Foo",
        parent_type_key="ffi.Object",
    )
    imports = RustImports()
    RustGenerator().reserve_defined_types(imports, {"demo::Foo"})
    object_block = _rust_object_block("demo.Foo")
    ty_map = RC.RUST_TY_MAP_DEFAULTS | {"demo.Foo": "external::Bar"}
    generate_rust_object(object_block, ty_map, imports, Options(), info)
    import_block = _rust_import_block()
    generate_rust_import_section(import_block, imports, Options(), defined_types={"external::Bar"})

    assert "::tvm_ffi::Result<Bar>" in "\n".join(object_block.lines)
    assert "use external::Bar;" in import_block.lines


def test_rust_type_key_attribute_is_escaped() -> None:
    info = ObjectInfo(
        fields=[],
        methods=[],
        type_key='de"mo.Valid',
        parent_type_key="ffi.Object",
    )
    text, _ = _gen_rust_object(info)
    assert '#[type_key = "de\\"mo.Valid"]' in text


@pytest.mark.parametrize(
    ("owner", "origin", "expected_use"),
    [
        ("tirx.Ramp", "ir.Expr", "super::ir::Expr"),
        ("tirx.transform.Pass", "ir.Expr", "super::super::ir::Expr"),
        ("tirx.transform.Pass", "tirx.Stmt", "super::super::tirx::Stmt"),
        ("tirx.Ramp", "tirx.Stmt", None),
        ("Root", "ir.Expr", "self::ir::Expr"),
    ],
)
def test_rust_generated_type_paths_are_rooted(
    owner: str, origin: str, expected_use: str | None
) -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema(origin))],
        methods=[],
        type_key=owner,
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)

    assert f"::tvm_ffi::Result<{origin.rsplit('.', 1)[-1]}>" in text
    paths = {item.path for item in imports.items}
    if expected_use is None:
        assert all(not path.endswith(f"::{origin.rsplit('.', 1)[-1]}") for path in paths)
    else:
        assert RustUse(expected_use) in imports.items


def test_rust_dotted_type_map_target_uses_generated_tree_root() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("legacy.Expr"))],
        methods=[],
        type_key="tirx.Use",
        parent_type_key="ffi.Object",
    )
    block = _rust_object_block("tirx.Use")
    imports = RustImports()
    ty_map = RC.RUST_TY_MAP_DEFAULTS | {"legacy.Expr": "ir.Expr"}
    generate_rust_object(block, ty_map, imports, Options(), info)
    assert RustUse("super::ir::Expr") in imports.items


def test_rust_unmapped_ffi_types_stay_in_generated_tree() -> None:
    enum = ObjectInfo(fields=[], methods=[], type_key="ffi.Enum", parent_type_key="ffi.Object")
    int_enum = ObjectInfo(fields=[], methods=[], type_key="ffi.IntEnum", parent_type_key="ffi.Enum")
    enum_text, _ = _gen_rust_object(enum)
    int_enum_text, imports = _gen_rust_object(int_enum)

    assert "pub struct EnumObj" in enum_text
    assert "base: EnumObj" in int_enum_text
    assert all("tvm_ffi::Enum" not in item.path for item in imports.items)


@pytest.mark.parametrize(
    "schema",
    [
        TypeSchema("Array", (TypeSchema("Any"),)),
        TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any"))),
        TypeSchema("Optional", (TypeSchema("Any"),)),
    ],
)
def test_render_dynamic_container_is_not_objectref(schema: TypeSchema) -> None:
    with pytest.raises(UnsupportedTypeError):
        _rust_render(schema)


def test_render_generic_object_container_uses_objectref() -> None:
    text, imports = _rust_render(TypeSchema("Array", (TypeSchema("Object"),)))
    assert text == "Array<ObjectRef>"
    assert RustUse("::tvm_ffi::object::ObjectRef") in imports.items


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


def test_rust_import_section_keeps_recorded_cross_module_types() -> None:
    block = _rust_import_block()
    imports = RustImports(items=[RustUse("cpp_rust_test::Expr"), RustUse("tvm_ffi::Tensor")])
    # Rust reserves local definitions before rendering; anything that reached
    # the collector is an actual use, even if a ty-map shares its canonical path.
    generate_rust_import_section(block, imports, Options(), defined_types={"cpp_rust_test::Expr"})
    assert block.lines == [
        "// tvm-ffi-stubgen(begin): import-section",
        "use cpp_rust_test::Expr;",
        "use tvm_ffi::Tensor;",
        "// tvm-ffi-stubgen(end)",
    ]


def test_rust_generator_wired() -> None:
    gen = get_generator("rust")
    assert isinstance(gen, RustGenerator)
    imp = gen.new_imports()
    assert isinstance(imp, RustImports)
    gen.add_imported_object(imp, "cpp_rust_test.Expr", "False", "")
    assert imp.items == [RustUse("cpp_rust_test::Expr")]
    assert gen.canonical_type_name("cpp_rust_test.Expr") == "cpp_rust_test::Expr"
    assert gen.canonical_type_name("::tvm_ffi::Module") == "::tvm_ffi::Module"
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
    # Object block is opaque; field access goes through reflection.
    assert "struct ExprObj {" in text
    assert "pub value:" not in text
    assert "pub fn value(&self) -> ::tvm_ffi::Result<i64>" in text
    assert "impl Expr {" in text
    assert "ObjectArc::new" not in text
    # Runtime support paths stay fully qualified, so the import block is empty.
    assert "use ::tvm_ffi" not in text
    # Expr defines itself -> no self `use`
    assert "use cpp_rust_test::Expr;" not in text


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_generated_rust_bindings_compile_against_runtime(tmp_path: Path) -> None:
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("value", TypeSchema("int")),
            NamedTypeSchema("dynamic", TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any")))),
        ],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    "rewrite",
                    TypeSchema(
                        "Callable",
                        (
                            TypeSchema("int"),
                            TypeSchema("compile.Probe"),
                            TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any"))),
                        ),
                    ),
                ),
                is_member=True,
            ),
        ],
        type_key="compile.Probe",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo(
                "value",
                NamedTypeSchema("value", TypeSchema("int")),
                kw_only=True,
                has_default=False,
            )
        ],
        has_init=True,
        mutable=True,
    )
    shadow_infos = [
        ObjectInfo(
            fields=[NamedTypeSchema("value", TypeSchema("str"))] if name == "String" else [],
            methods=[],
            type_key=f"compile.{name}",
            parent_type_key="ffi.Object",
        )
        for name in ("Result", "TryFrom", "ObjectRef", "String", "std", "tvm_ffi")
    ]
    infos = [info, *shadow_infos]
    imports = RustImports()
    generator = RustGenerator()
    generator.reserve_defined_types(
        imports, {generator.canonical_type_name(item.type_key or "") for item in infos}
    )
    generated_blocks: list[str] = []
    for item in infos:
        block = _rust_object_block(item.type_key or "")
        generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), item)
        generated_blocks.append("\n".join(block.lines))
    generated = "\n\n".join(generated_blocks)
    uses = "\n".join(sorted(item.as_use_line() for item in imports.items))

    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text(
        f"#![allow(dead_code, unused_imports)]\n{uses}\n\n{generated}\n",
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[2]
    runtime_path = (repo / "rust" / "tvm-ffi").as_posix()
    (tmp_path / "Cargo.toml").write_text(
        "\n".join(
            [
                "[package]",
                'name = "stubgen-compile-check"',
                'version = "0.0.0"',
                'edition = "2021"',
                "",
                "[dependencies]",
                f'tvm-ffi = {{ path = "{runtime_path}" }}',
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = f"{repo / '.venv' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["CARGO_TARGET_DIR"] = str(repo / "rust" / "target" / "stubgen-compile-check")
    result = subprocess.run(
        ["cargo", "check", "--quiet"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
    assert text.startswith("// FFI bindings for `demo`")
    assert "#![" not in text and "//!" not in text
    assert f"{C.RUST_SYNTAX.begin} import-section" in text
    assert f"{C.RUST_SYNTAX.begin} object/cpp_rust_test.Expr" in text
    # method lookup lives in the crate (`cached_type_method!`);
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

    prefixes = {"a.b", "a.c", "type.match"}
    finalize_rust_module_tree(tmp_path, prefixes)

    # root declares the top-level module; `a/mod.rs` (created) declares its children
    assert "pub mod a;" in (tmp_path / "mod.rs").read_text(encoding="utf-8")
    a_mod = (tmp_path / "a" / "mod.rs").read_text(encoding="utf-8")
    assert "pub mod b;" in a_mod and "pub mod c;" in a_mod
    assert "pub mod r#type;" in (tmp_path / "mod.rs").read_text(encoding="utf-8")
    assert "pub mod r#match;" in (tmp_path / "type" / "mod.rs").read_text(encoding="utf-8")
    # leaf binding files are untouched
    assert "// bindings b" in (tmp_path / "a" / "b" / "mod.rs").read_text(encoding="utf-8")

    # idempotent: re-running adds no duplicates
    finalize_rust_module_tree(tmp_path, prefixes)
    assert (tmp_path / "a" / "mod.rs").read_text(encoding="utf-8").count("pub mod b;") == 1


def test_rust_finalize_module_tree_merges_scopes_and_sorts(tmp_path: Path) -> None:
    root = tmp_path / "mod.rs"
    root.write_text("// pub mod z;\npub(crate) mod a;\n", encoding="utf-8")

    finalize_rust_module_tree(tmp_path, {"z", "a.child", "c"})
    text = root.read_text(encoding="utf-8")
    assert "pub(crate) mod a;" in text
    assert "pub mod a;" not in text
    lines = text.splitlines()
    assert lines.index("pub mod c;") < lines.index("pub mod z;")
    assert text.count("// tvm-ffi-stubgen-modules(begin)") == 1

    finalize_rust_module_tree(tmp_path, {"a.child"})
    text = root.read_text(encoding="utf-8")
    assert "pub(crate) mod a;" in text
    assert "pub mod c;" in text
    assert "pub mod z;" in text.splitlines()
    assert "// pub mod z;" in text  # comments are not mistaken for declarations


def test_rust_finalize_rejects_type_module_name_collision(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "mod.rs").write_text(
        f"{C.RUST_SYNTAX.begin} object/demo.child\n{C.RUST_SYNTAX.end}\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedTypeError, match="conflicts with child module"):
        finalize_rust_module_tree(tmp_path, {"demo", "demo.child"})


def test_rust_defined_type_collision_is_rejected() -> None:
    with pytest.raises(UnsupportedTypeError, match="both generate item 'FooObj'"):
        RustGenerator().reserve_defined_types(RustImports(), {"demo::Foo", "demo::FooObj"})


def test_rust_stage2_ignores_global_only_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stub_cli, "collect_type_keys", lambda: {})
    global_funcs = {
        "demo": [FuncInfo.from_schema("demo.f", TypeSchema("Callable", (TypeSchema("int"),)))]
    }
    generated = _stage_2(
        [],
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        InitConfig("demo", "demo", "demo."),
        tmp_path,
        global_funcs,
        RustGenerator(),
    )
    assert generated == set()
    assert list(tmp_path.iterdir()) == []


def test_rust_init_generation_is_repeatable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("int"))],
        methods=[],
        type_key="demo.Parent",
        parent_type_key="ffi.Object",
    )
    child = ObjectInfo(
        fields=[],
        methods=[],
        type_key="demo.child.Thing",
        parent_type_key="ffi.Object",
    )
    infos = {info.type_key: info for info in (parent, child)}
    registry = {"demo.child": ["demo.child.Thing"]}
    monkeypatch.setattr(stub_cli, "collect_type_keys", lambda: registry)
    monkeypatch.setattr(
        stub_cli,
        "toposort_objects",
        lambda keys: [infos[key] for key in keys],
    )
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", infos.__getitem__)
    generator = RustGenerator()
    init = InitConfig("demo", "demo", "demo.")

    # Generate a child first, leaving demo/mod.rs as an intermediate module.
    files: list[FileInfo] = []
    prefixes = _stage_2(files, RC.RUST_TY_MAP_DEFAULTS.copy(), init, tmp_path, {}, generator)
    for file in files:
        _stage_3(file, Options(), RC.RUST_TY_MAP_DEFAULTS.copy(), {}, generator)
    generator.finalize_init(tmp_path, prefixes)

    # Later promote that intermediate module into an object-bearing module.
    registry = {"demo": ["demo.Parent"], "demo.child": ["demo.child.Thing"]}
    files = collect_files([tmp_path])
    prefixes = _stage_2(files, RC.RUST_TY_MAP_DEFAULTS.copy(), init, tmp_path, {}, generator)
    for file in files:
        _stage_3(file, Options(), RC.RUST_TY_MAP_DEFAULTS.copy(), {}, generator)
    generator.finalize_init(tmp_path, prefixes)
    promoted = (tmp_path / "demo" / "mod.rs").read_text(encoding="utf-8")
    assert "pub mod child;" in promoted
    assert f"{C.RUST_SYNTAX.begin} object/demo.Parent" in promoted
    assert "#![" not in promoted and "//!" not in promoted

    first = {
        path.relative_to(tmp_path): path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.rs")
    }

    files = collect_files([tmp_path])
    prefixes = _stage_2(files, RC.RUST_TY_MAP_DEFAULTS.copy(), init, tmp_path, {}, generator)
    for file in files:
        _stage_3(file, Options(), RC.RUST_TY_MAP_DEFAULTS.copy(), {}, generator)
    generator.finalize_init(tmp_path, prefixes)
    second = {
        path.relative_to(tmp_path): path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.rs")
    }
    assert second == first


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


def test_rust_object_without_constructor_still_has_field_access() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("int"))],
        methods=[],
        type_key="demo.Plain",
        parent_type_key="ffi.Object",
        has_init=False,
    )
    text, _ = _gen_rust_object(info)
    assert "struct PlainObj {" in text
    assert "pub fn value(&self) -> ::tvm_ffi::Result<i64>" in text
    assert "impl Plain {" not in text
    assert "downcast" not in text
    assert "fn ffi_new" not in text


def test_rust_root_object_has_no_upcast() -> None:
    text, _ = _gen_rust_object(_expr_info())
    assert "impl ::core::convert::From<Expr>" not in text
