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

import ctypes
import itertools
import typing
from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
import tvm_ffi.stub.rust_generator.codegen as rust_codegen
from tvm_ffi import Object, method
from tvm_ffi.core import TypeSchema, _lookup_or_register_type_info_from_type_key
from tvm_ffi.dataclasses import py_class
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
    generate_rust_global_funcs,
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
    assert RustUse("tvm_ffi::Map") in imports.items
    assert RustUse("tvm_ffi::String") in imports.items


def test_render_optional_value_positions() -> None:
    # Value positions render plain `Option<T>`; field position routes
    # differently (see the `test_rust_optional_field_*` tests).
    assert _rust_render(TypeSchema("Optional", (TypeSchema("int"),)))[0] == "Option<i64>"
    assert _rust_render(TypeSchema("Optional", (TypeSchema("str"),)))[0] == "Option<String>"
    assert _rust_render(TypeSchema("Optional", (TypeSchema("bytes"),)))[0] == "Option<Bytes>"
    text, imports = _rust_render(
        TypeSchema("Optional", (TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))),))
    )
    assert text == "Option<Map<String, i64>>"
    assert RustUse("tvm_ffi::Map") in imports.items
    # Nested inside an Array (elements are Any-encoded, so `Option<T>` is fine).
    text, _ = _rust_render(TypeSchema("Array", (TypeSchema("Optional", (TypeSchema("int"),)),)))
    assert text == "Array<Option<i64>>"


@pytest.mark.parametrize(
    ("schema", "origin"),
    [
        # A genuinely unsupported origin buried inside a container still bubbles
        # up. (`Any` is NOT here anymore -- it renders as `AnyValue`; see
        # `test_render_any_element_maps_to_any_value`.)
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
    assert imports.items == [RustUse("tvm_ffi::Array")]  # recorded exactly once


def test_ty_render_same_leaf_different_path_raises() -> None:
    # No auto-aliasing: two different paths wanting the same in-scope name only
    # arise from pathological type names and are declared unsupported.
    imports = RustImports()
    assert imports.record("crate_a::Foo") == "Foo"  # first claims the bare leaf
    with pytest.raises(UnsupportedTypeError):
        imports.record("crate_b::Foo")
    assert imports.items == [RustUse("crate_a::Foo")]  # the loser is not recorded


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


def _trusted_ffi_object_info() -> ObjectInfo:
    """Minimal registry proof for the 24-byte, 8-aligned FFI object header."""
    return ObjectInfo(
        fields=[],
        methods=[],
        type_key="ffi.Object",
        parent_type_key=None,
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=24,
        has_native_layout_metadata=True,
        native_alignment=8,
        has_native_alignment_metadata=True,
    )


def _install_trusted_ffi_object(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _trusted_ffi_object_info()
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", lambda key: root)


def _expr_info(*, mutable: bool = True) -> ObjectInfo:
    """Root `Expr`: field `value: i64`, static `test() -> i64`, init(i64)."""
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


def _point_info() -> ObjectInfo:
    """Root auto-init `Point`: canonical reflected constructor takes x and y."""
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


def test_rust_auto_init_uses_reflected_keyword_protocol() -> None:
    text, _ = _gen_rust_object(_point_info())
    assert "pub fn ffi_new(x: i64, y: i64) -> Result<Point> {" in text
    assert 'PointObj::type_index(), "__ffi_init__")?;' in text
    assert 'Function::get_global_cached(&__TVM_FFI_KWARGS, "ffi.GetKwargsObject")?;' in text
    assert 'tvm_ffi::String::from("x")' in text
    assert 'tvm_ffi::String::from("y")' in text
    assert "AnyView::from(&__tvm_ffi_kwargs)" in text
    assert "AnyView::from(&x)" in text
    assert "AnyView::from(&y)" in text
    assert "ObjectArc::new" not in text
    assert "build_obj" not in text
    assert "PointBuilder" not in text


def test_rust_auto_init_preserves_defaults_kw_only_and_field_names() -> None:
    names = (
        "parent_required",
        "parent_default",
        "child_required",
        "child_default",
    )
    info = ObjectInfo(
        fields=[],
        methods=[],
        type_key="testing.AutoInit",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo(names[0], NamedTypeSchema(names[0], TypeSchema("int")), False, False),
            InitFieldInfo(names[1], NamedTypeSchema(names[1], TypeSchema("int")), False, True),
            InitFieldInfo(names[2], NamedTypeSchema(names[2], TypeSchema("int")), True, False),
            InitFieldInfo(names[3], NamedTypeSchema(names[3], TypeSchema("int")), True, True),
        ],
        has_init=True,
    )

    text, _ = _gen_rust_object(info)

    assert (
        "pub fn ffi_new(parent_required: i64, parent_default: i64, "
        "child_required: i64, child_default: i64) -> Result<AutoInit> {" in text
    )
    key_offsets = [text.index(f'tvm_ffi::String::from("{name}")') for name in names]
    assert key_offsets == sorted(key_offsets)
    assert "ffi.GetKwargsObject" in text


def test_rust_duplicate_auto_init_name_fails_transactionally() -> None:
    info = _point_info()
    info.init_fields.append(
        InitFieldInfo("x", NamedTypeSchema("x", TypeSchema("int")), True, False)
    )
    block = _rust_object_block(info.type_key or "x")
    original_lines = list(block.lines)
    imports = RustImports(items=[RustUse("tvm_ffi::Tensor")])
    original_imports = list(imports.items)

    with pytest.raises(UnsupportedTypeError, match="occurs more than once"):
        generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), info)

    assert block.lines == original_lines
    assert imports.items == original_imports


def test_rust_direct_layout_uses_exact_scalar_signedness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_trusted_ffi_object(monkeypatch)
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("signed", TypeSchema("int"), 4, 24, 4, signed=True),
            NamedTypeSchema("unsigned", TypeSchema("int"), 1, 28, 1, signed=False),
            NamedTypeSchema("weight", TypeSchema("float"), 4, 32, 4),
        ],
        methods=[],
        type_key="cpp_rust_test.Scalars",
        parent_type_key="ffi.Object",
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=40,
        parent_native_total_size=24,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )

    text, _ = _gen_rust_object(info)

    assert "#[repr(C, align(8))]" in text
    assert "pub signed: i32," in text
    assert "pub unsigned: u8," in text
    assert "pub weight: f32," in text
    assert "pub fn signed(&self) -> Result<i32>" in text
    assert "pub fn unsigned(&self) -> Result<u8>" in text
    assert "ObjectArc::new" not in text


def _scrambled_layout_info(*, gap: bool = False) -> ObjectInfo:
    """Build a proven layout whose registration order differs from memory order."""
    return ObjectInfo(
        fields=[
            NamedTypeSchema(
                "beta", TypeSchema("int"), size=8, offset=32, alignment=8, signed=False
            ),
            NamedTypeSchema(
                "gamma",
                TypeSchema("int"),
                size=4,
                offset=48 if gap else 40,
                alignment=4,
                signed=True,
            ),
            NamedTypeSchema(
                "alpha", TypeSchema("int"), size=4, offset=24, alignment=4, signed=True
            ),
        ],
        methods=[],
        type_key="cpp_rust_test.Scrambled",
        parent_type_key="ffi.Object",
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=56 if gap else 48,
        parent_native_total_size=24,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )


def test_rust_direct_layout_is_offset_ordered_and_self_checking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_trusted_ffi_object(monkeypatch)
    text, _ = _gen_rust_object(_scrambled_layout_info())
    alpha, beta, gamma = (text.index(f"pub {n}:") for n in ("alpha", "beta", "gamma"))
    assert alpha < beta < gamma
    assert "MaybeUninit<[u8; 4]>" in text
    assert "assert!(std::mem::size_of::<ScrambledObj>() == 48);" in text
    assert "assert!(std::mem::align_of::<ScrambledObj>() == 8);" in text
    assert "assert!(std::mem::offset_of!(ScrambledObj, beta) == 32);" in text
    assert "assert!(std::mem::size_of::<u64>() == 8);" in text
    assert "assert!(std::mem::align_of::<u64>() == 8);" in text


def test_rust_direct_layout_represents_hidden_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_trusted_ffi_object(monkeypatch)
    text, _ = _gen_rust_object(_scrambled_layout_info(gap=True))
    assert "pub struct ScrambledObj {" in text
    assert "__tvm_ffi_padding_1: MaybeUninit<[u8; 8]>" in text
    assert text.index("pub beta:") < text.index("__tvm_ffi_padding_1") < text.index("pub gamma:")


def test_rust_offset_padding_resumes_after_unverifiable_field() -> None:
    # One unproven field makes the whole physical mirror opaque; the object and
    # all semantic reads remain available through owning reflected getters.
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
    text, _ = _gen_rust_object(info)
    assert "pub a:" not in text
    assert "pub fn a(&self) -> Result<Any>" in text
    assert "get_reflected_field(self, 0)" in text
    assert "pub fn d(&self) -> Result<Any>" in text
    assert "get_reflected_field(self, 3)" in text


def test_rust_offset_overlap_falls_back_to_opaque(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_trusted_ffi_object(monkeypatch)
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("a", TypeSchema("int"), size=8, offset=24, alignment=8, signed=True),
            NamedTypeSchema("b", TypeSchema("int"), size=4, offset=28, alignment=4, signed=True),
        ],
        methods=[],
        type_key="cpp_rust_test.Overlap",
        parent_type_key="ffi.Object",
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=40,
        parent_native_total_size=24,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )
    text, _ = _gen_rust_object(info)
    assert "#[repr(C, align(" not in text
    assert "pub a:" not in text
    assert "pub fn a(&self) -> Result<i64>" in text
    assert "pub fn b(&self) -> Result<i32>" in text


def test_rust_padded_type_keeps_layout_and_uses_ffi_ctor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_trusted_ffi_object(monkeypatch)
    # Padding remains part of the read-only C++ layout mirror, but constructor
    # selection is uniform: every constructible type calls reflected FFI init.
    info = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "op_type", TypeSchema("int"), size=4, offset=24, alignment=4, signed=True
            ),
            NamedTypeSchema(
                "num_inputs", TypeSchema("int"), size=4, offset=32, alignment=4, signed=True
            ),
        ],
        methods=[],
        type_key="ir.Op",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo("op_type", NamedTypeSchema("op_type", TypeSchema("int")), False, False),
            InitFieldInfo(
                "num_inputs", NamedTypeSchema("num_inputs", TypeSchema("int")), False, False
            ),
        ],
        has_init=True,
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=40,
        parent_native_total_size=24,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )
    text, _ = _gen_rust_object(info)
    assert "__tvm_ffi_padding_0: MaybeUninit<[u8; 4]>" in text
    # FFI ctor, not the builder:
    assert "pub fn ffi_new(op_type: i64, num_inputs: i64) -> Result<Op> {" in text
    assert 'OpObj::type_index(), "__ffi_init__")?;' in text
    assert 'tvm_ffi::String::from("op_type")' in text
    assert 'tvm_ffi::String::from("num_inputs")' in text
    assert "ffi.GetKwargsObject" in text
    assert "OpBuilder" not in text


@pytest.mark.parametrize("init_arity", [2, 1])
def test_rust_explicit_init_signature_is_not_ignored(init_arity: int) -> None:
    info = _point_info()
    args = (TypeSchema("cpp_rust_test.Point"),) + (TypeSchema("int"),) * init_arity
    info.methods = [
        FuncInfo(
            NamedTypeSchema("__ffi_init__", TypeSchema("Callable", args)),
            is_member=False,
        )
    ]
    text, _ = _gen_rust_object(info)
    params = ", ".join(f"_{i}: i64" for i in range(init_arity))
    assert f"pub fn ffi_new({params}) -> Result<Point> {{" in text
    assert 'PointObj::type_index(), "__ffi_init__")?;' in text
    assert "ffi.GetKwargsObject" not in text
    for i in range(init_arity):
        assert f"AnyView::from(&_{i})" in text
    assert "ObjectArc::new" not in text
    assert "PointBuilder" not in text


def test_rust_optional_method_arg_and_return() -> None:
    # Value positions render plain `Option<T>`; no in-place mirror involved.
    info = _point_info()
    info.methods = [
        FuncInfo(
            NamedTypeSchema(
                "lookup",
                TypeSchema(
                    "Callable",
                    (
                        TypeSchema("Optional", (TypeSchema("str"),)),
                        TypeSchema("Optional", (TypeSchema("int"),)),
                    ),
                ),
            ),
            is_member=False,
        )
    ]
    text, _ = _gen_rust_object(info)
    assert "pub fn lookup(_0: Option<i64>) -> Result<Option<String>> {" in text
    assert "Optional<" not in text  # the field mirror never appears in value positions


def test_rust_nullable_object_ref_in_value_positions() -> None:
    # C++ nullable ObjectRefs are encoded as Optional<Object> in TypeSchema, so
    # method/global value positions use Rust Option<T> without a field carrier.
    point = TypeSchema("cpp_rust_test.Point", origin_type_index=128)
    nullable_point = TypeSchema("Optional", (point,))
    info = _point_info()
    info.methods = [
        FuncInfo(
            NamedTypeSchema(
                "lookup",
                TypeSchema("Callable", (nullable_point, nullable_point)),
            ),
            is_member=False,
        )
    ]
    text, _ = _gen_rust_object(info)
    assert "pub fn lookup(_0: Option<Point>) -> Result<Option<Point>> {" in text


def _optional_field_info(fields: list[NamedTypeSchema], *, has_init: bool = True) -> ObjectInfo:
    return ObjectInfo(
        fields=fields,
        methods=[],
        type_key="cpp_rust_test.OptHolder",
        parent_type_key="ffi.Object",
        has_init=has_init,
    )


@pytest.mark.parametrize(
    ("payload", "getter_type", "extra_use"),
    [
        pytest.param(TypeSchema("int"), "Option<i64>", None, id="int"),
        pytest.param(TypeSchema("float"), "Option<f64>", None, id="float"),
        pytest.param(TypeSchema("bool"), "Option<bool>", None, id="bool"),
        pytest.param(TypeSchema("str"), "Option<String>", "tvm_ffi::String", id="str"),
        pytest.param(TypeSchema("bytes"), "Option<Bytes>", "tvm_ffi::Bytes", id="bytes"),
        pytest.param(TypeSchema("Device"), "Option<DLDevice>", "tvm_ffi::DLDevice", id="device"),
        pytest.param(TypeSchema("dtype"), "Option<DLDataType>", "tvm_ffi::DLDataType", id="dtype"),
        # `cpp_rust_test.Point` shares the holder's module: a local name, no `use`.
        pytest.param(
            TypeSchema("cpp_rust_test.Point"),
            "Option<Point>",
            None,
            id="objref",
        ),
        # A cross-module payload anchors at the generated root (F1).
        pytest.param(
            TypeSchema("other.Point"),
            "Option<Point>",
            "super::other::Point",
            id="objref-cross-module",
        ),
        pytest.param(
            TypeSchema("Object"),
            "Option<ObjectRef>",
            "tvm_ffi::object::ObjectRef",
            id="objref-generic",
        ),
        # `Any` payload gets the element treatment: the owning `AnyValue`
        # carrier preserves the full dynamic value domain and is AnyCompatible.
        pytest.param(
            TypeSchema("Any"),
            "Option<AnyValue>",
            "tvm_ffi::AnyValue",
            id="any-value",
        ),
        pytest.param(
            TypeSchema("Array", (TypeSchema("int"),)),
            "Option<Array<i64>>",
            "tvm_ffi::Array",
            id="array",
        ),
        pytest.param(
            TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))),
            "Option<Map<String, i64>>",
            "tvm_ffi::Map",
            id="map",
        ),
        pytest.param(TypeSchema("Callable"), "Option<Function>", "tvm_ffi::Function", id="fn"),
        pytest.param(TypeSchema("Tensor"), "Option<Tensor>", "tvm_ffi::Tensor", id="tensor"),
        pytest.param(TypeSchema("Shape"), "Option<Shape>", "tvm_ffi::Shape", id="shape"),
    ],
)
def test_rust_optional_field_has_uniform_semantic_getter(
    payload: TypeSchema, getter_type: str, extra_use: str | None
) -> None:
    schema = NamedTypeSchema("x", TypeSchema("Optional", (payload,)), size=16)
    text, imports = _gen_rust_object(_optional_field_info([schema], has_init=False))
    assert f"pub fn x(&self) -> Result<{getter_type}>" in text
    assert "pub x:" not in text
    assert RustUse("tvm_ffi::Optional") not in imports.items
    if extra_use is not None:
        assert RustUse(extra_use) in imports.items


def test_rust_optional_field_without_size_still_has_semantic_getter() -> None:
    schema = NamedTypeSchema("x", TypeSchema("Optional", (TypeSchema("int"),)))
    text, _ = _gen_rust_object(_optional_field_info([schema]))
    assert "pub fn x(&self) -> Result<Option<i64>>" in text
    assert "OptHolderBuilder" not in text


def test_rust_nullable_object_ref_field_getter_uses_option() -> None:
    point = TypeSchema("cpp_rust_test.Point", origin_type_index=128)
    schema = NamedTypeSchema(
        "x",
        TypeSchema("Optional", (point,)),
        size=ctypes.sizeof(ctypes.c_void_p),
        default=None,
    )
    text, imports = _gen_rust_object(_optional_field_info([schema]))
    assert "pub fn x(&self) -> Result<Option<Point>>" in text
    assert "ObjectArc::new" not in text
    assert "OptHolderBuilder" not in text
    assert RustUse("tvm_ffi::Optional") not in imports.items


def test_rust_reflected_nullable_object_ref_field_carriers() -> None:
    # The C++ fields have distinct physical carriers but one user-facing
    # Optional schema, so an opaque API deliberately returns the same type.
    type_info = _lookup_or_register_type_info_from_type_key("testing.TestNullableObjectRefHolder")
    text, imports = _gen_rust_object(ObjectInfo.from_type_info(type_info))
    assert "pub fn value(&self) -> Result<Option<TestIntPair>>" in text
    assert "pub fn optional_value(&self) -> Result<Option<TestIntPair>>" in text
    assert RustUse("tvm_ffi::Optional") not in imports.items


@pytest.mark.parametrize(
    "schema",
    [
        # `void*` (`ctypes.c_void_p`) has no Rust rendering. Getter generation
        # must catch that at the field boundary instead of emitting a dangling
        # `use ctypes::c_void_p` or dropping the enclosing object.
        pytest.param(TypeSchema("ctypes.c_void_p"), id="field"),
        pytest.param(TypeSchema("Array", (TypeSchema("ctypes.c_void_p"),)), id="array-element"),
        pytest.param(
            TypeSchema("Optional", (TypeSchema("ctypes.c_void_p"),)), id="optional-payload"
        ),
    ],
)
def test_rust_void_ptr_field_falls_back_locally_to_any(schema: TypeSchema) -> None:
    field = NamedTypeSchema("x", schema, size=16)
    text, imports = _gen_rust_object(_optional_field_info([field], has_init=False))
    assert "pub struct OptHolderObj" in text
    assert "pub fn x(&self) -> Result<Any>" in text
    assert "get_reflected_field(self, 0)" in text
    assert RustUse("tvm_ffi::Any") in imports.items


def _f1_info(own_key: str, ref_key: str) -> ObjectInfo:
    """Build an object `own_key` with one field of type `ref_key` (for path tests)."""
    return ObjectInfo(
        fields=[NamedTypeSchema("x", TypeSchema(ref_key))],
        methods=[],
        type_key=own_key,
        parent_type_key="ffi.Object",
        has_init=False,
    )


@pytest.mark.parametrize(
    ("own_key", "ref_key", "expected_use"),
    [
        # A bare `use ir::Expr;` in edition 2021 resolves to an extern crate
        # `ir` (E0432): cross-module references must anchor at the shared
        # generated root -- one `super::` per segment of this file's module.
        pytest.param("tirx.Ramp", "ir.Expr", "super::ir::Expr", id="sibling-module"),
        # A sibling prefix whose name also exists as a *submodule* of this
        # module (`tirx::transform`) must not be captured by a bare path: the
        # `super::` anchor resolves to the top-level `transform` module.
        pytest.param(
            "tirx.Ramp",
            "transform.PassInfo",
            "super::transform::PassInfo",
            id="sibling-name-capture",
        ),
        # Nested module: one `super::` per segment (`tirx/transform/mod.rs` -> 2).
        pytest.param(
            "tirx.transform.UnrollConfig",
            "ir.Expr",
            "super::super::ir::Expr",
            id="nested-two-supers",
        ),
        # Up-tree reference from a nested module: uniform root-anchored path.
        pytest.param(
            "tirx.transform.UnrollConfig",
            "tirx.Ramp",
            "super::super::tirx::Ramp",
            id="up-tree-ref",
        ),
        # A dotless own key lands in the generated root itself: `self::` (a
        # bare `use ir::…` would not see the sibling submodule in 2021).
        pytest.param("Rootless", "ir.Expr", "self::ir::Expr", id="root-file-self"),
    ],
)
def test_rust_cross_module_ref_uses_rooted_path(
    own_key: str, ref_key: str, expected_use: str
) -> None:
    text, imports = _gen_rust_object(_f1_info(own_key, ref_key))
    ref_leaf = ref_key.rsplit(".", 1)[-1]
    assert f"pub fn x(&self) -> Result<{ref_leaf}>" in text
    assert RustUse(expected_use) in imports.items


def test_rust_same_module_ref_is_local() -> None:
    # `tirx.Stmt` lands in the same file as `tirx.Ramp` (one file per prefix):
    # a local item -- bare leaf, no `use` recorded at all.
    text, imports = _gen_rust_object(_f1_info("tirx.Ramp", "tirx.Stmt"))
    assert "pub fn x(&self) -> Result<Stmt>" in text
    assert all(u.leaf != "Stmt" for u in imports.items)


def test_rust_unmapped_ffi_key_keeps_crate_path() -> None:
    # An `ffi.*` key outside the ty_map lives in the crate (RUST_MOD_MAP head
    # rewrite), not the generated tree: never `super::`-anchored.
    text, imports = _gen_rust_object(_f1_info("tirx.Ramp", "ffi.Opaque"))
    assert "pub fn x(&self) -> Result<Opaque>" in text
    assert RustUse("tvm_ffi::Opaque") in imports.items


def test_rust_reflected_base_field_renames_parent_slot() -> None:
    # F5: `tirx.Ramp` has a reflected field literally named `base` -- the
    # synthesized parent-embed slot must dodge it (E0124), renaming itself
    # and its Deref body to `__base`;
    # C++ reserves `__`-prefixed identifiers, so the dodge cannot re-collide.
    # The REFLECTED `base` keeps its natural name everywhere.
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("base", TypeSchema("tirx.Expr")),
            NamedTypeSchema("lanes", TypeSchema("int")),
        ],
        methods=[],
        type_key="tirx.Ramp",
        parent_type_key="tirx.Expr",
        has_init=True,
    )
    text, _ = _gen_rust_object(info)
    # The offset-zero parent slot is still dodged; the reflected field is read
    # by name through the uniform getter API.
    assert "    __base: ExprObj," in text
    assert "pub fn base(&self) -> Result<Expr>" in text
    # Deref to the parent goes through the dodged slot
    assert "        &self.__base" in text
    assert "RampBuilder" not in text


def test_rust_root_reflected_base_field_renames_slot() -> None:
    # Root objects embed the bare crate `Object`; the slot dodge applies too.
    info = ObjectInfo(
        fields=[NamedTypeSchema("base", TypeSchema("int"))],
        methods=[],
        type_key="tirx.Load",
        parent_type_key="ffi.Object",
        has_init=True,
    )
    text, _ = _gen_rust_object(info)
    assert "    __base: Object," in text
    assert "pub fn base(&self) -> Result<Any>" in text


def test_rust_no_base_field_keeps_plain_slot() -> None:
    # Without a reflected `base`, the slot keeps its plain name (API stability).
    info = ObjectInfo(
        fields=[NamedTypeSchema("x", TypeSchema("int"))],
        methods=[],
        type_key="tirx.Plain",
        parent_type_key="ffi.Object",
        has_init=True,
    )
    text, _ = _gen_rust_object(info)
    assert "    base: Object," in text
    assert "__base" not in text


def test_rust_cross_module_parent_imports_ref_and_obj() -> None:
    # F3: `target.VirtualDevice`'s parent is `ir.Attrs` (cross-prefix). The
    # struct embeds `base: AttrsObj` and the upcast targets `Attrs` -- BOTH
    # names must come into scope through the generated tree, alongside the
    # F1 path rule.
    info = ObjectInfo(
        fields=[NamedTypeSchema("x", TypeSchema("int"))],
        methods=[],
        type_key="target.VirtualDevice",
        parent_type_key="ir.Attrs",
        has_init=False,
    )
    text, imports = _gen_rust_object(info)
    assert "    base: AttrsObj," in text
    assert "    type Target = AttrsObj;" in text  # Deref to the parent struct
    assert "impl From<VirtualDevice> for Attrs {" in text  # upcast to the ref
    assert RustUse("super::ir::Attrs") in imports.items
    assert RustUse("super::ir::AttrsObj") in imports.items


def test_rust_same_module_parent_stays_local() -> None:
    # A same-module parent is a local item: bare `ExprObj`/`Expr`, no `use`.
    info = ObjectInfo(
        fields=[],
        methods=[],
        type_key="tirx.Ramp",
        parent_type_key="tirx.Expr",
        has_init=False,
    )
    text, imports = _gen_rust_object(info)
    assert "    base: ExprObj," in text
    assert "impl From<Ramp> for Expr {" in text
    assert all(u.leaf not in ("Expr", "ExprObj") for u in imports.items)


def test_rust_keyword_field_raw_ident_all_positions() -> None:
    # Keywords are escaped consistently in getters and constructor parameters.
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("desc", TypeSchema("int")),
            NamedTypeSchema("impl", TypeSchema("int")),
            NamedTypeSchema("type", TypeSchema("int"), default=7),
        ],
        methods=[],
        type_key="tirx.TensorIntrin",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo("desc", NamedTypeSchema("desc", TypeSchema("int")), False, False),
            InitFieldInfo("impl", NamedTypeSchema("impl", TypeSchema("int")), False, False),
            InitFieldInfo("type", NamedTypeSchema("type", TypeSchema("int")), False, True),
        ],
        has_init=True,
    )
    text, _ = _gen_rust_object(info)
    assert "pub fn r#impl(&self) -> Result<Any>" in text
    assert "pub fn r#type(&self) -> Result<Any>" in text
    assert "pub fn ffi_new(desc: i64, r#impl: i64, r#type: i64)" in text
    assert "AnyView::from(&r#impl)" in text
    assert "TensorIntrinBuilder" not in text
    assert "pub fn desc(&self) -> Result<Any>" in text


def test_rust_keyword_method_raw_ident_keeps_ffi_name() -> None:
    # A reflected method named `match`: the Rust `fn` ident is escaped, the
    # FFI lookup string keeps the reflected spelling.
    info = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo(
                NamedTypeSchema("match", TypeSchema("Callable", (TypeSchema("int"),))),
                is_member=False,
            )
        ],
        type_key="tirx.TensorIntrin",
        parent_type_key="ffi.Object",
        has_init=False,
    )
    text, _ = _gen_rust_object(info)
    assert "pub fn r#match() -> Result<i64> {" in text
    assert '"match")?;' in text  # from_type_method_cached(.., "match")


def test_rust_non_raw_ident_field_skips() -> None:
    # `self`/`crate`/`super`/`Self` cannot be identifiers even raw: loud skip.
    info = ObjectInfo(
        fields=[NamedTypeSchema("self", TypeSchema("int"))],
        methods=[],
        type_key="tirx.Bad",
        parent_type_key="ffi.Object",
        has_init=False,
    )
    with pytest.raises(UnsupportedTypeError) as exc:
        _gen_rust_object(info)
    assert exc.value.origin == "self"


def test_rust_cross_module_ref_in_container_and_method() -> None:
    # Every position funnels through `_ty_render`: a container element and a
    # method return of a cross-module key record the same rooted import.
    info = ObjectInfo(
        fields=[NamedTypeSchema("kids", TypeSchema("Array", (TypeSchema("ir.Expr"),)))],
        methods=[
            FuncInfo(
                NamedTypeSchema("make", TypeSchema("Callable", (TypeSchema("ir.Expr"),))),
                is_member=False,
            )
        ],
        type_key="tirx.Ramp",
        parent_type_key="ffi.Object",
        has_init=False,
    )
    text, imports = _gen_rust_object(info)
    assert "pub fn kids(&self) -> Result<Array<Expr>>" in text
    assert "pub fn make() -> Result<Expr> {" in text
    assert RustUse("super::ir::Expr") in imports.items


@pytest.mark.parametrize(
    ("payload", "size"),
    [
        # Any size other than the 16-byte `TVMFFIAny` cell is unsupported: C++
        # `ffi::Optional<T>` is uniformly 16 bytes for storage-enabled `T`, so
        # an 8-byte reflected Optional is not a mirrorable layout.
        pytest.param(TypeSchema("int"), 8, id="scalar-8-not-cell"),
        # String has an object type index, but its Rust value is a 16-byte
        # inline cell rather than a pointer-backed ObjectRef wrapper.
        pytest.param(TypeSchema("str"), 8, id="string-object-pointer-not-mirrorable"),
        # `std::string` folds to "str" but is the ~40-byte std::optional fallback.
        pytest.param(TypeSchema("str"), 40, id="std-string-alias"),
    ],
)
def test_rust_optional_field_layout_size_guard(payload: TypeSchema, size: int) -> None:
    schema = NamedTypeSchema("x", TypeSchema("Optional", (payload,)), size=size)
    text, _ = _gen_rust_object(_optional_field_info([schema], has_init=False))
    # Carrier size cannot prove a direct layout, but it does not alter the
    # language-level Optional value returned through reflection.
    expected = "i64" if payload.origin == "int" else "String"
    assert f"pub fn x(&self) -> Result<Option<{expected}>>" in text
    assert "pub x:" not in text


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        # `Any` in element/payload position renders as the owning dynamic
        # `AnyValue` carrier. Unlike ObjectRef, it preserves scalars, strings,
        # objects, containers, and None.
        pytest.param(TypeSchema("Array", (TypeSchema("Any"),)), "Array<AnyValue>", id="array-any"),
        pytest.param(
            TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any"))),
            "Map<String, AnyValue>",
            id="map-any-value",
        ),
        pytest.param(
            TypeSchema("Optional", (TypeSchema("Any"),)), "Option<AnyValue>", id="optional-any"
        ),
        # A bare `Map` fills to (Any, Any) -> both sides stay dynamically typed.
        pytest.param(TypeSchema("Map"), "Map<AnyValue, AnyValue>", id="bare-map-fills-any"),
        # Nested: the `Any` normalization applies at every element depth.
        pytest.param(
            TypeSchema("Map", (TypeSchema("str"), TypeSchema("Array", (TypeSchema("Any"),)))),
            "Map<String, Array<AnyValue>>",
            id="map-of-array-any",
        ),
        pytest.param(
            TypeSchema("Optional", (TypeSchema("Array", (TypeSchema("Any"),)),)),
            "Option<Array<AnyValue>>",
            id="optional-array-any",
        ),
    ],
)
def test_render_any_element_maps_to_any_value(schema: TypeSchema, expected: str) -> None:
    text, imports = _rust_render(schema)
    assert text == expected
    assert RustUse("tvm_ffi::AnyValue") in imports.items
    assert RustUse("tvm_ffi::object::ObjectRef") not in imports.items


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        # A generic/opaque object renders as the single-pointer `ObjectRef`
        # handle in every container/value position (it IS `AnyCompatible`).
        pytest.param(TypeSchema("Object"), "ObjectRef", id="bare-object"),
        pytest.param(TypeSchema("ffi.Object"), "ObjectRef", id="bare-ffi-object"),
        pytest.param(
            TypeSchema("Array", (TypeSchema("Object"),)), "Array<ObjectRef>", id="array-object"
        ),
        pytest.param(
            TypeSchema("Map", (TypeSchema("str"), TypeSchema("Object"))),
            "Map<String, ObjectRef>",
            id="map-object-value",
        ),
        pytest.param(
            TypeSchema("Optional", (TypeSchema("Object"),)),
            "Option<ObjectRef>",
            id="optional-object-value",
        ),
    ],
)
def test_render_object_element_maps_to_objectref(schema: TypeSchema, expected: str) -> None:
    text, imports = _rust_render(schema)
    assert text == expected
    assert RustUse("tvm_ffi::object::ObjectRef") in imports.items


def test_rust_map_field_and_methods() -> None:
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("cfg", TypeSchema("Map", (TypeSchema("str"), TypeSchema("int")))),
        ],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    "merge",
                    TypeSchema(
                        "Callable",
                        (
                            TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))),
                            TypeSchema("Map", (TypeSchema("str"), TypeSchema("int"))),
                        ),
                    ),
                ),
                is_member=False,
            )
        ],
        type_key="cpp_rust_test.MapHolder",
        parent_type_key="ffi.Object",
        has_init=True,
    )
    text, imports = _gen_rust_object(info)
    assert "pub fn cfg(&self) -> Result<Map<String, i64>>" in text
    assert "pub fn merge(_0: Map<String, i64>) -> Result<Map<String, i64>> {" in text
    assert "MapHolderBuilder" not in text
    assert RustUse("tvm_ffi::Map") in imports.items


def _point3d_info() -> ObjectInfo:
    """Build the derived `Point3D : Point` fixture with a reflected constructor."""
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


def test_rust_derived_constructor_uses_full_reflected_init_chain() -> None:
    text, _ = _gen_rust_object(_point3d_info())
    assert "pub fn ffi_new(x: i64, y: i64, z: i64) -> Result<Point3D> {" in text
    assert 'Point3DObj::type_index(), "__ffi_init__")?;' in text
    assert "Point3DBuilder" not in text
    assert "build_obj" not in text
    assert "ObjectArc::new" not in text


def test_rust_object_root_struct_and_impl() -> None:
    text, imports = _gen_rust_object(_expr_info())
    # data struct embeds the root Object as `base`
    assert "#[repr(C)]" in text
    assert "struct ExprObj {" in text
    assert "    base: Object," in text
    assert "pub fn value(&self) -> Result<Any>" in text
    assert "get_reflected_field(self, 0)" in text
    # ObjectCore impl is folded into the `#[derive(Object)]` proc macro: the stub
    # only emits the derive + `#[type_key]` attr, not a hand-written impl.
    assert "#[derive(tvm_ffi::derive::Object)]" in text
    assert '#[type_key = "cpp_rust_test.Expr"]' in text
    assert "unsafe impl ObjectCore" not in text
    assert "lookup_type_index" not in text
    assert "object_header_mut" not in text
    # Shared reference handles expose immutable Deref only, even for a mutable
    # C++ object type.
    assert "#[repr(transparent)]\n#[derive(tvm_ffi::derive::ObjectRef, Clone)]" in text
    assert "struct Expr {" in text
    assert "    data: ObjectArc<ExprObj>," in text
    assert "impl Deref for Expr {" in text
    assert "DerefMut" not in text
    # Constructor calls the canonical reflected C++ path.
    assert "pub struct ExprObj {" in text
    assert "pub struct Expr {" in text
    assert "pub fn ffi_new(value: i64) -> Result<Expr> {" in text
    assert "pub fn test() -> Result<i64> {" in text
    assert 'ExprObj::type_index(), "__ffi_init__")?;' in text
    assert "ObjectArc::new" not in text
    assert "ExprBuilder" not in text
    # static method: no self; uniform packed-call convention with cached getter
    assert "static F: std::sync::OnceLock<tvm_ffi::Function>" in text
    assert (
        "let f = tvm_ffi::Function::from_type_method_cached(&F, "
        'ExprObj::type_index(), "test")?;' in text
    )
    assert "Ok(f.call_packed(&[])?.try_into()?)" in text
    uses = {u.as_use_line() for u in imports.items}
    assert "use tvm_ffi::Object;" in uses
    assert "use std::ops::DerefMut;" not in uses


def test_rust_object_derived_embeds_parent() -> None:
    text, _ = _gen_rust_object(_add_info())
    assert "struct AddObj {" in text
    assert "    base: ExprObj," in text  # parent Obj embedded, not Object
    assert "pub fn a(&self) -> Result<Expr>" in text
    # object_header_mut is derived by the `#[derive(Object)]` macro from the
    # first field (`base: ExprObj`), so the stub no longer hand-writes it.
    assert "object_header_mut" not in text
    # derived Obj also derefs to its embedded base
    assert "impl Deref for AddObj {" in text
    assert "    type Target = ExprObj;" in text
    # instance method: &mut self receiver (mutable class); self is packed as `&*self`
    assert "fn update(&self) -> Result<()> {" in text
    assert "Ok(f.call_packed(&[tvm_ffi::object::as_any_view(self)])?.try_into()?)" in text
    # Construction does not inspect or allocate the parent mirror; it invokes
    # this type's own reflected constructor.
    assert "pub fn ffi_new(a: Expr, b: Expr, value: i64) -> Result<Add> {" in text
    assert 'AddObj::type_index(), "__ffi_init__")?;' in text
    assert "AddBuilder" not in text


@pytest.mark.parametrize("mutable", [False, True])
def test_rust_object_never_has_derefmut(mutable: bool) -> None:
    text, imports = _gen_rust_object(_expr_info(mutable=mutable))
    assert "impl Deref for Expr {" in text
    assert "DerefMut" not in text
    assert RustUse("std::ops::DerefMut") not in imports.items


def test_rust_object_field_of_type_object_maps_to_objectref() -> None:
    # The struct `base` is the embedded 24-byte `Object` data struct (spelled
    # literally by codegen), while a field whose C++ type is a generic
    # `ffi.Object` is a single-pointer `ObjectRef` handle. The two are distinct
    # types with distinct leaves, so both `use`s coexist without collision.
    info = ObjectInfo(
        fields=[NamedTypeSchema("child", TypeSchema("ffi.Object"))],
        methods=[],
        type_key="demo.Holder",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)
    assert "    base: Object," in text  # boilerplate Object as the struct base
    assert "pub fn child(&self) -> Result<ObjectRef>" in text
    uses = [u.as_use_line() for u in imports.items]
    assert uses.count("use tvm_ffi::Object;") == 1
    assert uses.count("use tvm_ffi::object::ObjectRef;") == 1


def test_rust_object_any_container_fields_keep_dynamic_values() -> None:
    """Generated IR fields use the runtime carrier proven by the Rust tests."""
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("args", TypeSchema("Array", (TypeSchema("Any"),))),
            NamedTypeSchema("config", TypeSchema("Map", (TypeSchema("str"), TypeSchema("Any")))),
        ],
        methods=[],
        type_key="demo.DynamicFields",
        parent_type_key="ffi.Object",
        has_init=False,
    )
    text, imports = _gen_rust_object(info)

    assert "pub fn args(&self) -> Result<Array<AnyValue>>" in text
    assert "pub fn config(&self) -> Result<Map<String, AnyValue>>" in text
    assert RustUse("tvm_ffi::AnyValue") in imports.items
    assert RustUse("tvm_ffi::object::ObjectRef") not in imports.items


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
    assert "pub fn probe(&self, _0: AnyView<'_>) -> Result<Any> {" in text
    assert "Result<AnyView>" not in text  # the bug would have produced this
    # All methods use the uniform `call_packed` convention (which natively speaks
    # `AnyView` args and an `Any` return -- the only convention that can). An
    # `Any` return is forwarded directly, with no trailing `try_into`.
    assert "into_typed_fn!" not in text
    assert "f.call_packed(&[tvm_ffi::object::as_any_view(self), _0])" in text
    # owning Any return must record its `use`
    assert RustUse("tvm_ffi::Any") in imports.items
    assert RustUse("tvm_ffi::AnyView") in imports.items


def _has_map_info() -> ObjectInfo:
    # A field schema Rust cannot name. It must degrade locally, not erase the
    # enclosing object or any type that refers to it.
    return ObjectInfo(
        fields=[
            NamedTypeSchema(
                "cfg",
                TypeSchema("Map", (TypeSchema("str"), TypeSchema("List", (TypeSchema("int"),)))),
            ),
        ],
        methods=[],
        type_key="demo.HasMap",
        parent_type_key="ffi.Object",
    )


def test_rust_object_unsupported_field_falls_back_to_any() -> None:
    block = _rust_object_block("demo.HasMap")
    imports = RustImports(items=[RustUse("tvm_ffi::Tensor")])
    generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), _has_map_info())
    text = "\n".join(block.lines)
    assert "pub struct HasMapObj" in text
    assert "pub fn cfg(&self) -> Result<Any>" in text
    assert RustUse("tvm_ffi::Tensor") in imports.items
    assert RustUse("tvm_ffi::Any") in imports.items
    assert all(item.path != "tvm_ffi::Map" for item in imports.items)


def test_rust_stage3_keeps_objects_with_unsupported_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rs = tmp_path / "demo.rs"
    rs.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} import-section",
                C.RUST_SYNTAX.end,
                "",
                f"{C.RUST_SYNTAX.begin} object/demo.HasMap",
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
        "demo.HasMap": _has_map_info(),
        "demo.Holder": ObjectInfo(
            fields=[NamedTypeSchema("child", TypeSchema("demo.HasMap"))],
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
    assert "[Skipped]" not in capsys.readouterr().out
    assert "struct HasMapObj" in text
    assert "pub fn cfg(&self) -> Result<Any>" in text
    assert "struct HolderObj" in text
    assert "pub fn child(&self) -> Result<HasMap>" in text
    assert "use demo::HasMap;" not in text


def test_rust_bytes_field_maps_to_crate_bytes() -> None:
    # C++ `Bytes` fields carry the schema origin "bytes" (string.h TypeStr).
    info = ObjectInfo(
        fields=[NamedTypeSchema("payload", TypeSchema("bytes"))],
        methods=[],
        type_key="demo.Blob",
        parent_type_key="ffi.Object",
    )
    text, imports = _gen_rust_object(info)
    assert "pub fn payload(&self) -> Result<Bytes>" in text
    assert RustUse("tvm_ffi::Bytes") in imports.items


def test_rust_unknown_bare_origin_falls_back_to_any() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("name", TypeSchema("const char*"))],
        methods=[],
        type_key="demo.Raw",
        parent_type_key="ffi.Object",
    )
    text, _ = _gen_rust_object(info)
    assert "pub struct RawObj" in text
    assert "pub fn name(&self) -> Result<Any>" in text


@pytest.mark.parametrize(
    "missing",
    [
        "native_size",
        "native_alignment",
        "mutability",
        "parent_size",
        "parent_alignment",
        "field_offset",
        "field_size",
        "field_alignment",
        "field_signedness",
    ],
)
def test_rust_layout_proof_is_all_or_nothing(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_trusted_ffi_object(monkeypatch)
    info = _scrambled_layout_info()
    if missing == "native_size":
        info.has_native_layout_metadata = False
    elif missing == "native_alignment":
        info.has_native_alignment_metadata = False
    elif missing == "mutability":
        info.has_mutability_metadata = False
    elif missing == "parent_size":
        info.parent_has_native_layout_metadata = False
    elif missing == "parent_alignment":
        info.parent_has_native_alignment_metadata = False
    elif missing == "field_offset":
        info.fields[0].offset = None
    elif missing == "field_size":
        info.fields[0].size = None
    elif missing == "field_alignment":
        info.fields[0].alignment = None
    else:
        info.fields[0].signed = None

    text, _ = _gen_rust_object(info)

    assert "pub struct ScrambledObj" in text
    assert "#[repr(C, align(" not in text
    assert "pub beta:" not in text
    assert "get_reflected_field(self, 0)" in text
    assert "pub struct Scrambled" in text


@pytest.mark.parametrize(
    ("has_metadata", "mutable", "has_marker"),
    [(False, False, True), (True, True, True), (True, False, False)],
)
def test_rust_opaque_mutability_controls_send_sync_marker(
    has_metadata: bool, mutable: bool, has_marker: bool
) -> None:
    info = ObjectInfo(
        fields=[],
        methods=[],
        type_key="demo.State",
        parent_type_key="ffi.Object",
        mutable=mutable,
        has_mutability_metadata=has_metadata,
    )

    text, _ = _gen_rust_object(info)

    assert ("PhantomData<std::rc::Rc<()>>" in text) is has_marker


@pytest.mark.parametrize("parent_is_trusted", [True, False])
def test_rust_direct_layout_requires_complete_ancestor_proof(
    parent_is_trusted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _trusted_ffi_object_info()
    parent = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "parent_value",
                TypeSchema("int"),
                size=8,
                offset=24,
                alignment=8,
                signed=True,
            )
        ],
        methods=[],
        type_key="demo.Parent",
        parent_type_key="ffi.Object",
        mutable=False,
        has_mutability_metadata=parent_is_trusted,
        native_total_size=32,
        parent_native_total_size=24,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )
    child = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "child_value",
                TypeSchema("int"),
                size=4,
                offset=32,
                alignment=4,
                signed=False,
            )
        ],
        methods=[],
        type_key="demo.Child",
        parent_type_key="demo.Parent",
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=40,
        parent_native_total_size=32,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )
    registry = {"ffi.Object": root, "demo.Parent": parent}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", registry.__getitem__)

    text, _ = _gen_rust_object(child)

    assert ("#[repr(C, align(8))]" in text) is parent_is_trusted
    assert ("pub child_value: u32" in text) is parent_is_trusted
    assert "pub fn child_value(&self) -> Result<u32>" in text


def test_rust_direct_layout_rejects_opaque_ancestor_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _trusted_ffi_object_info()
    parent = ObjectInfo(
        fields=[
            NamedTypeSchema("wide", TypeSchema("int"), size=8, offset=24, alignment=8, signed=True),
            NamedTypeSchema(
                "overlap", TypeSchema("int"), size=4, offset=28, alignment=4, signed=True
            ),
        ],
        methods=[],
        type_key="demo.Parent",
        parent_type_key="ffi.Object",
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=40,
        parent_native_total_size=24,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )
    child = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "child_value", TypeSchema("int"), size=4, offset=40, alignment=4, signed=True
            )
        ],
        methods=[],
        type_key="demo.Child",
        parent_type_key="demo.Parent",
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=48,
        parent_native_total_size=40,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )
    registry = {"ffi.Object": root, "demo.Parent": parent}
    monkeypatch.setattr(rust_codegen, "object_info_from_type_key", registry.__getitem__)

    text, _ = _gen_rust_object(child)

    assert "#[repr(C, align(" not in text
    assert "pub child_value:" not in text
    assert "pub fn child_value(&self) -> Result<i32>" in text


def test_rust_direct_nullable_object_ref_is_pointer_proven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_trusted_ffi_object(monkeypatch)
    point = TypeSchema("demo.Point", origin_type_index=128)
    info = ObjectInfo(
        fields=[
            NamedTypeSchema(
                "maybe_point",
                TypeSchema("Optional", (point,)),
                size=ctypes.sizeof(ctypes.c_void_p),
                offset=24,
                alignment=ctypes.alignment(ctypes.c_void_p),
            )
        ],
        methods=[],
        type_key="demo.Holder",
        parent_type_key="ffi.Object",
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=32,
        parent_native_total_size=24,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )

    text, _ = _gen_rust_object(info)

    assert "pub maybe_point: Option<Point>" in text
    assert "pub fn maybe_point(&self) -> Result<Option<Point>>" in text
    assert "assert!(std::mem::size_of::<Option<Point>>() == 8);" in text


def test_rust_internal_layout_names_avoid_reflected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_trusted_ffi_object(monkeypatch)
    names_and_offsets = [
        ("base", 24),
        ("__base", 32),
        ("__tvm_ffi_base", 40),
        ("__tvm_ffi_padding_0", 56),
    ]
    info = ObjectInfo(
        fields=[
            NamedTypeSchema(
                name,
                TypeSchema("int"),
                size=8,
                offset=offset,
                alignment=8,
                signed=True,
            )
            for name, offset in names_and_offsets
        ],
        methods=[],
        type_key="demo.CollidingFields",
        parent_type_key="ffi.Object",
        mutable=False,
        has_mutability_metadata=True,
        native_total_size=64,
        parent_native_total_size=24,
        has_native_layout_metadata=True,
        parent_has_native_layout_metadata=True,
        native_alignment=8,
        parent_native_alignment=8,
        has_native_alignment_metadata=True,
        parent_has_native_alignment_metadata=True,
    )

    text, _ = _gen_rust_object(info)

    assert "    __tvm_ffi_base_2: Object," in text
    assert "    __tvm_ffi_padding_0_2: MaybeUninit<[u8; 8]>," in text
    assert "    pub __tvm_ffi_padding_0: i64," in text


def test_rust_type_key_is_emitted_as_a_safe_string_literal() -> None:
    info = ObjectInfo(
        fields=[],
        methods=[],
        type_key='demo"line\n.Type',
        parent_type_key="ffi.Object",
    )
    text, _ = _gen_rust_object(info)
    assert '#[type_key = "demo\\"line\\n.Type"]' in text


def test_rust_direct_and_opaque_layouts_share_getter_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_trusted_ffi_object(monkeypatch)
    direct = _scrambled_layout_info()
    opaque = _scrambled_layout_info()
    opaque.has_native_layout_metadata = False

    direct_text, _ = _gen_rust_object(direct)
    opaque_text, _ = _gen_rust_object(opaque)

    signatures = [
        "pub fn beta(&self) -> Result<u64>",
        "pub fn gamma(&self) -> Result<i32>",
        "pub fn alpha(&self) -> Result<i32>",
    ]
    for signature in signatures:
        assert signature in direct_text
        assert signature in opaque_text
    for index in range(3):
        call = f"get_reflected_field(self, {index})"
        assert call in direct_text
        assert call in opaque_text


def test_rust_unsupported_getter_schema_falls_back_only_that_field() -> None:
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("count", TypeSchema("int"), size=4, signed=False),
            NamedTypeSchema("config", TypeSchema("List", (TypeSchema("int"),))),
            NamedTypeSchema("label", TypeSchema("str")),
        ],
        methods=[],
        type_key="demo.Partial",
        parent_type_key="ffi.Object",
    )

    text, _ = _gen_rust_object(info)

    assert "pub fn count(&self) -> Result<u32>" in text
    assert "pub fn config(&self) -> Result<Any>" in text
    assert "pub fn label(&self) -> Result<String>" in text


def test_rust_method_names_normalize_but_lookup_original() -> None:
    info = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo(
                NamedTypeSchema(
                    "GetJSONGraph",
                    TypeSchema("Callable", (TypeSchema("None"), TypeSchema("demo.Node"))),
                ),
                is_member=True,
            )
        ],
        type_key="demo.Node",
        parent_type_key="ffi.Object",
        mutable=True,
    )

    text, _ = _gen_rust_object(info)

    assert "impl NodeObj" in text
    assert "pub fn get_json_graph(&self) -> Result<()>" in text
    assert 'NodeObj::type_index(), "GetJSONGraph")?;' in text


def test_rust_method_collision_is_transactional() -> None:
    info = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo(
                NamedTypeSchema("FooBar", TypeSchema("Callable", (TypeSchema("None"),))),
                is_member=False,
            ),
            FuncInfo(
                NamedTypeSchema("foo_bar", TypeSchema("Callable", (TypeSchema("None"),))),
                is_member=False,
            ),
        ],
        type_key="demo.Collision",
        parent_type_key="ffi.Object",
    )
    block = _rust_object_block("demo.Collision")
    block.lines.insert(1, "stale body")
    original_lines = list(block.lines)
    imports = RustImports(items=[RustUse("tvm_ffi::Tensor")])
    original_imports = list(imports.items)

    with pytest.raises(UnsupportedTypeError, match="normalizes"):
        generate_rust_object(block, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options(), info)

    assert block.lines == original_lines
    assert imports.items == original_imports


def test_rust_getter_method_collision_is_deterministic() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("str"))],
        methods=[
            FuncInfo(
                NamedTypeSchema("value", TypeSchema("Callable", (TypeSchema("str"),))),
                is_member=False,
            )
        ],
        type_key="demo.Named",
        parent_type_key="ffi.Object",
    )

    text, _ = _gen_rust_object(info)

    assert "pub fn get_value(&self) -> Result<String>" in text
    assert "pub fn value() -> Result<String>" in text


def test_rust_downcast_accepts_registered_subtypes() -> None:
    text, _ = _gen_rust_object(_expr_info())
    assert "tvm_ffi::object::is_instance_of::<N>((*header).type_index)" in text
    assert "(*header).type_index == <N as tvm_ffi::ObjectCore>::type_index()" not in text


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
    imports = RustImports(items=[RustUse("cpp_rust_test::Expr"), RustUse("tvm_ffi::Tensor")])
    # Expr is defined in this file -> its `use` must be dropped.
    generate_rust_import_section(block, imports, Options(), defined_types={"cpp_rust_test::Expr"})
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
    assert imp.items == [RustUse("cpp_rust_test::Expr")]
    assert gen.canonical_type_name("cpp_rust_test.Expr") == "cpp_rust_test::Expr"
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
    # Object block filled with the reflected-constructor path.
    assert "struct ExprObj {" in text
    assert "impl Expr {" in text
    assert 'ExprObj::type_index(), "__ffi_init__")?;' in text
    assert "ObjectArc::new" not in text
    # import-section filled with the machinery `use`s
    assert "use tvm_ffi::ObjectArc;" in text
    assert "use tvm_ffi::ObjectCore;" in text
    # Expr defines itself -> no self `use`
    assert "use cpp_rust_test::Expr;" not in text


def test_rust_global_stage3_end_to_end(tmp_path: Path) -> None:
    rs = tmp_path / "mod.rs"
    rs.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} global/testing",
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

    _stage_3(
        info,
        Options(dry_run=True),
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        {"testing": [_global_func("testing.AddOne", TypeSchema("int"), TypeSchema("int"))]},
        generator=RustGenerator(),
    )
    text = "\n".join(info.lines)

    assert "pub fn add_one(_0: i64) -> Result<i64>" in text
    assert 'Function::get_global_cached(&F, "testing.AddOne")?;' in text
    assert "use tvm_ffi::AnyView;" in text
    assert "use tvm_ffi::Result;" in text


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
    assert text.startswith("// Licensed to the Apache Software Foundation (ASF)")
    assert text.count("Licensed to the Apache Software Foundation") == 1
    assert "#![allow(dead_code, unused_imports)]" in text
    assert f"{C.RUST_SYNTAX.begin} import-section" in text
    assert f"{C.RUST_SYNTAX.begin} object/cpp_rust_test.Expr" in text
    # method lookup lives in the crate (`Function::from_type_method_cached`);
    # the scaffold carries no per-file helper block or support code.
    assert "helpers" not in text
    assert "fn get_type_method" not in text
    assert f"{C.RUST_SYNTAX.begin} global/demo" in text
    # no __all__ / export markers for Rust
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
    root_mod = (tmp_path / "mod.rs").read_text(encoding="utf-8")
    assert root_mod.startswith("// Licensed to the Apache Software Foundation (ASF)")
    assert "pub mod a;" in root_mod
    a_mod = (tmp_path / "a" / "mod.rs").read_text(encoding="utf-8")
    assert a_mod.startswith("// Licensed to the Apache Software Foundation (ASF)")
    assert "pub mod b;" in a_mod and "pub mod c;" in a_mod
    # leaf binding files are untouched
    leaf = (tmp_path / "a" / "b" / "mod.rs").read_text(encoding="utf-8")
    assert leaf == "// bindings b\n"

    # idempotent: re-running adds no duplicates
    finalize_rust_module_tree(tmp_path, {"a.b", "a.c"})
    assert (tmp_path / "a" / "mod.rs").read_text(encoding="utf-8").count("pub mod b;") == 1
    assert (tmp_path / "a" / "mod.rs").read_text(encoding="utf-8").count(
        "Licensed to the Apache Software Foundation"
    ) == 1


def _rust_global_block(prefix: str, *body: str) -> CodeBlock:
    return CodeBlock(
        kind="global",
        param=(prefix, ""),
        lineno_start=1,
        lineno_end=2 + len(body),
        lines=[f"// tvm-ffi-stubgen(begin): global/{prefix}", *body, C.RUST_SYNTAX.end],
    )


def _global_func(name: str, *callable_args: TypeSchema) -> FuncInfo:
    return FuncInfo(NamedTypeSchema(name, TypeSchema("Callable", callable_args)), is_member=False)


def test_rust_global_funcs_generate_typed_api_and_preserve_lookup_names() -> None:
    block = _rust_global_block("testing")
    imports = RustImports()
    funcs = [
        _global_func("testing.Match", TypeSchema("None")),
        _global_func("testing.EchoAny", TypeSchema("Any"), TypeSchema("Any")),
        _global_func("testing.GetJSONGraph", TypeSchema("int")),
        _global_func("testing.AddOne", TypeSchema("int"), TypeSchema("int")),
    ]

    generate_rust_global_funcs(block, funcs, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options())
    text = "\n".join(block.lines)

    # Registry order is normalized deterministically into Rust-facing names.
    assert (
        text.index("pub fn add_one")
        < text.index("pub fn echo_any")
        < text.index("pub fn get_json_graph")
        < text.index("pub fn r#match")
    )
    assert "pub fn add_one(_0: i64) -> Result<i64> {" in text
    assert 'Function::get_global_cached(&F, "testing.AddOne")?;' in text
    assert "Ok(f.call_packed(&[AnyView::from(&_0)])?.try_into()?)" in text

    # A top-level Any parameter is borrowed; the return remains owning.
    assert "pub fn echo_any(_0: AnyView<'_>) -> Result<Any> {" in text
    assert 'Function::get_global_cached(&F, "testing.EchoAny")?;' in text
    assert "    f.call_packed(&[_0])" in text

    # Keyword escaping changes only Rust source, never the complete FFI name.
    assert "pub fn r#match() -> Result<()> {" in text
    assert 'Function::get_global_cached(&F, "testing.Match")?;' in text
    assert {item.path for item in imports.items} >= {
        "tvm_ffi::Any",
        "tvm_ffi::AnyView",
        "tvm_ffi::Result",
    }


def test_rust_global_dynamic_callable_gets_honest_packed_fallback() -> None:
    block = _rust_global_block("testing")
    imports = RustImports()

    generate_rust_global_funcs(
        block,
        [_global_func("testing.Invoke")],
        RC.RUST_TY_MAP_DEFAULTS.copy(),
        imports,
        Options(),
    )
    text = "\n".join(block.lines)

    assert "pub fn invoke(args: &[AnyView<'_>]) -> Result<Any> {" in text
    assert 'Function::get_global_cached(&F, "testing.Invoke")?;' in text
    assert "    f.call_packed(args)" in text
    assert "call_packed(&[])" not in text


def test_rust_global_nested_any_and_generated_module_paths() -> None:
    block = _rust_global_block("tirx.transform")
    imports = RustImports()
    optional_any = TypeSchema("Optional", (TypeSchema("Any"),))
    funcs = [
        _global_func("tirx.transform.Maybe", optional_any, optional_any),
        _global_func(
            "tirx.transform.MakeExpr", TypeSchema("ir.Expr"), TypeSchema("tirx.transform.Block")
        ),
    ]

    generate_rust_global_funcs(block, funcs, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options())
    text = "\n".join(block.lines)

    assert "pub fn maybe(_0: Option<AnyValue>) -> Result<Option<AnyValue>>" in text
    assert "pub fn make_expr(_0: Block) -> Result<Expr>" in text
    assert RustUse("super::super::ir::Expr") in imports.items
    assert RustUse("tvm_ffi::AnyValue") in imports.items
    assert all(item.leaf != "Block" for item in imports.items)


def test_rust_global_normalized_collision_is_transactional() -> None:
    block = _rust_global_block("testing", "stale body")
    original_lines = list(block.lines)
    imports = RustImports(items=[RustUse("tvm_ffi::Tensor")])
    original_imports = list(imports.items)

    with pytest.raises(UnsupportedTypeError, match=r"FooBar.*foo_bar.*normalize"):
        generate_rust_global_funcs(
            block,
            [
                _global_func("testing.FooBar", TypeSchema("int")),
                _global_func("testing.foo_bar", TypeSchema("int")),
            ],
            RC.RUST_TY_MAP_DEFAULTS.copy(),
            imports,
            Options(),
        )

    assert block.lines == original_lines
    assert imports.items == original_imports


def test_rust_global_invalid_schema_is_transactional() -> None:
    block = _rust_global_block("testing", "stale body")
    original_lines = list(block.lines)
    imports = RustImports()
    funcs = [
        _global_func("testing.AValid", TypeSchema("int"), TypeSchema("str")),
        FuncInfo(NamedTypeSchema("testing.ZBad", TypeSchema("int")), is_member=False),
    ]

    with pytest.raises(UnsupportedTypeError, match="non-Callable schema"):
        generate_rust_global_funcs(block, funcs, RC.RUST_TY_MAP_DEFAULTS.copy(), imports, Options())

    assert block.lines == original_lines
    assert imports.items == []


def test_rust_global_empty_list_clears_stale_body() -> None:
    block = _rust_global_block("testing", "stale body")
    generate_rust_global_funcs(block, [], RC.RUST_TY_MAP_DEFAULTS.copy(), RustImports(), Options())
    assert block.lines == [block.lines[0], C.RUST_SYNTAX.end]


def test_rust_object_no_init_no_methods_has_only_ref_helpers() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("int"))],
        methods=[],
        type_key="demo.Plain",
        parent_type_key="ffi.Object",
        has_init=False,
    )
    text, _ = _gen_rust_object(info)
    assert "struct PlainObj {" in text
    # The impl block is always present for the `same_as`/`downcast` ref helpers,
    # but with no constructor or reflected methods.
    assert "impl Plain {" in text
    assert "pub fn same_as<" in text
    assert "pub fn downcast<" in text
    assert "fn ffi_new" not in text


def test_rust_object_ref_helpers_and_derived_upcast() -> None:
    # Every ref gets `same_as` + `downcast`; a derived type additionally gets the
    # offset-0 upcast `From<Derived> for <ParentRef>`.
    text, _ = _gen_rust_object(_add_info())
    assert "pub fn same_as<O: tvm_ffi::ObjectRefCore>(&self, other: &O) -> bool {" in text
    assert "pub fn downcast<N: tvm_ffi::ObjectCore>(&self) -> Option<&N> {" in text
    assert "impl From<Add> for Expr {" in text
    assert "ObjectArc::from_raw(ObjectArc::into_raw(arc) as *const ExprObj)" in text


def test_rust_root_object_has_ref_helpers_but_no_upcast() -> None:
    # A root object (parent `ffi.Object`) has no ref-typed parent, so no upcast.
    text, _ = _gen_rust_object(_expr_info())
    assert "pub fn same_as<" in text
    assert "impl From<Expr>" not in text
