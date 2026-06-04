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
"""Python code generation for the ``tvm-ffi-stubgen`` tool.

This module owns every act of turning language-agnostic FFI metadata
(:class:`tvm_ffi.stub.utils.FuncInfo` / :class:`~tvm_ffi.stub.utils.ObjectInfo`)
into concrete Python source text. The metadata dataclasses themselves carry no
rendering logic; the ``render_*`` helpers below do, so a different backend can
render the same data into a different language.
"""

from __future__ import annotations

from io import StringIO
from typing import Callable

from .. import consts as C
from ..file_utils import CodeBlock
from ..utils import FuncInfo, InitConfig, ObjectInfo, Options
from .imports import ImportItem

#: Renders a :class:`~tvm_ffi.core.TypeSchema` into a Python type expression.
#: The second argument is the per-block leaf-name mapper (records imports as a
#: side effect). ``None`` means "use the built-in Python rendering"
#: (:meth:`TypeSchema.repr`).
RenderType = Callable[..., str]


def _bind_render(
    render_type: RenderType | None, ty_map: Callable[[str], str]
) -> Callable[..., str]:
    """Bind a `RenderType` (or the Python default) to a leaf-name mapper."""
    if render_type is None:
        return lambda schema: schema.repr(ty_map)
    return lambda schema: render_type(schema, ty_map)


# --- metadata -> Python text rendering --------------------------------------
# These were previously methods on FuncInfo/ObjectInfo. They live here so the
# data classes stay language-agnostic (plan "a").


def render_func_signature(
    func: FuncInfo,
    ty_map: Callable[[str], str],
    indent: int,
    render_type: RenderType | None = None,
) -> str:
    """Render a function signature string for ``func``."""
    render = _bind_render(render_type, ty_map)
    try:
        _, func_name = func.schema.name.rsplit(".", 1)
    except ValueError:
        func_name = func.schema.name
    buf = StringIO()
    buf.write(" " * indent)
    buf.write(f"def {func_name}(")
    if func.schema.origin != "Callable":
        raise ValueError(f"Expected Callable type schema, but got: {func.schema}")
    if not func.schema.args:
        ty_map("Any")
        buf.write("*args: Any) -> Any: ...")
        return buf.getvalue()
    arg_ret = func.schema.args[0]
    arg_args = func.schema.args[1:]
    for i, arg in enumerate(arg_args):
        if func.is_member and i == 0:
            buf.write("self, ")
        else:
            buf.write(f"_{i}: ")
            buf.write(render(arg))
            buf.write(", ")
    if arg_args:
        buf.write("/")
    buf.write(") -> ")
    buf.write(render(arg_ret))
    buf.write(": ...")
    return buf.getvalue()


def render_object_fields(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    render_type: RenderType | None = None,
) -> list[str]:
    """Render field definitions for ``info``."""
    indent_str = " " * indent
    render = _bind_render(render_type, ty_map)
    return [f"{indent_str}{field.name}: {render(field)}" for field in info.fields]


def render_object_methods(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    render_type: RenderType | None = None,
) -> list[str]:
    """Render method definitions for ``info``."""
    indent_str = " " * indent
    ret = []
    for method in info.methods:
        func_name = method.schema.name.rsplit(".", 1)[-1]
        if func_name == "__ffi_init__":
            # __ffi_init__ is installed as an instance method (self, *args, **kwargs) -> None
            # by _install_ffi_init_attr, regardless of the C++ static registration.
            ret.append(_render_ffi_init_from_method(method, ty_map, indent, render_type))
            continue
        if not method.is_member:
            ret.append(f"{indent_str}@staticmethod")
        ret.append(render_func_signature(method, ty_map, indent, render_type))
    return ret


def _render_ffi_init_from_method(
    method: FuncInfo,
    ty_map: Callable[[str], str],
    indent: int,
    render_type: RenderType | None = None,
) -> str:
    """Render ``__ffi_init__`` TypeMethod as an instance method returning None."""
    indent_str = " " * indent
    render = _bind_render(render_type, ty_map)
    schema = method.schema
    # Subclass __ffi_init__ signatures legitimately differ from the parent
    # (different fields → different constructor params), so suppress LSP.
    ignore = "  # ty: ignore[invalid-method-override]"
    if schema.origin != "Callable" or not schema.args:
        ty_map("Any")
        return f"{indent_str}def __ffi_init__(self, *args: Any) -> None: ...{ignore}"
    # schema.args[0] is return type, schema.args[1:] are param types.
    parts: list[str] = []
    for i, arg in enumerate(schema.args[1:]):
        parts.append(f"_{i}: {render(arg)}")
    if parts:
        params = ", ".join(parts)
        return f"{indent_str}def __ffi_init__(self, {params}, /) -> None: ...{ignore}"
    return f"{indent_str}def __ffi_init__(self) -> None: ...{ignore}"


def render_object_ffi_init(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    render_type: RenderType | None = None,
) -> list[str]:
    """Render a ``__ffi_init__`` stub when it's not already in TypeMethod.

    For types whose ``__ffi_init__`` is auto-generated by ``RegisterFFIInit``
    (TypeAttrColumn only), synthesize a static-method stub from field metadata.
    Types that already have ``__ffi_init__`` in TypeMethod (from explicit
    ``refl::init<>``) get it via ``render_object_methods`` instead.
    """
    if not info.has_init:
        return []
    # If __ffi_init__ is already in methods (from TypeMethod), methods render it.
    if any(m.schema.name.rsplit(".", 1)[-1] == "__ffi_init__" for m in info.methods):
        return []
    return _render_ffi_init_from_fields(info, ty_map, indent, render_type)


def render_object_init(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    render_type: RenderType | None = None,
) -> list[str]:
    """Render an ``__init__`` stub from init-eligible field metadata."""
    if not info.has_init:
        return []
    return _render_init_from_fields(info, ty_map, indent, render_type)


def _format_field_params(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    render_type: RenderType | None = None,
) -> str:
    """Format init-eligible fields as a parameter string with defaults and kw_only."""
    render = _bind_render(render_type, ty_map)
    positional = [f for f in info.init_fields if not f.kw_only]
    kw_only = [f for f in info.init_fields if f.kw_only]

    pos_required = [f for f in positional if not f.has_default]
    pos_default = [f for f in positional if f.has_default]
    kw_required = [f for f in kw_only if not f.has_default]
    kw_default = [f for f in kw_only if f.has_default]

    parts: list[str] = []
    for f in pos_required:
        parts.append(f"{f.name}: {render(f.schema)}")
    for f in pos_default:
        parts.append(f"{f.name}: {render(f.schema)} = ...")
    if kw_required or kw_default:
        parts.append("*")
        for f in kw_required:
            parts.append(f"{f.name}: {render(f.schema)}")
        for f in kw_default:
            parts.append(f"{f.name}: {render(f.schema)} = ...")

    return ", ".join(parts)


def _render_init_from_fields(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    render_type: RenderType | None = None,
) -> list[str]:
    """Render ``__init__`` from init-eligible field metadata (auto-generated init)."""
    indent_str = " " * indent
    params = _format_field_params(info, ty_map, render_type)
    if params:
        return [f"{indent_str}def __init__(self, {params}) -> None: ..."]
    return [f"{indent_str}def __init__(self) -> None: ..."]


def _render_ffi_init_from_fields(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    render_type: RenderType | None = None,
) -> list[str]:
    """Render ``__ffi_init__`` stub from field metadata for auto-generated init."""
    indent_str = " " * indent
    # Subclass __ffi_init__ signatures legitimately differ from the parent
    # (different fields → different constructor params), so suppress LSP.
    ignore = "  # ty: ignore[invalid-method-override]"
    params = _format_field_params(info, ty_map, render_type)
    if params:
        return [f"{indent_str}def __ffi_init__(self, {params}) -> None: ...{ignore}"]
    return [f"{indent_str}def __ffi_init__(self) -> None: ...{ignore}"]


# --- Python scaffolding templates (init mode) -------------------------------
# These emit Python source plus stub-directive markers. The marker comment token
# comes from the supplied `MarkerSyntax`, so the directive structure stays
# language-aware even though the surrounding code is Python-specific.


def _prompt_globals(mod: str, syntax: C.MarkerSyntax) -> str:
    return f"""{syntax.begin} global/{mod}
{syntax.end}
"""


def _prompt_class_def(
    type_name: str, type_key: str, parent_type_name: str, syntax: C.MarkerSyntax
) -> str:
    return f'''@_FFI_REG_OBJ("{type_key}")
class {type_name}({parent_type_name}):
    """FFI binding for `{type_key}`."""

    {syntax.begin} object/{type_key}
    {syntax.end}\n\n'''


def _prompt_import_object(type_key: str, type_name: str, syntax: C.MarkerSyntax) -> str:
    return f"""{syntax.import_object} {type_key};False;{type_name}\n"""


def _prompt_import_section(syntax: C.MarkerSyntax) -> str:
    return f"""
{syntax.begin} import-section
{syntax.end}
"""


def _prompt_all_section(syntax: C.MarkerSyntax) -> str:
    return f"""
__all__ = [
    {syntax.begin} __all__
    {syntax.end}
]
"""


def _type_suffix_and_record(
    ty_map: dict[str, str],
    imports: list[ImportItem],
    func_names: set[str] | None = None,
) -> Callable[[str], str]:
    def _run(name: str) -> str:
        nonlocal ty_map, imports
        name = ty_map.get(name, name)
        suffix = name.rsplit(".", 1)[-1]
        if "." in name:
            alias = None
            if func_names and suffix in func_names:
                alias = f"_{suffix}"
            imports.append(ImportItem(name, type_checking_only=True, alias=alias))
            if alias:
                return alias
        return suffix

    return _run


def generate_python_global_funcs(
    code: CodeBlock,
    global_funcs: list[FuncInfo],
    ty_map: dict[str, str],
    imports: list[ImportItem],
    opt: Options,
    render_type: RenderType | None = None,
) -> None:
    """Generate function signatures for global functions.

    It processes: global/${prefix}@${import_from="tvm_ffi")
    """
    assert len(code.lines) >= 2
    if not global_funcs:
        return
    assert isinstance(code.param, tuple)
    prefix, import_from = code.param
    if not import_from:
        import_from = "tvm_ffi"
    imports.extend(
        [
            ImportItem(
                f"{import_from}.init_ffi_api",
                type_checking_only=False,
                alias="_FFI_INIT_FUNC",
            ),
            ImportItem(
                "typing.TYPE_CHECKING",
                type_checking_only=False,
            ),
        ]
    )
    func_names = {f.schema.name.rsplit(".", 1)[-1] for f in global_funcs}
    fn_ty_map = _type_suffix_and_record(ty_map, imports, func_names=func_names)
    results: list[str] = [
        "# fmt: off",
        f'_FFI_INIT_FUNC("{prefix}", __name__)',
        "if TYPE_CHECKING:",
        *[render_func_signature(func, fn_ty_map, opt.indent, render_type) for func in global_funcs],
        "# fmt: on",
    ]
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in results],
        code.lines[-1],
    ]


def generate_python_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: list[ImportItem],
    opt: Options,
    obj_info: ObjectInfo,
    render_type: RenderType | None = None,
) -> None:
    """Generate a class definition for an object type.

    It processes: object/${type_key}
    """
    assert len(code.lines) >= 2
    info = obj_info
    method_names = {m.schema.name.rsplit(".", 1)[-1] for m in info.methods}
    fn_ty_map = _type_suffix_and_record(ty_map, imports, func_names=method_names)
    init_lines = render_object_init(info, fn_ty_map, opt.indent, render_type)
    ffi_init_lines = render_object_ffi_init(info, fn_ty_map, opt.indent, render_type)
    type_checking_lines = [
        *init_lines,
        *ffi_init_lines,
        *render_object_methods(info, fn_ty_map, opt.indent, render_type),
    ]
    if type_checking_lines:
        imports.append(
            ImportItem(
                "typing.TYPE_CHECKING",
                type_checking_only=False,
            )
        )
        results = [
            "# fmt: off",
            *render_object_fields(info, fn_ty_map, 0, render_type),
            "if TYPE_CHECKING:",
            *type_checking_lines,
            "# fmt: on",
        ]
    else:
        results = [
            "# fmt: off",
            *render_object_fields(info, fn_ty_map, 0, render_type),
            "# fmt: on",
        ]
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in results],
        code.lines[-1],
    ]


def generate_python_import_section(
    code: CodeBlock,
    imports: list[ImportItem],
    opt: Options,
) -> None:
    """Generate import statements for the types used in the stub.

    It processes: import-section
    """
    imports_concrete: dict[str, list[ImportItem]] = {}
    imports_ty_check: dict[str, list[ImportItem]] = {}
    for item in imports:
        if item.type_checking_only:
            imports_ty_check.setdefault(item.mod, []).append(item)
        else:
            imports_concrete.setdefault(item.mod, []).append(item)
    if imports_ty_check:
        imports_concrete.setdefault("typing", []).append(
            ImportItem("typing.TYPE_CHECKING", type_checking_only=True)
        )

    def _make_line(mod: str, items: list[ImportItem], indent: int) -> str:
        items.sort(key=lambda item: item.name)
        names = ", ".join(sorted(set(item.name_with_alias for item in items)))
        indent_str = " " * indent
        if mod:
            return f"{indent_str}from {mod} import {names}"
        else:
            return f"{indent_str}import {names}"

    results: list[str] = []
    if imports_concrete:
        results.extend(
            _make_line(mod, imports_concrete[mod], indent=0) for mod in sorted(imports_concrete)
        )
    if imports_ty_check:
        results.append("if TYPE_CHECKING:")
        results.extend(
            _make_line(mod, imports_ty_check[mod], opt.indent) for mod in sorted(imports_ty_check)
        )
    if results:
        code.lines = [
            code.lines[0],
            "# fmt: off",
            "# isort: off",
            "from __future__ import annotations",
            *results,
            "# isort: on",
            "# fmt: on",
            code.lines[-1],
        ]


def generate_python_all(code: CodeBlock, names: set[str], opt: Options) -> None:
    """Generate an `__all__` variable for the given names."""
    assert len(code.lines) >= 2
    if not names:
        return

    indent = " " * code.indent
    names = {f.rsplit(".", 1)[-1] for f in names}

    def _sort_key(name: str) -> tuple[int, str]:
        if name.isupper():
            return (0, name)
        if name and name[0].isupper() and not "_" in name:
            return (1, name)
        return (2, name)

    code.lines = [
        code.lines[0],
        *[f'{indent}"{name}",' for name in sorted(names, key=_sort_key)],
        code.lines[-1],
    ]


def generate_python_export(code: CodeBlock) -> None:
    """Generate an `__all__` variable for the given names."""
    assert len(code.lines) >= 2

    mod = code.param
    code.lines = [
        code.lines[0],
        "# fmt: off",
        "# isort: off",
        f"from .{mod} import *  # noqa: F403",
        f"from .{mod} import __all__ as {mod}__all__",
        'if "__all__" not in globals():',
        "    __all__ = []",
        f"__all__.extend({mod}__all__)",
        "# isort: on",
        "# fmt: on",
        code.lines[-1],
    ]


def generate_python_ffi_api(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: InitConfig,
    is_root: bool,
    syntax: C.MarkerSyntax = C.PYTHON_SYNTAX,
) -> str:
    """Generate the initial FFI API stub code for a given module."""
    # TODO(@junrus): New code is appended to the end of the file.
    # We should consider a more sophisticated approach.
    append = ""

    # Part 0. Imports
    if not code_blocks:
        append += f"""\"\"\"FFI API bindings for {module_name}.\"\"\"\n"""
    if not any(code.kind == "import-section" for code in code_blocks):
        append += _prompt_import_section(syntax)

    # Part 1. Library loading
    if is_root:
        append += _prompt_import_object("tvm_ffi.libinfo.load_lib_module", "_FFI_LOAD_LIB", syntax)
        append += f"""LIB = _FFI_LOAD_LIB("{init_cfg.pkg}", "{init_cfg.shared_target}")\n"""

    # Part 2. Global functions
    if not any(code.kind == "global" for code in code_blocks):
        append += _prompt_globals(module_name, syntax)

    # Part 3. Object types
    if object_infos:
        append += _prompt_import_object("tvm_ffi.register_object", "_FFI_REG_OBJ", syntax)

    defined_type_keys = {info.type_key for info in object_infos if info.type_key}
    for info in object_infos:
        type_key = info.type_key
        parent_type_key = info.parent_type_key
        if type_key is None:
            continue
        # Canonicalize type key names
        type_key = ty_map.get(type_key, type_key)
        type_name = type_key.rsplit(".", 1)[-1]
        parent_type_key = (
            ty_map.get(parent_type_key, parent_type_key) if parent_type_key else parent_type_key
        )
        parent_type_name = parent_type_key.rsplit(".", 1)[-1] if parent_type_key else "Object"
        # Import parent type keys if they are not defined in the current module
        if parent_type_key and parent_type_key not in defined_type_keys:
            parent_type_name = "_" + parent_type_key.replace(".", "_")
            append += _prompt_import_object(parent_type_key, parent_type_name, syntax)
        # Generate class definition
        append += _prompt_class_def(
            type_name,
            type_key,
            parent_type_name,
            syntax,
        )
    # Part 4. __all__
    if not any(code.kind == "__all__" for code in code_blocks):
        append += _prompt_all_section(syntax)
    return append


def generate_python_init(
    code_blocks: list[CodeBlock],
    module_name: str,
    submodule: str = "_ffi_api",
    syntax: C.MarkerSyntax = C.PYTHON_SYNTAX,
) -> str:
    """Generate the `__init__.py` file for the `tvm_ffi` package."""
    code = f"""
{syntax.begin} export/{submodule}
{syntax.end}
"""
    if not code_blocks:
        return f"""\"\"\"Package {module_name}.\"\"\"\n""" + code
    if not any(code.kind == "export" for code in code_blocks):
        return code
    return ""
