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
"""Python backend for ``tvm-ffi-stubgen``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import consts as C
from . import codegen as G

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema

    from ..backend import TyRenderer
    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, ImportItem, InitConfig, ObjectInfo, Options


class PythonBackend:
    """Backend that emits Python type stubs by delegating to :mod:`.codegen`.

    This is a thin adapter over the existing functions so the new interface is
    backed by the proven Python implementation. Behaviour is unchanged.
    """

    name = "python"
    syntax = C.PYTHON_SYNTAX

    def default_ty_map(self) -> dict[str, str]:
        """Return the default FFI-origin -> Python-type name map."""
        return C.TY_MAP_DEFAULTS.copy()

    def render_type(self, schema: TypeSchema, ty_render: TyRenderer) -> str:
        """Render a type schema using Python typing syntax (delegates to `TypeSchema.repr`)."""
        return schema.repr(ty_render)

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: list[ImportItem],
        opt: Options,
    ) -> None:
        """Emit Python free-function signatures for a ``global/<prefix>`` block."""
        G.generate_python_global_funcs(code, global_funcs, ty_map, imports, opt, self.render_type)

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: list[ImportItem],
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit a Python class definition for an ``object/<key>`` block."""
        G.generate_python_object(code, ty_map, imports, opt, obj_info, self.render_type)

    def generate_import_section_block(
        self, code: CodeBlock, imports: list[ImportItem], opt: Options
    ) -> None:
        """Emit Python ``import`` statements for the collected imports."""
        G.generate_python_import_section(code, imports, opt)

    def generate_all_block(self, code: CodeBlock, names: set[str], opt: Options) -> None:
        """Emit a Python ``__all__`` list."""
        G.generate_python_all(code, names, opt)

    def generate_export_block(self, code: CodeBlock) -> None:
        """Emit a Python submodule re-export for an ``export/<submodule>`` block."""
        G.generate_python_export(code)

    def generate_api_file(
        self,
        code_blocks: list[CodeBlock],
        ty_map: dict[str, str],
        module_name: str,
        object_infos: list[ObjectInfo],
        init_cfg: InitConfig,
        is_root: bool,
    ) -> str:
        """Return text appended to a scaffolded ``_ffi_api.py``."""
        return G.generate_python_ffi_api(
            code_blocks, ty_map, module_name, object_infos, init_cfg, is_root, self.syntax
        )

    def generate_init_file(
        self, code_blocks: list[CodeBlock], module_name: str, submodule: str
    ) -> str:
        """Return text appended to a scaffolded ``__init__.py``."""
        return G.generate_python_init(code_blocks, module_name, submodule, self.syntax)
