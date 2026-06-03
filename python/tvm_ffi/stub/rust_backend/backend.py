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
"""The Rust code-generation backend for ``tvm-ffi-stubgen``.

:class:`RustBackend` implements the :class:`tvm_ffi.stub.backend.Backend`
protocol. Skeleton only — the method bodies that need a Rust implementation
raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .. import consts as C

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema

    from ..backend import TyRenderer
    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, ObjectInfo, Options


class RustBackend:
    """Backend that emits Rust binding stubs.

    Skeleton only — the method bodies that need a Rust implementation raise
    :class:`NotImplementedError`. Per the design decisions on this branch:

    * ``Union[...]`` rendering is *not* supported and must raise a clear error.
    * imports are modelled separately from Python (a Rust ``use`` collector).
    """

    name = "rust"
    syntax = C.RUST_SYNTAX

    #: TODO(rust): replace with the real FFI-origin -> Rust-type name map.
    _DEFAULT_TY_MAP: ClassVar[dict[str, str]] = {}

    def default_ty_map(self) -> dict[str, str]:
        """Return the default FFI-origin -> Rust-type name map."""
        return dict(self._DEFAULT_TY_MAP)

    def render_type(self, schema: TypeSchema, ty_render: TyRenderer) -> str:
        """Render a type schema as a Rust type expression. TODO(rust)."""
        raise NotImplementedError("RustBackend.render_type: implement Rust type rendering")

    def new_imports(self) -> Any:
        """Create a Rust ``use`` collector. TODO(rust)."""
        raise NotImplementedError("RustBackend.new_imports")

    def add_imported_object(
        self, imports: Any, name: str, type_checking_only: str, alias: str
    ) -> None:
        """Record an ``import-object`` directive into the Rust collector. TODO(rust)."""
        raise NotImplementedError("RustBackend.add_imported_object")

    def canonical_type_name(self, type_key: str) -> str:
        """Return the canonical Rust path for a defined type key. TODO(rust)."""
        raise NotImplementedError("RustBackend.canonical_type_name")

    def extra_export_names(self, imports: Any) -> set[str]:
        """Return extra Rust re-export names implied by the imports. TODO(rust)."""
        raise NotImplementedError("RustBackend.extra_export_names")

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: Any,
        opt: Options,
    ) -> None:
        """Emit Rust function signatures for a ``global/<prefix>`` block. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_global_funcs_block")

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: Any,
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit a Rust ``struct``/``impl`` for an ``object/<key>`` block. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_object_block")

    def generate_import_section_block(
        self, code: CodeBlock, imports: Any, opt: Options, defined_types: set[str]
    ) -> None:
        """Emit Rust ``use`` statements for the collected imports. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_import_section_block")

    def generate_all_block(self, code: CodeBlock, names: set[str], opt: Options) -> None:
        """Emit Rust public re-exports. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_all_block")

    def generate_export_block(self, code: CodeBlock) -> None:
        """Emit a Rust submodule re-export for an ``export/<submodule>`` block. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_export_block")

    def generate_api_file(
        self,
        code_blocks: list[CodeBlock],
        ty_map: dict[str, str],
        module_name: str,
        object_infos: list[ObjectInfo],
        init_cfg: InitConfig,
        is_root: bool,
    ) -> str:
        """Return text appended to a scaffolded Rust API module. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_api_file")

    def generate_init_file(
        self, code_blocks: list[CodeBlock], module_name: str, submodule: str
    ) -> str:
        """Return text appended to a scaffolded Rust module entry. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_init_file")
