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
"""Pluggable code-generation backends for ``tvm-ffi-stubgen``.

The stub generator separates two concerns:

1. *Language-agnostic* infrastructure — reading the FFI reflection registry
   (:mod:`.lib_state`), parsing/writing marker blocks (:mod:`.file_utils`), and
   the abstract object/function metadata (:class:`.utils.ObjectInfo`,
   :class:`.utils.FuncInfo`). None of this knows or cares about the target
   language.
2. *Language-specific* rendering — turning that metadata into concrete source
   text (Python ``def``/``class`` vs Rust ``fn``/``struct``/``impl``) and
   rendering a :class:`~tvm_ffi.core.TypeSchema` into a target-language type
   expression (``T | None`` vs ``Option<T>``).

A :class:`Backend` encapsulates concern (2). ``cli.py`` drives concern (1) and
delegates every act of emitting text to the active backend. Adding Rust support
is therefore "implement one more :class:`Backend`" rather than forking the
pipeline.

Status: this is a *draft* seam. The Python path still calls :mod:`.codegen`
directly; :class:`PythonBackend` wraps those functions so the interface is
exercised and verifiable, but ``cli.py`` is not yet rewired to go through a
backend. :class:`RustBackend` is a skeleton marking the work to be done.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Protocol, runtime_checkable

from . import codegen as G
from . import consts as C

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema

    from .file_utils import CodeBlock
    from .utils import FuncInfo, ImportItem, InitConfig, ObjectInfo, Options

#: A per-render type mapper: given an FFI origin name (e.g. ``"Array"``), return
#: the rendered target-language name and, as a side effect, record any import it
#: requires. Backends build one of these per block (see
#: ``codegen._type_suffix_and_record`` for the Python implementation).
TyRenderer = Callable[[str], str]


@runtime_checkable
class Backend(Protocol):
    """Language-specific rendering surface used by the stub-generation pipeline.

    Each method that ends in ``_block`` mutates ``code.lines`` in place to hold
    the freshly generated text between the ``begin``/``end`` markers. The
    ``*_file`` methods return whole-file scaffolding text used by ``--init``
    mode. Implementations must be stateless with respect to a single file so the
    pipeline can process files in any order.
    """

    #: Short identifier, e.g. ``"python"`` or ``"rust"``.
    name: str

    #: Comment-marker syntax for the files this backend emits.
    syntax: C.MarkerSyntax

    def default_ty_map(self) -> dict[str, str]:
        """Return the default FFI-origin -> target-type name map for this language."""
        ...

    def render_type(self, schema: TypeSchema, ty_render: TyRenderer) -> str:
        """Render a single :class:`TypeSchema` into a target-language type expression.

        This is the core seam: it walks ``schema.origin`` / ``schema.args`` and
        produces e.g. ``"int | None"`` (Python) or ``"Option<i64>"`` (Rust).
        ``ty_render`` maps a leaf origin name and records the import it needs.
        """
        ...

    # --- per-block generation (mutates `code.lines`) ------------------------

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: list[ImportItem],
        opt: Options,
    ) -> None:
        """Emit free function signatures for a ``global/<prefix>`` block."""
        ...

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: list[ImportItem],
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit a type definition (fields + methods + init) for an ``object/<key>`` block."""
        ...

    def generate_import_section_block(
        self, code: CodeBlock, imports: list[ImportItem], opt: Options
    ) -> None:
        """Emit the import/`use` statements collected while rendering other blocks."""
        ...

    def generate_all_block(self, code: CodeBlock, names: set[str], opt: Options) -> None:
        """Emit the public-export list (Python ``__all__``; Rust re-exports)."""
        ...

    def generate_export_block(self, code: CodeBlock) -> None:
        """Emit a submodule re-export for an ``export/<submodule>`` block."""
        ...

    # --- whole-file scaffolding (used by `--init` mode) ---------------------

    def generate_api_file(
        self,
        code_blocks: list[CodeBlock],
        ty_map: dict[str, str],
        module_name: str,
        object_infos: list[ObjectInfo],
        init_cfg: InitConfig,
        is_root: bool,
    ) -> str:
        """Return text appended to a freshly scaffolded API file (Python ``_ffi_api.py``)."""
        ...

    def generate_init_file(
        self, code_blocks: list[CodeBlock], module_name: str, submodule: str
    ) -> str:
        """Return text appended to a freshly scaffolded package entry (Python ``__init__.py``)."""
        ...


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


class RustBackend:
    """Backend that emits Rust binding stubs.

    Skeleton only. The seams that need a Rust implementation are marked below.
    A reasonable starting point for :meth:`render_type` is a recursive walk that
    maps well-known FFI origins to Rust constructs::

        Optional[T]   -> Option<T>
        Array[T]      -> Vec<T>            (or &[T])
        Map[K, V]     -> HashMap<K, V>
        Callable[A]R  -> Function          (no direct generic-fn stub)
        Union[...]    -> (needs a policy: enum / Any-equivalent)

    and falls back to ``ty_render(origin)`` for leaf/object types so imports
    (``use`` paths) are recorded the same way the Python backend records them.
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

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: list[ImportItem],
        opt: Options,
    ) -> None:
        """Emit Rust function signatures for a ``global/<prefix>`` block. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_global_funcs_block")

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: list[ImportItem],
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit a Rust ``struct``/``impl`` for an ``object/<key>`` block. TODO(rust)."""
        raise NotImplementedError("RustBackend.generate_object_block")

    def generate_import_section_block(
        self, code: CodeBlock, imports: list[ImportItem], opt: Options
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


_BACKENDS: dict[str, Callable[[], Backend]] = {
    "python": PythonBackend,
    "rust": RustBackend,
}


def get_backend(target: str) -> Backend:
    """Resolve a backend by name (``"python"`` / ``"rust"``)."""
    try:
        factory = _BACKENDS[target]
    except KeyError as e:
        known = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown stubgen backend: {target!r}. Known backends: {known}") from e
    return factory()
