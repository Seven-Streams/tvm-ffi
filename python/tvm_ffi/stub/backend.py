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
   text (Python ``def``/``class`` vs Rust ``fn``/``struct``/``impl``), rendering
   a :class:`~tvm_ffi.core.TypeSchema` into a target-language type expression
   (``T | None`` vs ``Option<T>``), and modelling that language's imports.

A :class:`Backend` encapsulates concern (2); ``cli.py`` drives concern (1) and
delegates every act of emitting text — and every act of collecting imports — to
the active backend. The import collector is opaque to the pipeline: ``cli.py``
asks the backend to create one, seed it from ``import-object`` directives, and
later render it, but never reaches inside. Adding a language is therefore
"implement one more :class:`Backend`" rather than forking the pipeline.

The Python backend lives in :mod:`.python_backend`; the Rust backend lives in
:mod:`.rust_backend` (currently a skeleton marking the work still to be done).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from . import consts as C
from .python_backend import PythonBackend
from .rust_backend import RustBackend

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema

    from .file_utils import CodeBlock
    from .utils import FuncInfo, InitConfig, ObjectInfo, Options

#: A per-render type mapper: given an FFI origin name (e.g. ``"Array"``), return
#: the rendered target-language name and, as a side effect, record any import it
#: requires. Backends build one of these per block.
TyRenderer = Callable[[str], str]


@runtime_checkable
class Backend(Protocol):
    """Language-specific rendering surface used by the stub-generation pipeline.

    Each method that ends in ``_block`` mutates ``code.lines`` in place to hold
    the freshly generated text between the ``begin``/``end`` markers. The
    ``*_file`` methods return whole-file scaffolding text used by ``--init``
    mode. Implementations must be stateless with respect to a single file so the
    pipeline can process files in any order.

    The ``imports`` parameter threaded through the ``_block`` methods is an
    opaque import collector created by :meth:`new_imports`; only the backend
    that created it understands its contents.
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

    # --- import collection (representation is backend-private) --------------

    def new_imports(self) -> Any:
        """Create a fresh, empty import collector for one file."""
        ...

    def add_imported_object(
        self, imports: Any, name: str, type_checking_only: str, alias: str
    ) -> None:
        """Record an ``import-object`` directive (raw directive fields) into ``imports``."""
        ...

    def canonical_type_name(self, type_key: str) -> str:
        """Return the canonical identifier for a locally-defined type key.

        Used to suppress importing a type the file itself defines, and to feed
        the public-export list. Must be comparable to the names produced while
        collecting imports.
        """
        ...

    def extra_export_names(self, imports: Any) -> set[str]:
        """Return extra public-export names implied by the collected imports."""
        ...

    # --- per-block generation (mutates `code.lines`) ------------------------

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: Any,
        opt: Options,
    ) -> None:
        """Emit free function signatures for a ``global/<prefix>`` block."""
        ...

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: Any,
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit a type definition (fields + methods + init) for an ``object/<key>`` block."""
        ...

    def generate_import_section_block(
        self, code: CodeBlock, imports: Any, opt: Options, defined_types: set[str]
    ) -> None:
        """Emit the import/`use` statements collected while rendering other blocks.

        ``defined_types`` holds the canonical names defined in this file so the
        backend can drop imports that would shadow a local definition.
        """
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
