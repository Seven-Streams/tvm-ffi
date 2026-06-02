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
"""Rust code generation for the ``tvm-ffi-stubgen`` tool.

This module owns Rust codegen orchestration. Low-level rendering helpers live in
``rust_generator.utils`` so the block-generation pipeline here stays focused on
directive handling and source assembly.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .. import consts as C
from .utils import (
    RustImports,
    UnsupportedTypeError,
    _class_is_mutable,
    _deref_impl,
    _packed_args_expr,
    _packed_call_lines,
    _rust_ident,
    _use,
    render_rust_type,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, ObjectInfo, Options


@dataclasses.dataclass
class _ObjectRenderer:
    """Renders one ``object/<key>`` block into Rust source lines.

    Bundles the per-object rendering environment -- the import collector, the
    ``ty_map``, the resolved struct/ref identifiers, and the mutability shape --
    that is invariant across the whole of one object's rendering. Holding it as
    state (rather than threading it through every helper signature) keeps the
    method signatures small and makes adding new render context a one-field
    change. The type-render entry points (:meth:`render_field` /
    :meth:`render_param`) feed the stateless :func:`.utils.render_rust_type` seam.
    """

    info: ObjectInfo
    leaf: str
    obj_struct: str
    base_type: str
    is_root: bool
    mutable: bool
    imports: RustImports
    ty_map: dict[str, str]

    def _ty_render(self, origin: str) -> str:
        """Resolve a leaf origin to its Rust leaf name and record its ``use``."""
        return self.imports.record(self.ty_map.get(origin, origin))

    def render_field(self, schema: TypeSchema) -> str:
        """Render a field/return type (owning form: a top-level ``Any`` is ``Any``)."""
        return render_rust_type(schema, self._ty_render)

    def render_param(self, schema: TypeSchema) -> str:
        """Render an argument type (a top-level ``Any`` is the non-owning ``AnyView``)."""
        if schema.origin == "Any":
            _use(self.imports, "tvm_ffi::AnyView")
            return "AnyView"
        return render_rust_type(schema, self._ty_render)

    def body(self) -> list[str]:
        """Build the Rust source lines for the object (raises on unsupported types)."""
        # Boilerplate `use`s the generated items rely on. Recorded first so they
        # claim their (un-aliased) leaf names before any field/method type does.
        _use(self.imports, "std::ops::Deref")
        _use(self.imports, "tvm_ffi::object::ObjectArc")
        _use(self.imports, "tvm_ffi::object::ObjectCore")
        _use(self.imports, "tvm_ffi::derive::ObjectRef", alias="DeriveObjectRef")
        _use(self.imports, "tvm_ffi::tvm_ffi_sys::TVMFFIObject")
        if self.is_root:
            _use(self.imports, "tvm_ffi::object::Object")
        if self.mutable:
            _use(self.imports, "std::ops::DerefMut")

        leaf, obj_struct, base_type = self.leaf, self.obj_struct, self.base_type
        lines: list[str] = []
        lines += ["#[repr(C)]", f"pub struct {obj_struct} {{", f"    base: {base_type},"]
        for field in self.info.fields:
            lines.append(f"    pub {_rust_ident(field.name)}: {self.render_field(field)},")
        lines += ["}", ""]

        lines += [
            f"unsafe impl ObjectCore for {obj_struct} {{",
            f'    const TYPE_KEY: &\'static str = "{self.info.type_key}";',
            "",
            "    fn type_index() -> i32 {",
            "        lookup_type_index(Self::TYPE_KEY)",
            "    }",
            "",
            "    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {",
            f"        {base_type}::object_header_mut(&mut this.base)",
            "    }",
            "}",
            "",
        ]

        lines += [
            "#[repr(C)]",
            "#[derive(DeriveObjectRef, Clone)]",
            f"pub struct {leaf} {{",
            f"    data: ObjectArc<{obj_struct}>,",
            "}",
            "",
        ]

        lines += _deref_impl(leaf, obj_struct, "data", self.mutable)
        if not self.is_root:
            lines += _deref_impl(obj_struct, base_type, "base", self.mutable)

        lines += self._impl_block()

        if lines and lines[-1] == "":
            lines.pop()
        return lines

    def _impl_block(self) -> list[str]:
        """Emit `impl <T> { new; methods }`; empty list when there's nothing to emit."""
        init_method = next(
            (m for m in self.info.methods if m.schema.name.rsplit(".", 1)[-1] == "__ffi_init__"),
            None,
        )
        methods = [
            m for m in self.info.methods if m.schema.name.rsplit(".", 1)[-1] != "__ffi_init__"
        ]
        if not self.info.has_init and not methods:
            return []

        _use(self.imports, "tvm_ffi::Result")

        inner: list[str] = []
        if self.info.has_init:
            inner += self._new_fn(init_method)
            if methods:
                inner.append("")
        for i, method in enumerate(methods):
            inner += self._method_fn(method)
            if i != len(methods) - 1:
                inner.append("")

        return [
            f"impl {self.leaf} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _new_fn(self, init_method: FuncInfo | None) -> list[str]:
        """Emit `fn new(...) -> Result<Self>` calling reflected `__ffi_init__`."""
        if init_method is not None:
            arg_schemas = list(init_method.schema.args[1:]) if init_method.schema.args else []
            params = [(f"_{i}", self.render_param(s)) for i, s in enumerate(arg_schemas)]
        else:
            params = [
                (_rust_ident(f.name), self.render_param(f.schema)) for f in self.info.init_fields
            ]
        sig = ", ".join(f"{n}: {t}" for n, t in params)
        if params:
            _use(self.imports, "tvm_ffi::AnyView")
        packed = _packed_args_expr(params, is_member=False)
        getter = f'    let ctor = get_type_method({self.obj_struct}::TYPE_KEY, "__ffi_init__")?;'
        return [
            f"pub fn new({sig}) -> Result<Self> {{",
            *_packed_call_lines("ctor", getter, packed, "Self"),
            "}",
        ]

    def _method_fn(self, method: FuncInfo) -> list[str]:
        """Emit one reflected method (instance or static) on `impl <T>`."""
        ffi_name = method.schema.name.rsplit(".", 1)[-1]
        rust_name = _rust_ident(ffi_name)
        args = method.schema.args or ()
        # Return type uses the owning render (a top-level `Any` stays `Any`).
        ret = self.render_field(args[0]) if args else "Any"
        rest = list(args[1:])
        if method.is_member:
            rest = rest[1:]
        params = [(f"_{i}", self.render_param(p)) for i, p in enumerate(rest)]

        self_recv = "&mut self" if self.mutable else "&self"
        if method.is_member:
            sig_parts = [self_recv, *[f"{n}: {t}" for n, t in params]]
        else:
            sig_parts = [f"{n}: {t}" for n, t in params]
        if method.is_member or params:
            _use(self.imports, "tvm_ffi::AnyView")
        packed = _packed_args_expr(params, method.is_member)
        getter = f'    let f = get_type_method({self.obj_struct}::TYPE_KEY, "{ffi_name}")?;'
        header = f"pub fn {rust_name}({', '.join(sig_parts)}) -> Result<{ret}> {{"
        return [header, *_packed_call_lines("f", getter, packed, ret), "}"]


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit a Rust ``struct``/``impl`` binding for an ``object/<key>`` block.

    Emits the standard binding shape: ``<T>Obj`` (``#[repr(C)]`` with the parent
    embedded as ``base``), an ``ObjectCore`` impl, the ``<T>`` ref (wrapping
    ``ObjectArc``), ``Deref``/``DerefMut`` (the latter only for mutable classes),
    and ``impl <T>`` with a ``new`` constructor + reflected methods. On an
    :class:`UnsupportedTypeError` the whole object is skipped with a warning.
    """
    assert len(code.lines) >= 2
    type_key = obj_info.type_key
    assert isinstance(type_key, str)
    leaf = type_key.rsplit(".", 1)[-1]
    obj_struct = f"{leaf}Obj"
    parent_key = obj_info.parent_type_key
    is_root = parent_key in (None, "ffi.Object")
    if is_root:
        base_type = "Object"
    else:
        assert isinstance(parent_key, str)
        base_type = f"{parent_key.rsplit('.', 1)[-1]}Obj"
    mutable, mixed = _class_is_mutable(obj_info)

    # Render into a local collector so a skip leaves `imports` untouched. Seeding
    # from `imports.items` is enough to carry forward every prior `use`: a fresh
    # `RustImports` rebuilds its collision trackers (`_binding_of`/`_used_names`)
    # from `items` in `__post_init__`. `items` is the single source of truth, so
    # the writeback below only needs to copy `items` back (see the writeback note).
    local = RustImports(items=list(imports.items))
    renderer = _ObjectRenderer(
        info=obj_info,
        leaf=leaf,
        obj_struct=obj_struct,
        base_type=base_type,
        is_root=is_root,
        mutable=mutable,
        imports=local,
        ty_map=ty_map,
    )

    try:
        body = renderer.body()
    except UnsupportedTypeError as e:
        # When the whole object is skipped, the `mixed` warning below is moot, so
        # the early return suppresses it on purpose.
        print(
            f"{C.TERM_YELLOW}[Skipped] object {type_key}: "
            f"unsupported type {e.origin!r}{C.TERM_RESET}"
        )
        code.lines = [code.lines[0], code.lines[-1]]
        return

    if mixed:
        print(
            f"{C.TERM_YELLOW}[Warning] object {type_key}: mixed read-only/read-write "
            f"fields; treating the whole type as read-only{C.TERM_RESET}"
        )

    # Writeback note: copy only `items` -- the next object block re-seeds a fresh
    # `local` from it and rebuilds the trackers, so `imports`'s own (now stale)
    # `_binding_of`/`_used_names` are never read again and need not be updated.
    imports.items[:] = local.items
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust object layout needs no `opt`


# --- import section (`use` statements) --------------------------------------


def generate_rust_import_section(
    code: CodeBlock,
    imports: RustImports,
    opt: Options,
    defined_types: set[str],
) -> None:
    """Render the collected ``use`` statements into an ``import-section`` block.

    Imports whose target is a type *defined in this same file* are dropped
    (``RustUse.full_name in defined_types``) -- you don't ``use`` what you
    define locally. The remaining uses are deduped and sorted for a stable,
    one-per-line rendering (Rust has no ``TYPE_CHECKING`` split).
    """
    assert len(code.lines) >= 2
    # `RustImports.record` never admits a bare / no-import type into `items`, so
    # every `as_use_line()` here is a real, non-empty `use` line.
    use_lines = sorted(
        {item.as_use_line() for item in imports.items if item.full_name not in defined_types}
    )
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in use_lines],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust needs no indent/TYPE_CHECKING handling


# --- whole-file scaffolding (`--init` mode) ---------------------------------

#: Shared per-file helper functions. Written fully-qualified with
#: zero `use`s so they never clash with the import-section block (which carries
#: every object-driven `use`). `lookup_type_index` resolves+caches a type index;
#: `get_type_method` pulls a reflected method off the type's method table.
_RUST_HELPERS = """fn lookup_type_index(type_key: &'static str) -> i32 {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<&'static str, i32>>,
    > = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(v) = cache.lock().unwrap().get(type_key) {
        return *v;
    }
    let arg = unsafe { tvm_ffi::tvm_ffi_sys::TVMFFIByteArray::from_str(type_key) };
    let mut tindex = 0;
    let ret = unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITypeKeyToIndex(&arg, &mut tindex) };
    assert_eq!(ret, 0, "type key `{type_key}` is not registered");
    cache.lock().unwrap().insert(type_key, tindex);
    tindex
}

fn get_type_method(
    type_key: &'static str,
    method_name: &str,
) -> tvm_ffi::Result<tvm_ffi::Function> {
    let type_index = lookup_type_index(type_key);
    unsafe {
        let info = tvm_ffi::tvm_ffi_sys::TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::TYPE_ERROR,
                &format!("no type info for `{type_key}`"),
                "",
            ));
        }
        let info = &*info;
        for i in 0..info.num_methods {
            let mi = &*info.methods.add(i as usize);
            if mi.name.as_str() == method_name {
                if !<tvm_ffi::Function as tvm_ffi::type_traits::AnyCompatible>::check_any_strict(
                    &mi.method,
                ) {
                    return Err(tvm_ffi::Error::new(
                        tvm_ffi::TYPE_ERROR,
                        &format!("method `{method_name}` on `{type_key}` is not a Function"),
                        "",
                    ));
                }
                return Ok(<tvm_ffi::Function as tvm_ffi::type_traits::AnyCompatible>::copy_from_any_view_after_check(&mi.method));
            }
        }
    }
    Err(tvm_ffi::Error::new(
        tvm_ffi::TYPE_ERROR,
        &format!("method `{method_name}` not found on `{type_key}`"),
        "",
    ))
}"""


def generate_rust_api_file(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: object,
    is_root: bool,
    syntax: C.MarkerSyntax = C.RUST_SYNTAX,
) -> str:
    """Scaffold a single Rust binding file (option A: one file per module prefix).

    A fresh file gets the ``#![allow(...)]`` header. A ``helpers`` marker block
    (filled by :func:`generate_rust_helpers` during stage processing with the
    shared support functions), an ``import-section`` marker, and an
    ``object/<type_key>`` marker per registered type are added if missing. Putting
    the helpers in a marker means they are (re)generated on every run -- even into
    a pre-existing file -- rather than only when the file is brand new. No
    ``global``/``__all__``/``export`` blocks are emitted.
    """
    append = ""
    if not code_blocks:
        append += "#![allow(dead_code, unused_imports)]\n"
        append += f"\n//! FFI bindings for `{module_name}` (generated by tvm-ffi-stubgen).\n\n"
    if not any(c.kind == "helpers" for c in code_blocks):
        append += f"{syntax.begin} helpers\n{syntax.end}\n\n"
    if not any(c.kind == "import-section" for c in code_blocks):
        append += f"{syntax.begin} import-section\n{syntax.end}\n\n"
    defined = {c.param for c in code_blocks if c.kind == "object"}
    for info in object_infos:
        type_key = info.type_key
        if type_key is None or type_key in defined:
            continue
        append += f"{syntax.begin} object/{type_key}\n{syntax.end}\n\n"
    _ = (ty_map, init_cfg, is_root)  # unused for the Rust single-file layout
    return append


def generate_rust_helpers(code: CodeBlock, opt: Options) -> None:
    """Fill a ``helpers`` block with the shared per-file support functions."""
    assert len(code.lines) >= 2
    indent = " " * code.indent
    body = _RUST_HELPERS.split("\n")
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt


def generate_rust_init(
    code_blocks: list[CodeBlock],
    module_name: str,
    submodule: str = "mod",
    syntax: C.MarkerSyntax = C.RUST_SYNTAX,
) -> str:
    """No init/re-export file for Rust (option A: the API file *is* the module)."""
    _ = (code_blocks, module_name, submodule, syntax)
    return ""


# --- module-tree stitching (auto-form `pub mod` declarations) ----------------


def finalize_rust_module_tree(init_path: Path, prefixes: set[str]) -> None:
    """Stitch the generated tree under ``init_path`` into a valid Rust module tree.

    For every generated prefix (e.g. ``a.b.c``) and each of its ancestor dirs,
    ensure the parent's ``mod.rs`` declares ``pub mod <child>;`` -- creating
    intermediate ``mod.rs`` files for prefixes that hold no types themselves, and
    a root ``init_path/mod.rs`` declaring the top-level modules. Declarations are
    idempotent (appended only when absent).

    The only thing left to the user is one ``mod``/``pub mod`` line at their crate
    root mounting ``init_path`` (stubgen cannot safely edit ``lib.rs``/``main.rs``).
    """
    children: dict[Path, set[str]] = {}
    for prefix in prefixes:
        segs = [s for s in prefix.split(".") if s]
        for i, seg in enumerate(segs):
            parent = init_path.joinpath(*segs[:i])
            children.setdefault(parent, set()).add(seg)

    for parent, names in children.items():
        parent.mkdir(parents=True, exist_ok=True)
        mod_rs = parent / "mod.rs"
        existing = mod_rs.read_text(encoding="utf-8") if mod_rs.exists() else ""
        to_add = [f"pub mod {n};" for n in sorted(names) if f"pub mod {n};" not in existing]
        if not to_add:
            continue
        text = existing
        if text and not text.endswith("\n"):
            text += "\n"
        if text.strip():  # separate from any existing bindings
            text += "\n"
        text += "\n".join(to_add) + "\n"
        mod_rs.write_text(text, encoding="utf-8")
