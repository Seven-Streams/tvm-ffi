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

Codegen orchestration lives here; low-level rendering helpers live in
``rust_generator.utils``.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

from .. import consts as C
from . import consts as C_RUST
from .utils import (
    RustImports,
    UnsupportedTypeError,
    _deref_impl,
    _packed_args_expr,
    _packed_call_lines,
    render_rust_type,
    rust_identifier,
    rust_type_key_path,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, NamedTypeSchema, ObjectInfo, Options


def _rust_string_literal(s: str) -> str:
    """Escape ``s`` as a double-quoted Rust string literal."""
    out = ['"']
    for ch in s:
        if ch in ('"', "\\"):
            out.append("\\" + ch)
        elif ch.isprintable():
            out.append(ch)
        else:
            out.append(f"\\u{{{ord(ch):x}}}")
    out.append('"')
    return "".join(out)


@dataclasses.dataclass
class _ObjectRenderer:
    """Renders one ``object/<key>`` block into Rust source lines.

    Holds the per-object rendering context (imports, ``ty_map``, resolved
    names) so helper methods don't have to thread it through.
    """

    info: ObjectInfo
    leaf: str
    obj_struct: str
    base_type: str
    is_root: bool
    imports: RustImports
    ty_map: dict[str, str]
    #: Module segments of the file this object lands in (its type key minus the
    #: leaf; ``tirx.transform.X`` -> ``("tirx", "transform")``): one file per
    #: prefix, mounted at ``<out>/<seg>/.../mod.rs`` (see ``cli`` and
    #: :func:`finalize_rust_module_tree`).
    mod_segments: tuple[str, ...]

    def _ty_render(self, origin: str) -> str:
        """Resolve a leaf origin to its Rust name and record its ``use``.

        Unmapped dotted names (object type keys) resolve against the generated
        module tree via :meth:`_generated_type_path`. An unmapped bare origin
        (e.g. ``const char*``) or a ``ctypes.*`` sentinel (``ctypes.c_void_p``
        -- ``void*`` -- is dotted but is not an object key and has no Rust
        rendering) raises. Value/argument renderers catch those schema failures
        and use ``Any``/``AnyView`` without confusing them with a named IR type.
        """
        mapped = self.ty_map.get(origin)
        if mapped is None:
            if "." not in origin or origin.startswith("ctypes."):
                raise UnsupportedTypeError(origin)
            mapped = self._generated_type_path(origin)
        elif "::" not in mapped and "." in mapped:
            # Dotted map targets name another generated reflected type. Rust
            # crate paths use `::` and pass through unchanged.
            mapped = self._generated_type_path(mapped)
        return self.imports.record(mapped)

    def _generated_type_path(self, type_key: str, *, object_data: bool = False) -> str:
        """Resolve a generated-tree type key to a path valid from this file.

        A bare ``use ir::Expr;`` is broken in edition 2021 (it resolves to an
        extern crate ``ir``, or silently captures an equally-named *submodule*),
        so cross-module references must anchor at the shared generated root:
        ``super::`` once per segment of this file's own module path, then the
        referenced key's full path (``super::ir::Expr`` from ``tirx/mod.rs``,
        ``super::super::ir::Expr`` from ``tirx/transform/mod.rs``). A key in
        *this* file's module is a local item: bare leaf, no ``use``. A head
        with a :data:`~.consts.RUST_MOD_MAP` rewrite (builtin ``ffi.*`` keys)
        lives in the crate, not the generated tree, and passes through for
        :class:`~.utils.RustUse` to rewrite.
        """
        mapped = C_RUST.RUST_TY_MAP_DEFAULTS.get(type_key)
        if mapped is not None:
            if object_data:
                raise UnsupportedTypeError(
                    type_key, f"Rust has no exposed object container for builtin {type_key!r}"
                )
            return mapped
        mod, _, type_leaf = type_key.rpartition(".")
        type_name = rust_identifier(f"{type_leaf}Obj" if object_data else type_leaf)
        if tuple(mod.split(".")) == self.mod_segments:
            return type_name
        supers = "super::" * len(self.mod_segments)
        path = rust_type_key_path(type_key, object_data=object_data)
        return f"{supers or 'self::'}{path}"

    def render_result(self, schema: TypeSchema) -> tuple[str, bool]:
        """Render an owning result, falling back to ``Any`` when it is dynamic.

        In particular, a container containing ``Any`` cannot be represented as
        a Rust ``Array<T>``/``Map<K, V>`` without lying about its element type.
        Keeping the complete value as owning ``Any`` is lossless and lets
        callers inspect it with the dynamic APIs or ``structural_walk``.
        """
        checkpoint = len(self.imports.items)
        try:
            if schema.origin == "Any":
                raise UnsupportedTypeError("Any")
            return render_rust_type(schema, self._ty_render), False
        except UnsupportedTypeError:
            del self.imports.items[checkpoint:]
            return "::tvm_ffi::Any", True

    def render_param(self, schema: TypeSchema) -> tuple[str, bool]:
        """Render an argument, using ``AnyView`` for dynamic schemas."""
        checkpoint = len(self.imports.items)
        try:
            if schema.origin == "Any":
                raise UnsupportedTypeError("Any")
            return render_rust_type(schema, self._ty_render), False
        except UnsupportedTypeError:
            del self.imports.items[checkpoint:]
            return "::tvm_ffi::AnyView", True

    def body(self) -> list[str]:
        """Build one opaque object binding and its reflection-backed API."""
        if self.is_root:
            self.base_type = "::tvm_ffi::Object"
        else:
            parent = self.info.parent_type_key
            assert isinstance(parent, str)
            self.base_type = self.imports.record(
                self._generated_type_path(parent, object_data=True)
            )

        leaf, obj_struct, base_type = self.leaf, self.obj_struct, self.base_type
        lines = [
            "#[repr(C)]",
            "#[derive(::tvm_ffi::derive::Object)]",
            f"#[type_key = {_rust_string_literal(self.info.type_key)}]",
            f"pub struct {obj_struct} {{",
            f"    base: {base_type},",
        ]
        lines += [
            "}",
            "",
            "#[repr(C)]",
            "#[derive(::tvm_ffi::derive::ObjectRef, Clone)]",
            f"pub struct {leaf} {{",
            f"    data: ::tvm_ffi::ObjectArc<{obj_struct}>,",
            "}",
            "",
        ]

        lines += _deref_impl(leaf, obj_struct, "data")
        if not self.is_root:
            lines += _deref_impl(obj_struct, base_type, "base")
            lines += self._upcast_lines()

        methods = self._non_overloaded_methods()
        lines += self._object_impl_block([method for method in methods if method.is_member])
        lines += self._ref_impl_block([method for method in methods if not method.is_member])

        lines.pop()  # every section above ends with a `""` separator
        return lines

    def _field_fn(self, field: NamedTypeSchema) -> list[str]:
        """Emit one safe accessor backed by the field's owning FFI getter."""
        ret, dynamic = self.render_result(field)
        name = field.name
        rust_name = rust_identifier(name)
        call = (
            f"::tvm_ffi::object::get_reflected_field_unchecked(self, {_rust_string_literal(name)})"
        )
        body = f"unsafe {{ {call} }}" if dynamic else f"unsafe {{ {call} }}?.try_into()"
        return [
            f"pub fn {rust_name}(&self) -> ::tvm_ffi::Result<{ret}> {{",
            f"    {body}",
            "}",
        ]

    @staticmethod
    def _impl_block(target: str, sections: list[list[str]]) -> list[str]:
        """Wrap non-empty method sections in one inherent ``impl`` block."""
        if not sections:
            return []
        inner: list[str] = []
        for i, section in enumerate(sections):
            if i:
                inner.append("")
            inner += section
        return [
            f"impl {target} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _non_overloaded_methods(self) -> list[FuncInfo]:
        """Return methods Rust can name uniquely, skipping overloaded groups."""
        groups: dict[tuple[str, bool], list[FuncInfo]] = {}
        for method in self.info.methods:
            ffi_name = method.schema.name.rsplit(".", 1)[-1]
            if ffi_name == "__ffi_init__":
                continue
            try:
                rust_name = rust_identifier(ffi_name)
            except UnsupportedTypeError as err:
                print(
                    f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping "
                    f"Rust method {ffi_name!r}: {err}{C.TERM_RESET}"
                )
                continue
            groups.setdefault((rust_name, method.is_member), []).append(method)

        methods: list[FuncInfo] = []
        for (rust_name, _is_member), overloads in groups.items():
            if len(overloads) == 1:
                methods.append(overloads[0])
                continue
            print(
                f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping "
                f"overloaded Rust method {rust_name!r}; use Function reflection explicitly"
                f"{C.TERM_RESET}"
            )
        return methods

    def _object_impl_block(self, methods: list[FuncInfo]) -> list[str]:
        """Emit own fields and instance methods on ``Obj`` for inheritance."""
        named_sections: list[tuple[str, list[str]]] = []
        for field in self.info.fields:
            try:
                named_sections.append((rust_identifier(field.name), self._field_fn(field)))
            except UnsupportedTypeError as err:
                print(
                    f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping "
                    f"Rust field accessor {field.name!r}: {err}{C.TERM_RESET}"
                )
        named_sections += [
            (rust_identifier(method.schema.name.rsplit(".", 1)[-1]), self._method_fn(method))
            for method in methods
        ]
        named_sections = self._dedupe_sections(named_sections, self.obj_struct)
        return self._impl_block(self.obj_struct, [section for _, section in named_sections])

    def _ref_impl_block(self, methods: list[FuncInfo]) -> list[str]:
        """Emit the reflected constructor and static methods on the ref type."""
        named_sections: list[tuple[str, list[str]]] = []
        constructor = self._constructor_fn()
        if constructor is not None:
            named_sections.append(("ffi_new", constructor))
        named_sections += [
            (rust_identifier(method.schema.name.rsplit(".", 1)[-1]), self._method_fn(method))
            for method in methods
        ]
        named_sections = self._dedupe_sections(named_sections, self.leaf)
        return self._impl_block(self.leaf, [section for _, section in named_sections])

    def _dedupe_sections(
        self, named_sections: list[tuple[str, list[str]]], target: str
    ) -> list[tuple[str, list[str]]]:
        """Keep the first API for each Rust name and warn about later collisions."""
        seen: set[str] = set()
        unique: list[tuple[str, list[str]]] = []
        for name, section in named_sections:
            if name in seen:
                print(
                    f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping duplicate "
                    f"Rust method {name!r} on {target}{C.TERM_RESET}"
                )
                continue
            seen.add(name)
            unique.append((name, section))
        return unique

    def _upcast_lines(self) -> list[str]:
        """`impl From<Leaf> for <ParentRef>` -- offset-0 prefix retype (upcast).

        Sound because `<Leaf>Obj` embeds the parent as its offset-0 `base`, so
        the object pointer is unchanged; only the arc's static type moves
        (ownership transfers, no refcount change). Emitted for derived types
        only -- the parent's ref is the generated `<ParentLeaf>`; a root object
        has no ref-typed parent (its `base` is the bare `Object` data struct).
        """
        parent = self.info.parent_type_key
        assert isinstance(parent, str)
        parent_ref = self.imports.record(self._generated_type_path(parent))
        parent_obj = self.base_type
        return [
            f"impl ::core::convert::From<{self.leaf}> for {parent_ref} {{",
            f"    fn from(x: {self.leaf}) -> {parent_ref} {{",
            f"        let arc = <{self.leaf} as ::tvm_ffi::ObjectRefCore>::into_data(x);",
            "        let up = unsafe {",
            "            ::tvm_ffi::ObjectArc::from_raw("
            f"::tvm_ffi::ObjectArc::into_raw(arc) as *const {parent_obj})",
            "        };",
            f"        <{parent_ref} as ::tvm_ffi::ObjectRefCore>::from_data(up)",
            "    }",
            "}",
            "",
        ]

    def _cached_getter_lines(
        self, fvar: str, ffi_name: str, *, type_attr: bool = False
    ) -> list[str]:
        """Bind ``fvar`` to a reflected callable cached at this call site."""
        lookup = "cached_type_attr" if type_attr else "cached_type_method"
        return [
            f"    let {fvar} = ::tvm_ffi::{lookup}!("
            f"<{self.obj_struct} as ::tvm_ffi::ObjectCore>::type_index(), "
            f"{_rust_string_literal(ffi_name)})?;"
        ]

    @staticmethod
    def _callable_local(params: list[tuple[str, str, bool]]) -> str:
        """Choose an internal function variable that cannot shadow a parameter."""
        used = {name for name, _ty, _dynamic in params}
        name = "__tvm_ffi_func"
        while name in used:
            name = "_" + name
        return name

    def _auto_constructor_params(self) -> list[tuple[str, str, bool]] | None:
        """Render reflected init fields with unique, stable Rust parameter names."""
        field_names = [field.name for field in self.info.init_fields]
        if len(field_names) != len(set(field_names)):
            print(
                f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping typed "
                "Rust constructor because inherited init fields have duplicate names"
                f"{C.TERM_RESET}"
            )
            return None

        rendered: list[tuple[str | None, str, bool]] = []
        used: set[str] = set()
        for field in self.info.init_fields:
            ty, dynamic = self.render_param(field.schema)
            try:
                preferred: str | None = rust_identifier(field.name)
            except UnsupportedTypeError:
                preferred = None
            if preferred is not None:
                used.add(preferred)
            rendered.append((preferred, ty, dynamic))

        params: list[tuple[str, str, bool]] = []
        for index, (preferred, ty, dynamic) in enumerate(rendered):
            chosen = preferred or f"_{index}"
            while preferred is None and chosen in used:
                chosen = "_" + chosen
            used.add(chosen)
            params.append((chosen, ty, dynamic))
        return params

    def _constructor_fn(self) -> list[str] | None:
        """Emit a thin, FFI-backed constructor when reflection exposes one."""
        init_methods = [
            method
            for method in self.info.methods
            if method.schema.name.rsplit(".", 1)[-1] == "__ffi_init__"
        ]
        if len(init_methods) > 1:
            print(
                f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping "
                "overloaded Rust constructor; use Function reflection explicitly"
                f"{C.TERM_RESET}"
            )
            return None

        params: list[tuple[str, str, bool]] = []
        type_attr = not init_methods
        if init_methods:
            method = init_methods[0]
            args = method.schema.args or ()
            for i, schema in enumerate(args[1:]):
                ty, dynamic = self.render_param(schema)
                params.append((f"_{i}", ty, dynamic))
        elif self.info.has_init:
            auto_params = self._auto_constructor_params()
            if auto_params is None:
                return None
            params = auto_params
        else:
            return None

        signature = ", ".join(f"{name}: {ty}" for name, ty, _ in params)
        fvar = self._callable_local(params)
        getter = self._cached_getter_lines(fvar, "__ffi_init__", type_attr=type_attr)
        if type_attr:
            packed = ""
            kwargs = (
                ", ".join(
                    f"({_rust_string_literal(field.name)}, "
                    f"{name if dynamic else f'::tvm_ffi::AnyView::from(&{name})'})"
                    for field, (name, _ty, dynamic) in zip(self.info.init_fields, params)
                )
                or None
            )
        else:
            packed = _packed_args_expr(params, None)
            kwargs = None
        return [
            "/// Construct through the registered `__ffi_init__` function.",
            f"pub fn ffi_new({signature}) -> ::tvm_ffi::Result<Self> {{",
            *_packed_call_lines(fvar, getter, packed, False, kwargs=kwargs),
            "}",
        ]

    def _method_fn(self, method: FuncInfo) -> list[str]:
        """Emit one reflected method on its object or reference wrapper."""
        ffi_name = method.schema.name.rsplit(".", 1)[-1]
        rust_name = rust_identifier(ffi_name)
        args = method.schema.args or ()
        if args:
            ret, dynamic_result = self.render_result(args[0])
        else:
            ret, dynamic_result = "::tvm_ffi::Any", True
        rest = args[2:] if method.is_member else args[1:]
        params: list[tuple[str, str, bool]] = []
        for i, schema in enumerate(rest):
            ty, dynamic = self.render_param(schema)
            params.append((f"_{i}", ty, dynamic))

        if method.is_member:
            sig_parts = ["&self", *[f"{name}: {ty}" for name, ty, _ in params]]
            self_expr = "::tvm_ffi::object::object_core_as_any_view(self)"
        else:
            sig_parts = [f"{name}: {ty}" for name, ty, _ in params]
            self_expr = None
        packed = _packed_args_expr(params, self_expr)
        fvar = self._callable_local(params)
        getter = self._cached_getter_lines(fvar, ffi_name)
        header = f"pub fn {rust_name}({', '.join(sig_parts)}) -> ::tvm_ffi::Result<{ret}> {{"
        return [
            header,
            *_packed_call_lines(fvar, getter, packed, dynamic_result),
            "}",
        ]


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit an opaque Rust object wrapper with owning reflected accessors."""
    assert len(code.lines) >= 2
    type_key = obj_info.type_key
    assert isinstance(type_key, str)
    type_leaf = type_key.rsplit(".", 1)[-1]
    leaf = rust_identifier(type_leaf)
    obj_struct = rust_identifier(f"{type_leaf}Obj")
    parent_key = obj_info.parent_type_key
    is_root = parent_key in (None, "ffi.Object")
    if is_root:
        base_type = "Object"
    else:
        assert isinstance(parent_key, str)
        base_type = rust_identifier(f"{parent_key.rsplit('.', 1)[-1]}Obj")
    renderer = _ObjectRenderer(
        info=obj_info,
        leaf=leaf,
        obj_struct=obj_struct,
        base_type=base_type,
        is_root=is_root,
        imports=imports,
        ty_map=ty_map,
        mod_segments=tuple(type_key.split(".")[:-1]),
    )

    import_checkpoint = len(imports.items)
    try:
        body = renderer.body()
    except UnsupportedTypeError:
        del imports.items[import_checkpoint:]
        raise

    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity


# --- import section (`use` statements) --------------------------------------


def generate_rust_import_section(
    code: CodeBlock,
    imports: RustImports,
    opt: Options,
    defined_types: set[str],
) -> None:
    """Render the collected ``use`` statements into an ``import-section`` block.

    Imports are deduped and sorted. Local generated names were reserved before
    rendering, so a recorded item is always a real external/cross-module use.
    """
    assert len(code.lines) >= 2
    # `record` never admits bare types, so every `as_use_line()` is non-empty.
    use_lines = sorted({item.as_use_line() for item in imports.items})
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in use_lines],
        code.lines[-1],
    ]
    _ = (opt, defined_types)  # accepted for protocol parity


# --- whole-file scaffolding (`--init` mode) ---------------------------------


def generate_rust_api_file(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: InitConfig,
    is_root: bool,
    syntax: C.MarkerSyntax,
) -> str:
    """Scaffold a single Rust binding file (one file per module prefix)."""
    append = ""
    if not code_blocks:
        # This may be appended to an intermediate ``mod.rs`` that already has
        # child-module declarations, so use a regular comment rather than an
        # inner attribute or inner doc comment that would be illegal there.
        append += f"// FFI bindings for `{module_name}` (generated by tvm-ffi-stubgen).\n\n"
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


# --- module-tree stitching (auto-form `pub mod` declarations) ----------------

_RUST_MODULES_BEGIN = "// tvm-ffi-stubgen-modules(begin)"
_RUST_MODULES_END = "// tvm-ffi-stubgen-modules(end)"


def finalize_rust_module_tree(init_path: Path, prefixes: set[str]) -> None:  # noqa: PLR0912
    """Stitch the generated tree under ``init_path`` into a valid Rust module tree.

    Generated declarations live in a sorted marker block and are merged with
    declarations from earlier prefix-scoped invocations. Existing user-owned
    ``mod`` declarations outside that block are left untouched. The user still
    mounts ``init_path`` from the crate root.
    """
    children: dict[Path, set[str]] = {}
    for prefix in prefixes:
        segs = [s for s in prefix.split(".") if s]
        for i, seg in enumerate(segs):
            parent = init_path.joinpath(*segs[:i])
            children.setdefault(parent, set()).add(seg)

    begin = _RUST_MODULES_BEGIN
    end = _RUST_MODULES_END
    parents = set(children)
    if init_path.exists():
        for mod_rs in init_path.rglob("mod.rs"):
            if begin in mod_rs.read_text(encoding="utf-8"):
                parents.add(mod_rs.parent)

    for parent in sorted(parents):
        names = {rust_identifier(name) for name in children.get(parent, set())}
        parent.mkdir(parents=True, exist_ok=True)
        mod_rs = parent / "mod.rs"
        existing = mod_rs.read_text(encoding="utf-8") if mod_rs.exists() else ""
        lines = existing.splitlines()
        try:
            start = lines.index(begin)
            stop = lines.index(end, start + 1)
        except ValueError:
            start = stop = -1
        if start >= 0:
            managed = re.compile(r"^pub mod ((?:r#)?[A-Za-z_][A-Za-z0-9_]*);$")
            names.update(
                match.group(1) for line in lines[start + 1 : stop] if (match := managed.match(line))
            )

        outside = lines if start < 0 else lines[:start] + lines[stop + 1 :]
        generated_items: set[str] = set()
        object_marker = re.compile(rf"^\s*{re.escape(C.RUST_SYNTAX.begin)}\s+object/(\S+)\s*$")
        for line in outside:
            if match := object_marker.match(line):
                raw_leaf = match.group(1).rsplit(".", 1)[-1]
                generated_items.add(rust_identifier(raw_leaf))
                generated_items.add(rust_identifier(f"{raw_leaf}Obj"))
        declarations: list[str] = []
        for ident in sorted(names):
            if ident in generated_items:
                raise UnsupportedTypeError(
                    ident,
                    f"generated Rust item {ident!r} conflicts with child module {ident!r}",
                )
            pattern = re.compile(
                rf"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+{re.escape(ident)}\s*(?:;|\{{)"
            )
            if not any(pattern.match(line) for line in outside):
                declarations.append(f"pub mod {ident};")
        block = [begin, *declarations, end]
        if start >= 0:
            lines[start : stop + 1] = block
        elif names:
            if lines and lines[-1].strip():
                lines.append("")
            lines.extend(block)
        mod_rs.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
