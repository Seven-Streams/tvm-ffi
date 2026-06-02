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

Currently implements the type renderer (:func:`render_rust_type`). It turns a
language-agnostic :class:`~tvm_ffi.core.TypeSchema` into a Rust type expression,
recording the ``use`` imports it needs via a ``ty_render`` callback. FFI types
the ``rust/tvm-ffi`` crate cannot represent raise :class:`UnsupportedTypeError`,
which the (future) object/function generators catch to warn-and-skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .. import consts as C
from .consts import RUST_NO_IMPORT_FULLPATH, RUST_UNSUPPORTED_ORIGINS
from .imports import RustImports, RustUse

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, ObjectInfo, Options


class UnsupportedTypeError(Exception):
    """Raised when an FFI type has no representation in the ``rust/tvm-ffi`` crate.

    Carries the offending FFI ``origin`` (e.g. ``"Map"``) so callers can produce
    a precise ``[Skipped]`` message before skipping the enclosing object/function.
    """

    def __init__(self, origin: str) -> None:
        """Record the unsupported ``origin`` and build a clear message."""
        super().__init__(f"Rust backend does not support FFI type {origin!r}")
        self.origin = origin


def build_ty_render(ty_map: dict[str, str], imports: RustImports) -> Callable[[str], str]:
    """Build a leaf-origin -> Rust-leaf-name mapper that records ``use`` imports.

    Given an FFI origin, it looks up the (fully-qualified) Rust path in ``ty_map``,
    records the ``use`` it needs, and returns the name to use in scope:

    * bare prelude/primitive types (``i64`` / ``bool`` / ``()`` / ``Option``) carry
      no ``::`` -> no ``use`` is recorded and the bare name is returned;
    * a qualified path is recorded once (repeat references to the *same* path reuse
      the binding rather than emitting a duplicate ``use``);
    * if a different path wants a leaf name already taken (e.g. ``a::Foo`` and
      ``b::Foo``), the later one is aliased -- ``use b::Foo as Foo2;`` -- and the
      alias is returned. This type-vs-type clash is the only ``use`` collision that
      arises in Rust (function/method names live in a separate namespace, and
      methods are scoped inside ``impl`` blocks, so they never shadow a ``use``).
    """
    # Seed from anything already recorded (e.g. import-object directives) so we
    # don't re-import or collide with pre-existing uses.
    binding_of: dict[str, str] = {u.full_name: u.name_in_scope for u in imports.items}
    used_names: set[str] = {u.name_in_scope for u in imports.items}

    def _run(origin: str) -> str:
        probe = RustUse(ty_map.get(origin, origin))
        if not probe.as_use_line():
            # bare prelude/primitive: no import, no collision tracking.
            return probe.leaf
        if probe.full_name in RUST_NO_IMPORT_FULLPATH:
            # rendered fully-qualified inline; no `use` (avoids shadowing a
            # prelude name -- e.g. `String` vs `std::string::String`).
            return probe.full_name
        full = probe.full_name
        if full in binding_of:
            return binding_of[full]  # same path already imported (maybe aliased)
        leaf = probe.leaf
        if leaf not in used_names:
            name, use = leaf, probe
        else:
            n = 2
            while f"{leaf}{n}" in used_names:
                n += 1
            name = f"{leaf}{n}"
            use = RustUse(full, alias=name)
        imports.items.append(use)
        used_names.add(name)
        binding_of[full] = name
        return name

    return _run


def render_rust_type(schema: TypeSchema, ty_render: Callable[[str], str]) -> str:
    """Render a :class:`TypeSchema` into a Rust type expression.

    ``ty_render`` maps a leaf origin name to its Rust leaf name and records the
    ``use`` import it needs (see :func:`build_ty_render`).

    Raises
    ------
    UnsupportedTypeError
        If ``schema`` (or any nested arg) uses an FFI origin the crate cannot
        represent (``Union`` / ``Map`` / ``Dict`` / ``List``).

    """
    origin = schema.origin
    args = schema.args or ()

    if origin in RUST_UNSUPPORTED_ORIGINS:
        raise UnsupportedTypeError(origin)

    if origin == "Optional":
        # post_init guarantees exactly one arg.
        inner_schema = next(iter(args), None)
        if inner_schema is None:
            raise ValueError("Optional type requires exactly one argument")
        inner = render_rust_type(inner_schema, ty_render)
        return f"{ty_render('Optional')}<{inner}>"

    if origin == "Array":
        # post_init guarantees (Any,) when no element type is given.
        elem = render_rust_type(args[0], ty_render) if args else ty_render("Any")
        return f"{ty_render('Array')}<{elem}>"

    if origin == "Callable":
        # The crate's Function is type-erased / concrete: no generic params.
        return ty_render("Callable")

    if origin == "tuple":
        if not args:
            return "()"
        inner = ", ".join(render_rust_type(a, ty_render) for a in args)
        return f"({inner})"

    # leaf / object type -> resolve via ty_render (records its `use`).
    return ty_render(origin)


# --- object generation (struct + ObjectCore + ref + Deref + impl) -----------

#: Rust strict keywords that, used as a field/method identifier, need a raw
#: identifier (``r#name``). A few (``self``/``Self``/``super``/``crate``) cannot
#: be raw identifiers; they are left as-is (extremely unlikely as FFI names).
_RUST_KEYWORDS = frozenset(
    """as break const continue crate dyn else enum extern false fn for if impl in let loop
    match mod move mut pub ref return static struct super trait true type unsafe use where while
    async await self Self""".split()
)
_RUST_RAW_FORBIDDEN = frozenset({"self", "Self", "super", "crate"})


def _rust_ident(name: str) -> str:
    """Make ``name`` a usable Rust identifier (raw-escape keywords)."""
    if name in _RUST_KEYWORDS and name not in _RUST_RAW_FORBIDDEN:
        return f"r#{name}"
    return name


def _use(imports: RustImports, path: str, alias: str | None = None) -> None:
    """Record a ``use`` (deduped) onto the collector."""
    item = RustUse(path, alias=alias)
    if item not in imports.items:
        imports.items.append(item)


def _class_is_mutable(info: ObjectInfo) -> tuple[bool, bool]:
    """Return (mutable, mixed) per Q1 from the fields' read-only flags.

    mutable = all fields writable; immutable = all read-only; mixed (some of
    each) -> not mutable, mixed=True so the caller can warn.
    """
    frozens = [bool(f.frozen) for f in info.fields]
    if not frozens:
        return False, False
    if all(not z for z in frozens):
        return True, False
    if all(frozens):
        return False, False
    return False, True


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit a Rust ``struct``/``impl`` binding for an ``object/<key>`` block.

    Mirrors the hand-written pattern in ``cpp_rust_test1/rust/src/main.rs``:
    ``<T>Obj`` (``#[repr(C)]`` with parent embedded as ``base``), ``ObjectCore``
    impl, ``<T>`` ref (``ObjectArc``), ``Deref``/``DerefMut`` (the latter only for
    mutable classes), and ``impl <T>`` with a ``new`` constructor + reflected
    methods. On an :class:`UnsupportedTypeError` the whole object is skipped with
    a warning (decision 1).
    """
    assert len(code.lines) >= 2
    info = obj_info
    type_key = info.type_key
    assert isinstance(type_key, str)
    leaf = type_key.rsplit(".", 1)[-1]
    obj_struct = f"{leaf}Obj"
    parent_key = info.parent_type_key
    is_root = parent_key in (None, "ffi.Object")
    if is_root:
        base_type = "Object"
    else:
        assert isinstance(parent_key, str)
        base_type = f"{parent_key.rsplit('.', 1)[-1]}Obj"
    mutable, mixed = _class_is_mutable(info)

    # Render into a local collector so a skip leaves `imports` untouched.
    local = RustImports(items=list(imports.items))
    ty_render = build_ty_render(ty_map, local)

    def render_field(schema: TypeSchema) -> str:
        return render_rust_type(schema, ty_render)

    def render_param(schema: TypeSchema) -> str:
        # Q5: a top-level Any in argument position is the non-owning AnyView.
        if schema.origin == "Any":
            _use(local, "tvm_ffi::AnyView")
            return "AnyView"
        return render_rust_type(schema, ty_render)

    try:
        body = _render_object_body(
            info, leaf, obj_struct, base_type, is_root, mutable, render_field, render_param, local
        )
    except UnsupportedTypeError as e:
        if mixed:
            pass  # mixed warning is informational; suppressed when also skipping
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

    imports.items[:] = local.items
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # indent/opt currently unused for Rust object layout


def _render_object_body(
    info: ObjectInfo,
    leaf: str,
    obj_struct: str,
    base_type: str,
    is_root: bool,
    mutable: bool,
    render_field: Callable[[TypeSchema], str],
    render_param: Callable[[TypeSchema], str],
    imports: RustImports,
) -> list[str]:
    """Build the Rust source lines for one object (raises on unsupported types)."""
    # Boilerplate `use`s the generated items rely on.
    _use(imports, "std::ops::Deref")
    _use(imports, "tvm_ffi::object::ObjectArc")
    _use(imports, "tvm_ffi::object::ObjectCore")
    _use(imports, "tvm_ffi::derive::ObjectRef", alias="DeriveObjectRef")
    _use(imports, "tvm_ffi::tvm_ffi_sys::TVMFFIObject")
    if is_root:
        _use(imports, "tvm_ffi::object::Object")
    if mutable:
        _use(imports, "std::ops::DerefMut")

    lines: list[str] = []

    # (1) `<T>Obj` data struct (#[repr(C)], parent embedded as `base`).
    # The struct is `pub` (usable across modules); `base` stays private (an
    # implementation detail reached only via the generated Deref impls).
    lines += ["#[repr(C)]", f"pub struct {obj_struct} {{", f"    base: {base_type},"]
    for field in info.fields:
        lines.append(f"    pub {_rust_ident(field.name)}: {render_field(field)},")
    lines += ["}", ""]

    # (2) ObjectCore impl.
    lines += [
        f"unsafe impl ObjectCore for {obj_struct} {{",
        f'    const TYPE_KEY: &\'static str = "{info.type_key}";',
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

    # (3) `<T>` ref struct (`pub`; `data` stays private).
    lines += [
        "#[repr(C)]",
        "#[derive(DeriveObjectRef, Clone)]",
        f"pub struct {leaf} {{",
        f"    data: ObjectArc<{obj_struct}>,",
        "}",
        "",
    ]

    # (4) Deref (+ DerefMut if mutable) for the ref.
    lines += _deref_impl(leaf, obj_struct, "data", mutable)

    # (5) Deref (+ DerefMut if mutable) for a derived Obj -> embedded base.
    if not is_root:
        lines += _deref_impl(obj_struct, base_type, "base", mutable)

    # (6) impl with `new` + methods.
    impl_lines = _impl_block(info, leaf, obj_struct, mutable, render_field, render_param, imports)
    if impl_lines:
        lines += impl_lines

    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _deref_impl(ref: str, target: str, field: str, mutable: bool) -> list[str]:
    """Emit `Deref` (+ `DerefMut` if mutable) for `ref` -> `target` via `self.<field>`."""
    out = [
        f"impl Deref for {ref} {{",
        f"    type Target = {target};",
        f"    fn deref(&self) -> &{target} {{",
        f"        &self.{field}",
        "    }",
        "}",
        "",
    ]
    if mutable:
        out += [
            f"impl DerefMut for {ref} {{",
            f"    fn deref_mut(&mut self) -> &mut {target} {{",
            f"        &mut self.{field}",
            "    }",
            "}",
            "",
        ]
    return out


def _impl_block(
    info: ObjectInfo,
    leaf: str,
    obj_struct: str,
    mutable: bool,
    render_ret: Callable[[TypeSchema], str],
    render_param: Callable[[TypeSchema], str],
    imports: RustImports,
) -> list[str]:
    """Emit `impl <T> { new; methods }`; empty list when there's nothing to emit."""
    init_method = next(
        (m for m in info.methods if m.schema.name.rsplit(".", 1)[-1] == "__ffi_init__"), None
    )
    methods = [m for m in info.methods if m.schema.name.rsplit(".", 1)[-1] != "__ffi_init__"]
    if not info.has_init and not methods:
        return []

    _use(imports, "tvm_ffi::Result")

    inner: list[str] = []
    if info.has_init:
        inner += _new_fn(info, obj_struct, render_param, init_method, imports)
        if methods:
            inner.append("")
    for i, method in enumerate(methods):
        inner += _method_fn(method, obj_struct, mutable, render_ret, render_param, imports)
        if i != len(methods) - 1:
            inner.append("")

    return [f"impl {leaf} {{", *[f"    {line}" if line else "" for line in inner], "}", ""]


def _packed_args_expr(params: list[tuple[str, str]], is_member: bool) -> str:
    """Build the ``&[AnyView]`` element list for a packed call.

    Each non-``AnyView`` argument is borrowed as a view via ``AnyView::from(&x)``
    (works for any ``AnyCompatible`` value -- scalars, ``String``, ``Array``,
    ``ObjectRef``); an ``AnyView`` argument is already a view and is passed
    through unchanged. A member call prepends the receiver as ``&*self``.
    """
    parts = ["AnyView::from(&*self)"] if is_member else []
    for name, ty in params:
        parts.append(name if ty == "AnyView" else f"AnyView::from(&{name})")
    return ", ".join(parts)


def _packed_call_lines(fvar: str, getter: str, packed: str, ret: str) -> list[str]:
    """Build the two body lines for a reflected call via ``Function::call_packed``.

    Every method/constructor uses one uniform calling convention: pack the
    arguments into ``&[AnyView]`` and call. ``call_packed`` returns ``Result<Any>``,
    so a top-level ``Any`` return is forwarded as-is; any other return type is
    converted with ``try_into`` (whose error is ``tvm_ffi::Error``). This is the
    only convention the FFI exposes that can carry ``AnyView`` arguments and an
    ``Any`` return, so it is used everywhere rather than special-casing them.
    """
    if ret == "Any":
        return [getter, f"    {fvar}.call_packed(&[{packed}])"]
    return [getter, f"    Ok({fvar}.call_packed(&[{packed}])?.try_into()?)"]


def _new_fn(
    info: ObjectInfo,
    obj_struct: str,
    render_param: Callable[[TypeSchema], str],
    init_method: FuncInfo | None,
    imports: RustImports,
) -> list[str]:
    """Emit `fn new(...) -> Result<Self>` calling reflected `__ffi_init__`.

    Parameter order/types come from the ``__ffi_init__`` method schema when
    available (``args[0]`` is the constructed object / return; ``args[1:]`` are
    the constructor params) -- this is the authoritative order. The schema has no
    parameter names, so positional names ``_0, _1, ...`` are used. Only when the
    init is a column-only auto-init (no ``__ffi_init__`` method) do we fall back
    to ``init_fields`` (named), which may not match the true constructor order.
    """
    if init_method is not None:
        arg_schemas = list(init_method.schema.args[1:]) if init_method.schema.args else []
        params = [(f"_{i}", render_param(s)) for i, s in enumerate(arg_schemas)]
    else:
        params = [(_rust_ident(f.name), render_param(f.schema)) for f in info.init_fields]
    sig = ", ".join(f"{n}: {t}" for n, t in params)
    if params:
        _use(imports, "tvm_ffi::AnyView")
    packed = _packed_args_expr(params, is_member=False)
    getter = f'    let ctor = get_type_method({obj_struct}::TYPE_KEY, "__ffi_init__")?;'
    # The constructor return is the object itself, converted from `Any` via `try_into`.
    return [
        f"pub fn new({sig}) -> Result<Self> {{",
        *_packed_call_lines("ctor", getter, packed, "Self"),
        "}",
    ]


def _method_fn(
    method: FuncInfo,
    obj_struct: str,
    mutable: bool,
    render_ret: Callable[[TypeSchema], str],
    render_param: Callable[[TypeSchema], str],
    imports: RustImports,
) -> list[str]:
    """Emit one reflected method (instance or static) on `impl <T>`.

    The return type (``args[0]``) is rendered with ``render_ret`` (owning
    semantics): per decision Q5 a top-level ``Any`` return stays ``Any``, not the
    non-owning ``AnyView`` -- a borrow has no lifetime source coming back out of
    an FFI call. Parameters use ``render_param`` (top-level ``Any -> AnyView``).
    All calls go through the uniform ``call_packed`` convention.
    """
    ffi_name = method.schema.name.rsplit(".", 1)[-1]
    rust_name = _rust_ident(ffi_name)
    args = method.schema.args or ()
    ret = render_ret(args[0]) if args else "Any"
    rest = list(args[1:])
    if method.is_member:
        rest = rest[1:]  # drop the leading `self` schema arg
    params = [(f"_{i}", render_param(p)) for i, p in enumerate(rest)]

    # Receiver reflects class mutability; the FFI call only borrows self as a view,
    # so a `&mut self` receiver reborrows to `&*self` when packed.
    self_recv = "&mut self" if mutable else "&self"
    if method.is_member:
        sig_parts = [self_recv, *[f"{n}: {t}" for n, t in params]]
    else:
        sig_parts = [f"{n}: {t}" for n, t in params]
    if method.is_member or params:
        _use(imports, "tvm_ffi::AnyView")
    packed = _packed_args_expr(params, method.is_member)
    getter = f'    let f = get_type_method({obj_struct}::TYPE_KEY, "{ffi_name}")?;'
    header = f"pub fn {rust_name}({', '.join(sig_parts)}) -> Result<{ret}> {{"
    return [header, *_packed_call_lines("f", getter, packed, ret), "}"]


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
    seen: dict[str, RustUse] = {}
    for item in imports.items:
        if item.full_name in defined_types:
            continue
        seen.setdefault(item.as_use_line(), item)
    use_lines = sorted(seen)
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in use_lines],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust needs no indent/TYPE_CHECKING handling


# --- whole-file scaffolding (`--init` mode) ---------------------------------

#: Shared per-file helper functions (decision Q4). Written fully-qualified with
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
    (filled by ``stage_3`` with the shared support functions), an
    ``import-section`` marker, and an ``object/<type_key>`` marker per registered
    type are added if missing. Putting the helpers in a marker means they are
    (re)generated on every run -- even into a pre-existing file -- rather than
    only when the file is brand new. No ``global``/``__all__``/``export`` blocks
    are emitted (decision 5 / step 7b).
    """
    append = ""
    if not code_blocks:
        append += "#![allow(dead_code, unused_imports)]\n"
        append += f"//! FFI bindings for `{module_name}` (generated by tvm-ffi-stubgen).\n\n"
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
