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

from typing import TYPE_CHECKING, Callable

from .. import consts as C
from .consts import RUST_UNSUPPORTED_ORIGINS
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
    lines += ["#[repr(C)]", f"struct {obj_struct} {{", f"    base: {base_type},"]
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

    # (3) `<T>` ref struct.
    lines += [
        "#[repr(C)]",
        "#[derive(DeriveObjectRef, Clone)]",
        f"struct {leaf} {{",
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
    impl_lines = _impl_block(info, leaf, obj_struct, mutable, render_param, imports)
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
    render_param: Callable[[TypeSchema], str],
    imports: RustImports,
) -> list[str]:
    """Emit `impl <T> { new; methods }`; empty list when there's nothing to emit."""
    methods = [m for m in info.methods if m.schema.name.rsplit(".", 1)[-1] != "__ffi_init__"]
    if not info.has_init and not methods:
        return []

    _use(imports, "tvm_ffi::Result")
    _use(imports, "tvm_ffi::into_typed_fn")

    inner: list[str] = []
    if info.has_init:
        inner += _new_fn(info, leaf, obj_struct, render_param)
        if methods:
            inner.append("")
    for i, method in enumerate(methods):
        inner += _method_fn(method, leaf, obj_struct, mutable, render_param)
        if i != len(methods) - 1:
            inner.append("")

    return [f"impl {leaf} {{", *[f"    {line}" if line else "" for line in inner], "}", ""]


def _new_fn(
    info: ObjectInfo,
    leaf: str,
    obj_struct: str,
    render_param: Callable[[TypeSchema], str],
) -> list[str]:
    """Emit `fn new(<init_fields>) -> Result<Self>` calling reflected `__ffi_init__`."""
    params = [(f.name, render_param(f.schema)) for f in info.init_fields]
    sig = ", ".join(f"{_rust_ident(n)}: {t}" for n, t in params)
    fn_types = ", ".join(t for _, t in params)
    call_args = ", ".join(_rust_ident(n) for n, _ in params)
    return [
        f"fn new({sig}) -> Result<Self> {{",
        f'    let ctor = get_type_method({obj_struct}::TYPE_KEY, "__ffi_init__")?;',
        f"    let call = into_typed_fn!(ctor, Fn({fn_types}) -> Result<{leaf}>);",
        f"    call({call_args})",
        "}",
    ]


def _method_fn(
    method: FuncInfo,
    leaf: str,
    obj_struct: str,
    mutable: bool,
    render_param: Callable[[TypeSchema], str],
) -> list[str]:
    """Emit one reflected method (instance or static) on `impl <T>`."""
    ffi_name = method.schema.name.rsplit(".", 1)[-1]
    rust_name = _rust_ident(ffi_name)
    args = method.schema.args or ()
    ret = render_param(args[0]) if args else "Any"
    rest = list(args[1:])
    if method.is_member:
        rest = rest[1:]  # drop the leading `self` schema arg
    params = [(f"_{i}", render_param(p)) for i, p in enumerate(rest)]
    param_types = [t for _, t in params]

    # Receiver reflects class mutability; the `into_typed_fn!` self type is always a
    # shared borrow (`&T`) -- the FFI call borrows self as a view, and a `&mut self`
    # receiver reborrows to `&self` at the call site (matches the worked example).
    self_recv = "&mut self" if mutable else "&self"
    self_ty = f"&{leaf}"

    if method.is_member:
        sig_parts = [self_recv, *[f"{n}: {t}" for n, t in params]]
        fn_types = ", ".join([self_ty, *param_types])
        call_args = ", ".join(["self", *[n for n, _ in params]])
    else:
        sig_parts = [f"{n}: {t}" for n, t in params]
        fn_types = ", ".join(param_types)
        call_args = ", ".join(n for n, _ in params)

    return [
        f"fn {rust_name}({', '.join(sig_parts)}) -> Result<{ret}> {{",
        f'    let f = get_type_method({obj_struct}::TYPE_KEY, "{ffi_name}")?;',
        f"    let call = into_typed_fn!(f, Fn({fn_types}) -> Result<{ret}>);",
        f"    call({call_args})",
        "}",
    ]
