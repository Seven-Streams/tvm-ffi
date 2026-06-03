<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Plan: `tvm-ffi-stubgen` Rust 后端实现计划

本文件是 `proposal.md` 的落地执行版，记录**已完成的拆分**与**剩余的 Rust 实现**，
粒度细化到"每步改哪些文件、怎么测、验收标准"，便于增量完善与测试。

- 关联：`proposal.md`（设计提案）、`tests/python/test_stubgen.py`（现有测试）
- 已锁定决策（见 proposal §6）：
  1. **不支持的类型 → 警告并跳过（不硬错）**：Rust 表达不了的类型（`Union`、
     `Map`、`Dict`、`List`）在 `render_type` 处以**哨兵异常**冒泡；由 object /
     global-func 级生成函数捕获，打一条 `[Skipped]` warning 并**跳过该 object /
     函数**，其余照常生成。不在文件级中断。
  2. **import 拆分**：每个后端拥有自己的 import 表示；管线只见不透明收集器。
  3. **utils.py 方案 a**：`utils` 仅存语言无关数据，渲染逻辑全在各后端 `codegen`。
  4. **Rust 文件布局**：暂缓，待定（见 P3-步骤 9）。
  5. **生成范围**：
     - **全局函数：不生成**。Rust 运行时已可经 `Function::get_global(name)` +
       `call_*` 动态调用 C++ 注册的全局函数，无需静态 stub。
     - **object：生成 `<T>Obj`（`#[repr(C)]` 布局）+ `<T>` ref（`ObjectArc`）+
       `Deref/DerefMut` + `impl <T>`（方法**真实调用**）**。方法调用机制由
       `cpp_rust_test/` 示例确立（不再是 `todo!()` 占位）：方法存于类型反射方法表
       `TVMFFIMethodInfo.method`（首参恒为 `self`，**不在**全局注册表），生成代码经
       `get_type_method(TYPE_KEY, name)` 取出 `Function`，再 `into_typed_fn!` 按类型签名调用。
       详见步骤 5 模板。

---

## 一、proposal §4 各项状态对照

| proposal 项 | 内容 | 状态 |
| --- | --- | --- |
| 基础设施 | `file_utils`/`lib_state`/marker 语法 语言无关 | ✅ 早已就绪 |
| §4③ | utils.py 方案 a：数据与渲染分离 | ✅ 已完成 |
| §4④ | import 拆分：后端私有 + 不透明收集器协议 | ✅ 协议已就绪，Python 侧已实现 |
| 结构拆分 | Python 内容 → `python_backend/` | ✅ 已完成 |
| 结构拆分 | Rust 骨架 → `rust_backend/` | ✅ 已完成（仅骨架） |
| 测试适配 | `test_stubgen.py` 迁移到新结构 | ✅ 26 passed |
| §4⑤ | `rust_backend` 常量表（类型/路径/builtin） | ✅ 已完成（步骤 1） |
| §4④(Rust) | Rust import 表示（`use` 收集器） | ✅ 已完成（步骤 2） |
| §4② | `RustBackend.render_type` 类型遍历 + 不支持类型抛异常 | ✅ 已完成（步骤 3） |
| §4②附 | Rust `TyRenderer`（去重 + 同名 leaf 起别名，方案 A） | ✅ 已完成（步骤 4） |
| §4① | `rust_backend/codegen.py`：object 生成（`generate_rust_object`：struct+ObjectCore+ref+Deref+impl/new/methods） | ✅ 已完成（步骤 5） |
| §4①附 | `generate_rust_import_section`（`use` 段 + `defined_types` 过滤） | ✅ 已完成（步骤 7a） |
| — | `generate_rust_all` / `generate_rust_export`（再导出） | ⏸ 推迟（步骤 7b，依赖布局步骤 9） |
| 装配 | `RustBackend` 接线（normal 模式端到端） | ✅ 已完成（步骤 8） |
| §4⑥ | `cli.py` 文件名后端化（`api_filename`/`init_filename`） | ✅ 已完成（步骤 10） |
| 布局 | Rust 布局 = 单文件/前缀、全 `pub`*、helper 每文件、无 all/export | ✅ 已定（步骤 9，全 A） |
| 脚手架 | `generate_rust_api_file`（`#![allow]`+helper+import/object marker）+ `--init` 端到端 | ✅ 已完成（步骤 11，cargo run 验证） |

---

## 二、已完成结构（基线，勿回退）

```text
stub/                       # 语言无关
  consts.py                 # MarkerSyntax / SYNTAX_BY_EXT / STUB_BLOCK_KINDS / FN_NAME_MAP / BUILTIN_TYPE_KEYS / 颜色
  utils.py                  # 纯数据: Options/InitConfig/NamedTypeSchema/FuncInfo/ObjectInfo/InitFieldInfo + from_type_info
  file_utils.py             # CodeBlock/FileInfo 解析回写
  lib_state.py              # 读 FFI 注册表 → ObjectInfo/FuncInfo
  backend.py                # Backend 协议 + get_backend + _BACKENDS 装配
  cli.py                    # 编排 stage1/2/3（import 流程后端无关）
  python_backend/
    __init__.py             # 导出 PythonBackend
    backend.py              # PythonBackend
    codegen.py              # generate_python_* + render_func_signature/render_object_*
    imports.py              # ImportItem + PythonImports
    consts.py               # TY_MAP_DEFAULTS + MOD_MAP
  rust_backend/
    __init__.py             # 导出 RustBackend
    backend.py              # RustBackend（全部方法 raise NotImplementedError）
```

**Backend 协议**（`backend.py`）当前方法集（Rust 需逐一实现）：
`default_ty_map` / `render_type` / `new_imports` / `add_imported_object` /
`canonical_type_name` / `extra_export_names` / `generate_global_funcs_block` /
`generate_object_block` / `generate_import_section_block(.., defined_types)` /
`generate_all_block` / `generate_export_block` / `generate_api_file` /
`generate_init_file`。

**验证现状**：`get_backend("python"|"rust")` 均满足 `runtime_checkable` 协议；
全树 `--dry-run` 无异常；`test_stubgen.py` 26 passed。

---

## 三、剩余实现（按优先级 + 可独立测试的增量切分）

> 原则：每步只动少量文件，结束时有明确验收 + 单元测试。Rust 数据来自共享
> `lib_state`/`utils`，因此每个渲染函数都能脱离 CLI 单测。

### P1 — 数据与基础设施（无渲染，先把"原材料"备齐）

#### 步骤 1：`rust_backend/consts.py`

- 新建，仿 `python_backend/consts.py`。
- **依据**：映射必须以 `rust/tvm-ffi` crate **实际支持**的类型为准（来自
  `type_traits.rs` 的 `AnyCompatible` impl，以及各类型的 `#[type_key = "..."]`），
  **不能凭空假设**。Rust crate 用自己的 `Array<T>`，且**没有 `Map`/`Dict`/`List`/
  `HashMap`/`Vec` 作为 FFI 类型**。
- 内容：
  - `RUST_TY_MAP_DEFAULTS`：FFI origin → Rust 类型，**值带完整 `tvm_ffi::` 路径**
    （仿 Python map：值=完整路径，便于 `TyRenderer` 按 `::` 拆出 leaf + 推导 `use`）。
    primitives/`()`/`Option` 无 `::`、无需 import。按下表（已核对 crate）：

    | FFI origin | Rust 值 | 备注 |
    | --- | --- | --- |
    | `int` | `i64` | 所有整型→FFI `int`，默认回填 `i64`；无 import |
    | `float` | `f64` | `f32/f64`→FFI `float`，默认 `f64`；无 import |
    | `bool` | `bool` | 无 import |
    | `None` | `()` | crate 用 `()` 表示 None/void |
    | `Optional` | `Option` | std prelude，`Option<T>`，无 import |
    | `Any` | `tvm_ffi::Any` | 另有 `AnyView`（非拥有），按位置可能用引用 |
    | `Callable` | `tvm_ffi::Function` | |
    | `Array` | `tvm_ffi::Array` | **`Array<T>`，不是 `Vec`** |
    | `Object` | `tvm_ffi::Object` | |
    | `Tensor` | `tvm_ffi::Tensor` | |
    | `Shape` | `tvm_ffi::Shape` | |
    | `Device` | `tvm_ffi::DLDevice` | dlpack `DLDevice`（crate 根再导出）；FFI type_str = `"Device"` |
    | `dtype` / `DataType` | `tvm_ffi::DLDataType` | dlpack `DLDataType` + `DLDataTypeExt` |
    | `str` / `ffi.String` | `tvm_ffi::String` | 避免撞 `std::string::String` |
    | `ffi.Bytes` | `tvm_ffi::Bytes` | |
    | `ffi.Module` | `tvm_ffi::Module` | |
    | `ffi.Error` | `tvm_ffi::Error` | |

  - `RUST_UNSUPPORTED_ORIGINS`：**crate 不支持**的 origin 集合 —— `Map`、`Dict`、
    `List`、`Union`。`render_type` 命中即抛**哨兵异常 `UnsupportedTypeError`**
    （定义建议放 `rust_backend/codegen.py` 或 `rust_backend/__init__.py`），由上层
    捕获后 warn+skip（决策 1）。**不要**强行映射成 `HashMap`/`Vec`。
  - `RUST_MOD_MAP`：crate 路径前缀映射（如 `ffi → tvm_ffi`），供 `use` 路径构造。
    具体路径以 `rust/tvm-ffi/src/lib.rs` 的 `pub use` 再导出为准
    （`Array`/`Shape`/`Tensor`/`Function`/`Object`/`String`/`Bytes`/`Module`/`Error`
    等都从 crate 根 `tvm_ffi::` 再导出）。
- **测试**：`test_rust_consts.py`，断言上表关键 origin 的值；断言
  `Map`/`Dict`/`List`/`Union` 在 unsupported 集合中。
- **验收**：表与 crate 一致；不支持类型集合明确；`use` 路径与 `lib.rs` 再导出对齐。

#### 步骤 2：`rust_backend/imports.py`

- 新建，仿 `python_backend/imports.py`。
- 内容：
  - `RustUse`（数据类）：表示一条 `use a::b::c [as d];`。字段建议
    `path: str`、`alias: str | None`，属性 `full_name` / 渲染用 `as_use_line()`。
  - `RustImports`（收集器）：`items: list[RustUse]` + 必要标志位
    （对应 Python 的 `has_lib_load`，Rust 是否需要待定，先留空/False）。
- **测试**：`test_rust_imports.py`：构造若干 `RustUse`，断言路径解析、别名、去重。
- **验收**：`RustImports()` 可创建；`canonical`/`full_name` 语义自洽。

### P2 — 渲染核心

#### 步骤 3：`RustBackend.render_type`（类型遍历器）

- 位置：`rust_backend/`（建议放 `codegen.py` 作自由函数，再由 backend 调用，
  与 Python 侧 `render_*` 自由函数风格一致）。
- 递归处理 `schema.origin`/`schema.args`（映射以步骤 1 的 crate 实测表为准）：
  - `Optional[T] → Option<{T}>`
  - `Array[T] → Array<{T}>`（crate 自有类型，**非 Vec**）
  - `Callable → Function`
  - `tuple → ({T1}, {T2}, ...)`，空 tuple 待定（`()` / None）
  - **`Union` / `Map` / `Dict` / `List` → 抛 `UnsupportedTypeError`**（哨兵异常，
    crate 不支持；决策 1 + 步骤 1 的 `RUST_UNSUPPORTED_ORIGINS`）。注意：**嵌套**也算，
    如 `Optional[Map[..]]` 递归到内层时一样冒泡。
  - leaf/object → `ty_render(origin)`（记录 import，复用步骤 2 收集器）
- **测试**：`test_rust_render_type.py`，对每条规则用 `TypeSchema(...)` 直接断言输出字符串；
  对 `Union`/`Map`/`Dict`/`List`（含嵌套，如 `Array[Map[...]]`）各断言抛
  `UnsupportedTypeError`。
- **验收**：覆盖所有分支；不支持类型（含嵌套）都抛哨兵异常。

#### 步骤 4：Rust 的 `TyRenderer` + import 记录

- 仿 `python_backend.codegen._type_suffix_and_record`：给定 origin，返回 Rust 名
  并把所需 `use` 记入 `RustImports`。
- **测试**：调用后断言收集器里出现期望的 `RustUse`（含别名去冲突逻辑，如有）。
- **验收**：render_type 的 leaf 分支经由它记录 import。

#### 步骤 5：`generate_rust_object`（`<T>Obj` + `<T>` ref + Deref + `impl`）

- 位置：`rust_backend/codegen.py`。写入 `code.lines`（mutate，遵循
  `code.lines[0] + 缩进体 + code.lines[-1]` 模式）。
- **依据**：`cpp_rust_test/`（`cpp/expr_lib.cc` ↔ `rust/src/main.rs`）确立的真实模板。
  对每个 object（ref 名 `T`=type_key 末段，数据结构 `<T>Obj`，type_key `K`，父类型键 `P`）生成：

  **(1) `<T>Obj` 数据结构**（`#[repr(C)]`，镜像 C++ 内存布局）
  - 首字段是父类嵌入：父为根 `ffi.Object` → `object: Object`；否则 → `base: <P 末段>Obj`。
  - 之后是该类型**自有字段**（`ObjectInfo.fields`，不含继承字段——继承靠嵌入父 Obj）：
    `pub <name>: <render_type(field)>`。**对象类型字段用 ref 类型**（如 `Expr` 而非 `ExprObj`）。

  **(2) `unsafe impl ObjectCore for <T>Obj`**
  - `const TYPE_KEY: &'static str = "K";`
  - `fn type_index() -> i32 { lookup_type_index(Self::TYPE_KEY) }`
  - `unsafe fn object_header_mut(...)`：根 → `Object::object_header_mut(&mut this.object)`；
    派生 → `<P>Obj::object_header_mut(&mut this.base)`。

  **(3) `<T>` ref**：`#[repr(C)] #[derive(ObjectRef, Clone)] struct <T> { data: ObjectArc<<T>Obj> }`。

  **(0) 类的可变性（决策 Q1，由字段注册模式决定）**
  - 每个字段经反射带 `frozen` 标志（`frozen=(flags & Writable)==0`：`def_ro→frozen`，`def_rw→可写`）。
  - 全部字段可写 → 类为 **mutable**；全部字段只读 → 类为 **immutable**；
    **既有只读又有可写 → 打 warning，按 immutable 处理**。无字段的类默认 immutable。
  - mutable 类 → 生成 `DerefMut`，实例方法用 `&mut self`；immutable 类 → **不生成 `DerefMut`**，
    实例方法用 `&self`。（字段始终 `pub`；可变与否由是否有 `DerefMut` 把关。）
  - 实现备注：`ObjectInfo` 需新增携带每字段 `frozen`（当前未捕获），来源 `TypeField.frozen`。

  **(4) `Deref`（+ mutable 类才有 `DerefMut`）**：ref `<T>` → `<T>Obj`（`&self.data`）；
    派生 `<T>Obj` → 父 Obj（`&self.base`）。`DerefMut` 仅在类为 mutable 时生成。

  **(5) `impl <T>` 方法**（`ObjectInfo.methods`，**真实调用**，非占位）：
  - **实例方法**（`is_member=True`）——self 接收器按类可变性（见 (0)）：

    ```rust
    fn <name>(&self /* 或 &mut self */, _0: P0, ...) -> Result<<Ret>> {
        let f = get_type_method(<T>Obj::TYPE_KEY, "<ffi 方法名>")?;
        let call = into_typed_fn!(f, Fn(&<T> /* 或 &mut <T> */, P0, ...) -> Result<<Ret>>);
        call(self, _0, ...)
    }
    ```

  - **静态方法**（`is_member=False`，示例 `Expr::test`）——关联函数，无 self：

    ```rust
    fn <name>(_0: P0, ...) -> Result<<Ret>> {
        let f = get_type_method(<T>Obj::TYPE_KEY, "<ffi 方法名>")?;
        let call = into_typed_fn!(f, Fn(P0, ...) -> Result<<Ret>>);
        call(_0, ...)
    }
    ```

  - 方法 schema 是 `Callable`：`args[0]`=返回类型，其后为参数；实例方法的 `args[1]`=self（渲染跳过，
    与 Python `gen` 一致），静态方法无 self。返回恒包 `Result<…>`，void → `Result<()>`。

  **(5b) `new` 构造函数（决策 Q3，仅当 `has_init`）**：`fn new(<init_fields>) -> Result<<T>>`，
    参数取自 `ObjectInfo.init_fields`（按序展开为位置参数；Rust 无 kwargs/默认值）。经反射
    `__ffi_init__` 构造（自包含，**无需**全局工厂命名约定）：

    ```rust
    fn new(value: i64) -> Result<Self> {
        let ctor = get_type_method(<T>Obj::TYPE_KEY, "__ffi_init__")?;
        let call = into_typed_fn!(ctor, Fn(<init_params>) -> Result<<T>>);
        call(<init_params>)
    }
    ```

    `has_init=False` → 不生成 `new`。方法循环里需**跳过 `__ffi_init__`**（它变成 `new`，不作普通方法）。

  **(6) 每文件共享辅助函数**（emit 一次/文件，决策 Q4）：
    `lookup_type_index(type_key) -> i32`（`TVMFFITypeKeyToIndex` + per-key `OnceLock` 缓存，
    match 臂覆盖**本文件**定义的全部 type key）和 `get_type_method(type_key, name) -> Result<Function>`
    （遍历 `TVMFFIGetTypeInfo(idx).methods` 按名取 `Function`，原样照搬示例）。
- **额外 `use`**：除字段/返回类型外，生成物还需要机制类型的 `use`：
  `tvm_ffi::{Result, Function, AnyView, into_typed_fn}`、`tvm_ffi::object::{Object, ObjectArc, ObjectCore}`、
  `tvm_ffi::derive::ObjectRef`、以及 helper 用到的 `tvm_ffi::tvm_ffi_sys::{TVMFFIGetTypeInfo,
  TVMFFITypeKeyToIndex, TVMFFIByteArray}`。这些是**固定样板 use**，可在文件头一次性产出
  （不走 render_type 的 ty_render 路径）。
- **warn+skip（决策 1）**：整个 object 的渲染**包一层** `try/except UnsupportedTypeError`——
  任一字段/方法的类型命中不支持，就打 `[Skipped] object <type_key>: 不支持的类型 <...>`
  （黄色，仿 `lib_state` 风格），marker 块**留空**（仅首尾两行），不抛到文件级。
- **测试**：`test_object`（含字段 + 方法的 object → 断言生成 `<T>Obj`/`ObjectCore`/ref/Deref/impl 各段，
  方法体含 `get_type_method` + `into_typed_fn!`）；`test_object_derived`（父非 root → `base: <P>Obj` 嵌入）；
  `test_object_skipped`（含 `Map` 字段 → 块留空 + warning）。
- **验收**：能为 `cpp_rust_test` 的 `Expr`/`Add` 生成与 `rust/src/main.rs` **同构**的绑定
  （命名/布局/方法分发一致，可编译可跑）；含不支持类型的 object 被整体跳过。

#### 步骤 6：全局函数块 = **no-op**（决策 5）✅ 已完成

- `global/<prefix>` 块在 Rust 后端**不生成任何内容**——Rust 运行时经
  `Function::get_global` 动态调用，无需 stub。
- `RustBackend.generate_global_funcs_block` 实现为**空操作**（marker 块保持原样/留空），
  `--init` 脚手架（步骤 11）也**不产出** `global/` marker。
- **测试**：`test_rust_codegen.py::test_global_funcs_noop`，断言 `global/` 块经过后内容不变。
- **验收**：含 `global/` 块的 `.rs` 文件处理后该块无新增内容、不报错。

#### 步骤 7a：`generate_rust_import_section`（`use` 段）——现做

- 把 step 5 攒在 `RustImports` 里的 `use` 真正落地到 `import-section` 块。
- **按 `defined_types` 过滤本地定义**：`RustUse.full_name` 若在 `defined_types`（由
  `canonical_type_name` 产出，step 8）里则跳过——清掉 step 5 留的伪 import
  （如 `Add` 字段 `a: Expr` 记的 `use cpp_rust_test::Expr;`，因 `Expr` 本地定义而丢弃）。
- 去重（`RustUse` 可哈希）+ 排序后，一条一行写入块（Rust 无 `TYPE_CHECKING` 概念）。
- **测试**：渲染若干 `use` + 断言过滤掉 `defined_types` 中的项 + 去重 + 排序稳定。
- **验收**：`use` 段正确落地，本地类型不出现在 `use` 里。

#### 步骤 7b：`generate_rust_all` / `generate_rust_export`（再导出）——**推迟到步骤 9 后**

- `all`（Python `__all__` 等价物）与 `export`（子模块再导出）本质是**模块对外再导出结构**，
  取决于**尚未拍板的 Rust 文件布局（步骤 9）**：单文件 vs `mod.rs`+多文件、生成类型是否 `pub`、
  再导出如何组织。且这两种 marker 块由脚手架（步骤 11）放入，布局没定时连"Rust 文件里有没有
  这两种块"都不确定。
- **决定**：布局未定前不实现 ②③；待步骤 9 后再做（或那时确认 Rust 不需要 `__all__`/`export`）。

#### 步骤 8：装配 `RustBackend`（去掉 NotImplementedError）✅ 已完成

- 在 `rust_backend/backend.py` 把各方法委托到上面 `codegen` 自由函数：
  - `default_ty_map → RUST_TY_MAP_DEFAULTS.copy()`
  - `new_imports → RustImports()`
  - `add_imported_object` / `canonical_type_name` / `extra_export_names`
  - 各 `generate_*_block`
- **测试**：`isinstance(get_backend("rust"), Backend)` 仍 True；
  端到端：构造一个含 marker 的 `.rs` 测试文件，跑 `_stage_3`，断言回写内容。
- **验收**：`.rs` 文件的 in-place 生成（normal mode）端到端可用。

### P3 — `--init` 模式与 CLI（依赖布局决策）

#### 步骤 9：确定 Rust 文件布局（**需用户拍板**）

- 决策点：`mod.rs` + 每模块 `.rs`？还是 `lib.rs` 顶层？与 `rust/` 现有 crate 如何对齐。
- 产出：目标文件名规则（对应 Python 的 `_ffi_api.py` / `__init__.py`）。

#### 步骤 10：`cli.py` 文件名后端化（proposal §4⑥）

- 现状：`_stage_2` 硬编码 `_ffi_api.py` / `__init__.py` / 子模块名 `"_ffi_api"`。
- 改为由 backend 提供（在 `Backend` 协议加 `api_filename()` / `init_filename()` 等）。
- gate 掉 Python 专属的 `LIB`/`load_lib_module` 特判对 Rust 的影响
  （已由 `extra_export_names` 抽象，Rust 返回空集即可）。
- **测试**：Python 路径回归（`test_stubgen.py` 仍 26 passed）；Rust `--init` 冒烟。
- **验收**：Python 行为不变；Rust init 生成正确文件名。

#### 步骤 11：`generate_rust_api_file` / `generate_rust_init_file`（脚手架）

- 仿 `generate_python_ffi_api` / `generate_python_init`，产出 Rust 整文件脚手架 + marker。
- **仅为 object 产出 marker**（决策 5）：脚手架里**不要**产 `global/` marker，只产
  `object/<type_key>` 与（如需要）import/export marker。
- **测试**：`test_rust_codegen.py` 脚手架断言（含"无 global marker"）；`_stage_2` 端到端 init 一个 Rust 模块。
- **验收**：`--target rust --init-*` 能从零生成可编译骨架（编译验证可后置）；产物不含 global 函数块。

---

## 四、测试总览（建议新增）

| 测试文件 | 覆盖步骤 |
| --- | --- |
| `test_rust_consts.py` | 1 |
| `test_rust_imports.py` | 2 |
| `test_rust_render_type.py` | 3, 4（不支持类型抛 `UnsupportedTypeError`，含嵌套） |
| `test_rust_codegen.py` | 5, 6, 7, 11（含 object 整体跳过、global 块 no-op） |
| `test_stubgen.py`（扩展） | 8, 10 端到端 + Python 回归 |

回归底线：每步结束 `uv run pytest tests/python/test_stubgen.py` 必须保持通过；
Python 全树 `uv run tvm-ffi-stubgen --dry-run python/tvm_ffi` 无异常。

---

## 五、未决问题（实现前需明确）

已由 crate 核对确定（见步骤 1 表）：标量 `int→i64`/`float→f64`/`bool→bool`、
`None→()`、`String`/`Bytes`、Object 等类型名均以 `rust/tvm-ffi` 为准；
`Map`/`Dict`/`List` 确认不支持、需报错。**方法调用机制已由 `cpp_rust_test/` 示例确定**
（`get_type_method` + `into_typed_fn!`，见步骤 5）。

**已全部拍板**（用户确认）：

- **Q1 可变性**：由字段 `frozen` 决定——全可写→mutable（`DerefMut` + `&mut self`）；全只读→immutable
  （仅 `Deref` + `&self`）；混合→warn 后按 immutable。
- **Q2 static 方法**：关联函数、无 self（示例 `Expr::test`）。
- **Q3 `new`**：经反射 `__ffi_init__` 构造（cpp_rust_test1 已演示），参数取 `init_fields`，
  仅当 `has_init` 才生成；方法循环跳过 `__ffi_init__`。
- **Q4 helper**：`lookup_type_index` / `get_type_method` 每文件各生成一份。
- **Q5 `Any` vs `AnyView`**（已采纳，文档 `docs/concepts/any.rst:38/230` 背书）：字段→`Any`、
  返回→`Any`、**方法参数顶层 `Any`→`AnyView`**（嵌套仍 `Any`；仅在 step 5 渲染参数时特判，不改 `render_type`）。
- **Q6 tuple**：空→`()`，非空→`(T1, T2, ...)`。

实现期需顺手处理的小事（不阻塞，step 5 内解决）：

- **标识符净化**：FFI 字段/方法名若是 Rust 关键字或非法 ident（如 `type`/`match`），用 `r#` 原始标识符或跳过。
- `init_fields` 含 `kw_only`/`has_default` 元信息，Rust 无对应概念 → 一律按位置参数展开。

仍待确认：

- Rust 文件布局（步骤 9，属 P3，不挡 step 5）。
