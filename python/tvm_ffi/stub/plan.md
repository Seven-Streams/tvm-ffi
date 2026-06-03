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
     - **object：生成 `struct`（字段）+ `impl`（方法）**。方法的调用机制（方法存于类型
       反射方法表 `TVMFFIMethodInfo.method`，首参恒为 `self`，**不在**全局注册表）由用户
       后续提供；当前先按选项 2 生成方法签名 + 占位体（如 `todo!()`），待接入。

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
| §4⑤ | `rust_backend` 常量表（类型/路径/builtin） | ⬜ 待做（P1） |
| §4④(Rust) | Rust import 表示（`use` 收集器） | ⬜ 待做（P1） |
| §4② | `RustBackend.render_type` 类型遍历 + 不支持类型抛异常 | ⬜ 待做（P2） |
| §4① | `rust_backend/codegen.py`：object 生成（struct+impl）；**global 函数不生成**（决策 5） | ⬜ 待做（P2） |
| §4⑥ | `cli.py` 目标文件名由 backend 决定 | ⬜ 待做（P3，依赖布局决策） |

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
  - `RUST_TY_MAP_DEFAULTS`：FFI origin → Rust 类型名，按下表（已核对 crate）：

    | FFI origin | Rust | 备注 |
    | --- | --- | --- |
    | `int` | `i64` | crate 中所有整型都映射到 FFI `int`，默认回填 `i64` |
    | `float` | `f64` | `f32/f64` → FFI `float`，默认 `f64` |
    | `bool` | `bool` | |
    | `None` | `()` | crate 用 `()` 表示 None/void |
    | `Optional` | `Option` | `Option<T>` |
    | `Any` | `Any` | 另有 `AnyView`（非拥有），按位置可能用引用 |
    | `Callable` | `Function` | `ffi.Function` |
    | `Array` | `Array` | **`Array<T>`，不是 `Vec`** |
    | `Object` | `Object` | |
    | `Tensor` | `Tensor` | `ffi.Tensor` |
    | `Shape` | `Shape` | `ffi.Shape` |
    | `Device` | `Device` | type_str = `"Device"` |
    | `dtype` / `DataType` | `DataType` | dtype.rs，type_str = `"DataType"` |
    | `ffi.String` | `String` | `tvm_ffi::String`（注意与 std 区分） |
    | `ffi.Bytes` | `Bytes` | |
    | `ffi.Module` | `Module` | |
    | `ffi.Error` | `Error` | |

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

#### 步骤 5：`generate_rust_object`（struct + impl）

- 位置：`rust_backend/codegen.py`。
- 从 `ObjectInfo` 渲染（决策 5）：
  - 字段 → `struct` 字段（类型经 render_type）。
  - 方法 → `impl` 方法**签名 + 占位体**（如 `todo!()`）。方法的真正调用机制
    （从类型反射方法表取 `TVMFFIMethodInfo.method`、首参 `self`）由用户后续提供，
    届时把占位体替换为真实分发。member vs static、`init`/`ffi_init` 的处理策略待定。
- 写入 `code.lines`（mutate，遵循 `code.lines[0] + 缩进体 + code.lines[-1]` 模式）。
- **warn+skip（决策 1）**：整个 object 的渲染**包一层** `try/except
  UnsupportedTypeError`——任一字段/方法命中不支持类型，就打
  `[Skipped] object <type_key>: 不支持的类型 <...>`（黄色，仿 `lib_state` 现有风格），
  并把该 object 的 marker 块**留空**（仅 `code.lines[0]` + `code.lines[-1]`），不抛到文件级。
- **测试**：`test_rust_codegen.py::test_object`（正常渲染：struct 字段 + impl 方法签名）
  - `test_object_skipped`（含 `Map` 字段的 object → 块为空 + 有 warning）。
- **验收**：含字段与若干方法的 object 能生成可读 Rust；含不支持字段的 object 被整体跳过。

#### 步骤 6：全局函数块 = **no-op**（决策 5）

- `global/<prefix>` 块在 Rust 后端**不生成任何内容**——Rust 运行时经
  `Function::get_global` 动态调用，无需 stub。
- `RustBackend.generate_global_funcs_block` 实现为**空操作**（marker 块保持原样/留空），
  `--init` 脚手架（步骤 11）也**不产出** `global/` marker。
- **测试**：`test_rust_codegen.py::test_global_funcs_noop`，断言 `global/` 块经过后内容不变。
- **验收**：含 `global/` 块的 `.rs` 文件处理后该块无新增内容、不报错。

#### 步骤 7：`generate_rust_import_section` / `generate_rust_all` / `generate_rust_export`

- `import_section`：把 `RustImports` 渲染成 `use` 行，按 `defined_types` 过滤本地定义。
- `all`：Rust 公开再导出（`pub use`）。
- `export`：子模块再导出（`pub mod` / `pub use`）。
- **测试**：各一条断言生成行。
- **验收**：三种块都能生成；过滤逻辑有测试。

#### 步骤 8：装配 `RustBackend`（去掉 NotImplementedError）

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
`Map`/`Dict`/`List` 确认不支持、需报错。仍待明确：

- `Any` vs `AnyView`：返回/参数/字段位置分别用拥有型 `Any` 还是非拥有 `AnyView`/引用。
- `Callable` 在签名位置 vs 字段位置是否都映射成 `Function`（带不带泛型签名）。
- `__init__`/`__ffi_init__` 在 Rust 端如何表达（构造函数？关联函数？先 TODO）。
- `tuple`（含空 tuple）的 Rust 落点（`()` 还是元组类型）。
- `dtype` 用 `DataType` 还是 `DLDataTypeExt`（lib.rs 再导出的是后者）。
- Rust 文件布局（步骤 9）。
