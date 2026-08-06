..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _sec-stubgen:

Stub Generation
===============

TVM-FFI provides ``tvm-ffi-stubgen``, a tool that generates typed language bindings from C++
reflection metadata. It turns registered global functions and classes into native type hints,
enabling IDE auto-completion and static type checking.

The sections below describe the default Python target. The Rust target is covered in
:ref:`sec-stubgen-rust`.

.. admonition:: Prerequisite
   :class: hint

   This tutorial uses `examples/python_packaging <https://github.com/apache/tvm-ffi/tree/main/examples/python_packaging>`_
   as a running example. The package registers global functions (``my_ffi_extension.add_one``,
   ``my_ffi_extension.raise_error``) and an object type (``my_ffi_extension.IntPair``).

.. _sec-stubgen-cmake:

CMake-based Generation
----------------------

The recommended approach is to use CMake's ``tvm_ffi_configure_target`` function.
This runs stub generation automatically after each build.

.. code-block:: cmake

   tvm_ffi_configure_target(<target>
       STUB_DIR <dir>
       [STUB_INIT ON|OFF]
       [STUB_PKG <pkg>]
       [STUB_PREFIX <prefix>]
       [STUB_TARGET python rust]   # one or more backends
   )

   # Or, to generate Python and Rust stubs together (each backend writes a
   # different tree, so the output directories are given per backend):
   tvm_ffi_configure_target(<target>
       STUB_TARGET python rust
       STUB_DIR_PYTHON <python-dir>
       STUB_DIR_RUST <rust-dir>
       [STUB_INIT ON|OFF]
       [STUB_PKG <pkg>]
       [STUB_PREFIX <prefix>]
   )

From the example's
`CMakeLists.txt <https://github.com/apache/tvm-ffi/blob/main/examples/python_packaging/CMakeLists.txt>`_:

.. code-block:: cmake

   add_library(my_ffi_extension SHARED src/extension.cc)
   tvm_ffi_configure_target(my_ffi_extension STUB_DIR "./python" STUB_INIT ON)

STUB_DIR
   Required. Directory containing Python source files, relative to ``CMAKE_CURRENT_SOURCE_DIR``.

STUB_INIT (default: ``OFF``)
   Optional. When ``OFF``, fills in existing inline directives without creating
   new ones. When ``ON``, generates new files and directives.

``STUB_INIT`` is ``OFF``
~~~~~~~~~~~~~~~~~~~~~~~~

The tool fills in existing inline directives only. It scans files for directive markers
and regenerates the content within those blocks:

- Class fields and methods (``object/<type_key>``)

  .. code-block:: python

     @tvm_ffi.register_object("my_ffi_extension.IntPair")
     class IntPair(tvm_ffi.Object):
         # tvm-ffi-stubgen(begin): object/my_ffi_extension.IntPair
         a: int
         b: int
         if TYPE_CHECKING:
             def __init__(self, a: int, b: int) -> None: ...
             def __ffi_init__(self, a: int, b: int) -> None: ...
             def sum(self, /) -> int: ...
         # tvm-ffi-stubgen(end)

- Global functions (``global/<prefix>``)

  .. code-block:: python

     # tvm-ffi-stubgen(begin): global/my_ffi_extension
     tvm_ffi.init_ffi_api("my_ffi_extension", __name__)
     if TYPE_CHECKING:
         def add_one(_0: int, /) -> int: ...
         def raise_error(_0: str, /) -> None: ...
     # tvm-ffi-stubgen(end)

- Import sections, ``__all__`` lists, and exports (``import-section``, ``__all__``, ``export/<module>``)

See :ref:`sec-stubgen-directives` for a complete reference of all inline directives.

.. important::

   No new files or directives are created. Use this mode after you have customized the
   generated files and want to preserve your changes.

``STUB_INIT`` is ``ON``
~~~~~~~~~~~~~~~~~~~~~~~

The tool generates new files (``_ffi_api.py``, ``__init__.py``) and new directives from
scratch. This is the recommended starting point for new projects. Two additional options
are available:

STUB_PKG (default: ``SKBUILD_PROJECT_NAME`` or CMake target name)
   Python package name. Determines the directory structure (``<STUB_DIR>/<STUB_PKG>/``) and
   generates the library loading code in ``_ffi_api.py``:

   .. code-block:: python

      LIB = tvm_ffi.libinfo.load_lib_module("<STUB_PKG>", "<cmake-target-name>")

   This uses :py:func:`tvm_ffi.libinfo.load_lib_module` to load the shared library.

STUB_PREFIX (default: ``<STUB_PKG>.``)
   Registry prefix filter. Only functions and classes whose registered names start with this
   prefix are included. The tool generates:

   - ``global/<prefix>`` directives for global functions. In ``<STUB_DIR>/<STUB_PKG>/_ffi_api.py``:

     .. code-block:: python

        # tvm-ffi-stubgen(begin): global/<STUB_PREFIX>
        ...
        # tvm-ffi-stubgen(end)

     Functions in nested namespaces (e.g., ``<STUB_PREFIX>submodule.func``) are placed in
     corresponding subdirectories.

   - ``object/<type_key>`` directives for each registered class. For a class with type key
     ``<STUB_PREFIX>IntPair``:

     .. code-block:: python

        @tvm_ffi.register_object("<STUB_PREFIX>IntPair")
        class IntPair(_ffi_Object):
            # tvm-ffi-stubgen(begin): object/<STUB_PREFIX>IntPair
            a: int
            b: int
            if TYPE_CHECKING:
                def __init__(self, a: int, b: int) -> None: ...
                def __ffi_init__(self, a: int, b: int) -> None: ...
                def sum(self, /) -> int: ...
            # tvm-ffi-stubgen(end)

     Classes in nested namespaces are similarly placed in corresponding subdirectories.


.. _sec-stubgen-cli:

CLI-based Generation
--------------------

For standalone usage or custom build systems, use the command-line tool directly.
The CLI options correspond to CMake options as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - CLI Option
     - CMake Option
     - Description
   * - ``<directory>``
     - ``STUB_DIR``
     - Directory containing Python source files
   * - ``--init-*`` flags
     - ``STUB_INIT ON``
     - Presence of ``--init-*`` flags enables full generation
   * - ``--init-pypkg``
     - ``STUB_PKG``
     - Python package name
   * - ``--init-prefix``
     - ``STUB_PREFIX``
     - Registry prefix filter
   * - ``--init-lib``
     - *(target name)*
     - CMake target name for the shared library

Basic Usage
~~~~~~~~~~~

To generate complete stub files from scratch (equivalent to ``STUB_INIT ON``):

.. code-block:: bash

   tvm-ffi-stubgen                          \
     python/my_ffi_extension                \
     --dlls build/libmy_ffi_extension.so    \
     --init-pypkg my-ffi-extension          \
     --init-lib my_ffi_extension            \
     --init-prefix "my_ffi_extension."

To update only existing directives (equivalent to ``STUB_INIT OFF``), omit the ``--init-*`` flags:

.. code-block:: bash

   tvm-ffi-stubgen                          \
     python/my_ffi_extension                \
     --dlls build/libmy_ffi_extension.so

Command Line Options
~~~~~~~~~~~~~~~~~~~~

**Required arguments:**

``<directory>`` (positional)
   Directory to generate/update Python stubs. Corresponds to ``STUB_DIR``.

``--dlls``
   Shared libraries to load during generation. The stub generator loads these DLLs to
   discover registered global functions and object types. Without loading the DLLs, the
   generator cannot know what functions and classes exist. Use semicolons to separate
   multiple libraries.

**Arguments for full generation** (corresponds to ``STUB_INIT ON``):

All three are required together. When omitted, the tool operates in directive-only mode
(equivalent to ``STUB_INIT OFF``).

``--init-pypkg``
   Python package name. Corresponds to ``STUB_PKG``.

``--init-lib``
   CMake target name. Used as the second argument in the generated
   ``load_lib_module("<STUB_PKG>", "<init-lib>")`` call.

``--init-prefix``
   Registry prefix filter. Corresponds to ``STUB_PREFIX``.

**Optional arguments:**

``--target``
   Code generator backend: ``python`` (default) or ``rust``. See :ref:`sec-stubgen-rust`.

``--verbose``
   Print a unified diff of changes to each file.

``--dry-run``
   Preview changes without writing to files.

``--imports``
   Additional Python modules to import before generation (semicolon-separated).

``--indent``
   Indentation width for generated code (default: 4).

For a complete list of options, run ``tvm-ffi-stubgen --help``.

.. _sec-stubgen-rust:

Rust Code Stubgen (Experimental)
--------------------------------

TVM-FFI exposes registered C++ classes to Rust through opaque object handles.
``tvm-ffi-stubgen`` generates the Rust object/reference types plus typed accessors
for the reflection metadata; it does not duplicate the trailing C++ object layout.

To generate Rust stubs, pass ``STUB_TARGET rust`` to ``tvm_ffi_configure_target``
(see :ref:`sec-stubgen-cmake`) or ``--target rust`` on the command line
(see :ref:`sec-stubgen-cli`).

To generate Python and Rust stubs from a single target, list both backends in
``STUB_TARGET``. Because each backend writes a different file tree, give the output
directories per backend via ``STUB_DIR_PYTHON`` and ``STUB_DIR_RUST`` (instead of
``STUB_DIR``):

.. code-block:: cmake

   tvm_ffi_configure_target(my_ffi_extension
       STUB_TARGET python rust
       STUB_INIT ON
       STUB_DIR_PYTHON "./python"
       STUB_DIR_RUST "./rust/src/generated")

This is equivalent to invoking stub generation once per listed backend (with
``--target python`` and ``--target rust``). The shared ``STUB_PKG`` / ``STUB_PREFIX``
apply to every listed backend.

Key Features
~~~~~~~~~~~~

- Generate Rust code for registered C++ classes automatically (via CLI or CMake).
- Read fields through owning reflection getters, including inherited fields.
- Construct objects through their registered ``__ffi_init__`` function.
- Expose inherited instance methods and static methods through typed FFI calls.
- Preserve dynamic schemas losslessly as ``Any`` / ``AnyView``.

Stubgen only rewrites its marked regions and does not run ``rustfmt`` over the
surrounding file. Run the project's normal ``cargo fmt`` step after generation.

Generation Output
~~~~~~~~~~~~~~~~~

This section uses `examples/rust_stubgen <https://github.com/apache/tvm-ffi/tree/main/examples/rust_stubgen>`_
as the running example. The library registers a single type, ``rust_stubgen.IntPair``,
with two required fields ``a`` / ``b``, a defaulted field ``scale``, and a method ``sum``:

.. literalinclude:: ../../examples/rust_stubgen/src/int_pair.cc
   :language: cpp
   :start-after: [object.begin]
   :end-before: [object.end]

For this type the tool generates the following Rust code (bodies abridged):

.. code-block:: rust

   #[repr(C)]
   #[derive(tvm_ffi::derive::Object)]
   #[type_key = "rust_stubgen.IntPair"]
   pub struct IntPairObj {
       base: Object,
   }

   #[repr(C)]
   #[derive(tvm_ffi::derive::ObjectRef, Clone)]
   pub struct IntPair {
       data: ObjectArc<IntPairObj>,
   }

   impl IntPairObj {
       pub fn a(&self) -> Result<i64> { /* owning reflection getter */ }
       pub fn b(&self) -> Result<i64> { /* owning reflection getter */ }
       pub fn scale(&self) -> Result<i64> { /* owning reflection getter */ }
       pub fn sum(&self) -> Result<i64> { /* reflected C++ method */ }
   }

   impl IntPair {
       pub fn ffi_new(a: i64, b: i64) -> Result<Self> { /* reflected constructor */ }
   }

``IntPairObj`` contains only the object-system prefix needed for typed
``Deref`` and inheritance. The allocation remains owned by the runtime, and every
getter returns an owning Rust value, so a returned string/container/object can
outlive the source handle. Instance methods live on ``IntPairObj`` so a derived
object inherits them through its ``Deref`` chain.

Construction also crosses the FFI boundary. An explicit reflected constructor
uses its registered signature. For an auto-generated ``__ffi_init__``, Rust has
no default or keyword-only parameters, so ``ffi_new`` takes every
``init=true`` field and sends it by reflected name. This includes fields with a
C++ default and lets a rewrite pass preserve non-default spans, annotations, and
other metadata exactly. Fields declared ``init=false`` remain runtime-managed.

.. code-block:: rust

   use tvm_ffi::ObjectRefCore;

   let pair = IntPair::ffi_new(1, 2)?;
   assert_eq!(pair.a()?, 1);
   assert_eq!(pair.scale()?, 1);
   assert_eq!(pair.sum()?, 3);
   assert!(pair.same_as(&pair.clone()));

Limitations
~~~~~~~~~~~

Typed ``Array<T>``, ``Map<K, V>``, and ``Optional<T>`` values render as the
corresponding Rust container types. A schema that cannot be represented
faithfully -- for example ``Map<String, Any>``, ``List``, ``Dict``, ``Union``,
or ``tuple`` -- remains available as an owning ``Any`` result or an ``AnyView``
parameter. It is never rewritten to a misleading ``ObjectRef`` element type.

Rust has no function overloading. If reflection reports multiple signatures
with the same method name and receiver kind, stubgen skips that typed method
with a warning; callers can still use ``Function::from_type_method`` explicitly.

Generated fields are read-only accessors. Rewriting code should construct a
replacement object through ``ffi_new`` or another registered factory rather
than mutating shared object storage. Global registry functions are not yet
emitted as typed Rust wrappers and remain available through
``Function::get_global``.


.. _sec-stubgen-advanced:

Advanced Topics
---------------

.. _sec-stubgen-directives:

Inline Directives
~~~~~~~~~~~~~~~~~

Inline directives are special comments that mark regions where the tool should generate
or update code. They allow you to:

- Add custom code alongside generated stubs
- Control the exact placement of generated code
- Maintain hand-written documentation or additional methods
- Integrate with existing codebases

The tool preserves content outside directive blocks, only modifying marked regions.

**Directive Syntax**

Directives use comment markers to delimit generated regions:

.. code-block:: python

   # tvm-ffi-stubgen(begin): <directive-type>/<parameter>
   # ... generated code appears here ...
   # tvm-ffi-stubgen(end)

When you run the tool, it:

1. Parses files for directive markers
2. Regenerates content between ``begin`` and ``end`` markers
3. Preserves everything outside the markers

**Directive Reference**

``global/<prefix>`` - Global Functions
   Marks a section for global function stubs. All functions registered with the given
   prefix are included.

   .. code-block:: python

      # tvm-ffi-stubgen(begin): global/my_ext.arith
      # fmt: off
      tvm_ffi.init_ffi_api("my_ext.arith", __name__)
      if TYPE_CHECKING:
          def add_one(_0: int, /) -> int: ...
          def add_two(_0: int, /) -> int: ...
      # fmt: on
      # tvm-ffi-stubgen(end)

``object/<type_key>`` - Object Types
   Marks a section for class field and method stubs. Place inside a class definition
   decorated with ``@tvm_ffi.register_object``.

   .. code-block:: python

      @tvm_ffi.register_object("my_ffi_extension.IntPair")
      class IntPair(_ffi_Object):
          # tvm-ffi-stubgen(begin): object/my_ffi_extension.IntPair
          # fmt: off
          a: int
          b: int
          if TYPE_CHECKING:
              def __init__(self, a: int, b: int) -> None: ...
              def __ffi_init__(self, a: int, b: int) -> None: ...
              def sum(self, /) -> int: ...
          # fmt: on
          # tvm-ffi-stubgen(end)

``import-section`` - Import Section
   Populates import statements for types used in generated stubs. The tool automatically
   determines which imports are needed.

   .. code-block:: python

      # tvm-ffi-stubgen(begin): import-section
      # fmt: off
      # isort: off
      from __future__ import annotations
      from tvm_ffi import init_ffi_api as _FFI_INIT_FUNC
      from typing import TYPE_CHECKING
      if TYPE_CHECKING:
          from collections.abc import Mapping, Sequence
          from tvm_ffi import Device, Object, Tensor, dtype
      # isort: on
      # fmt: on
      # tvm-ffi-stubgen(end)

``export/<module>`` - Export
   Re-exports names from a submodule, typically used in ``__init__.py`` to aggregate exports.

   .. code-block:: python

      # tvm-ffi-stubgen(begin): export/_ffi_api
      # fmt: off
      # isort: off
      from ._ffi_api import *  # noqa: F403
      from ._ffi_api import __all__ as _ffi_api__all__
      if "__all__" not in globals():
          __all__ = []
      __all__.extend(_ffi_api__all__)
      # isort: on
      # fmt: on
      # tvm-ffi-stubgen(end)

``__all__`` - Export List
   Populates the ``__all__`` list with all generated class and function names.

   .. code-block:: python

      __all__ = [
          # tvm-ffi-stubgen(begin): __all__
          "LIB",
          "IntPair",
          "add_one",
          # tvm-ffi-stubgen(end)
      ]

``ty-map`` - Type Mapping
   Maps a C++ type key to a different Python type path. This is useful when the Python
   class is defined in a different module than the type key suggests.

   .. code-block:: python

      # tvm-ffi-stubgen(ty-map): ffi.reflection.AccessStep -> ffi.access_path.AccessStep

``import-object`` - Import Object
   Injects a custom import into generated code. The format is
   ``<full_name>;<type_checking_only>;<alias>``.

   .. code-block:: python

      # tvm-ffi-stubgen(import-object): ffi.Object;False;_ffi_Object

   This imports ``ffi.Object`` as ``_ffi_Object`` for use in generated code. The second
   field (``False``) indicates the import is not TYPE_CHECKING-only. The Rust target
   records the name as a plain ``use`` and ignores the other two fields.

``skip-file`` - Skip File
   Prevents the tool from modifying the file. Place anywhere in the file.

   .. code-block:: python

      # tvm-ffi-stubgen(skip-file)
