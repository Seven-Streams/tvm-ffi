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
"""Generate downcast_test.so for tests/test_downcast.rs.

A 2-level reflected hierarchy ``For : Stmt : Object`` to exercise subtype-aware
downcasting of the generated Rust bindings:

  - make_for(min, extent) -> Stmt   (a ``For`` upcast to ``Stmt``)
  - make_stmt_array()      -> Array<Stmt>   (three ``For``s held at ``Stmt``)

The Rust side mirrors ``Stmt``/``For`` with matching type keys and asserts that a
base-typed value downcasts to its real subtype and that ``Array<Stmt>`` iterates
all elements.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

import tvm_ffi.cpp

CPP = r"""
#include <tvm/ffi/object.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace ffi {

class StmtObj : public Object {
 public:
  StmtObj() = default;
  TVM_FFI_DECLARE_OBJECT_INFO("test_downcast.Stmt", StmtObj, Object);
};

class Stmt : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Stmt, ObjectRef, StmtObj);
};

class ForObj : public StmtObj {
 public:
  int64_t min{0};
  int64_t extent{0};
  ForObj() = default;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test_downcast.For", ForObj, StmtObj);
};

class For : public Stmt {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(For, Stmt, ForObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<StmtObj>();
  refl::ObjectDef<ForObj>()
      .def_ro("min", &ForObj::min)
      .def_ro("extent", &ForObj::extent);
}

}  // namespace ffi
}  // namespace tvm

// Exported module functions must live in the global namespace.
using namespace tvm::ffi;

// Returns a `For` upcast to its base `Stmt`. The `Any` that crosses the FFI must
// carry `For`'s real header type_index so the Rust side can downcast it back.
Stmt make_for(int64_t min, int64_t extent) {
  ObjectPtr<ForObj> n = make_object<ForObj>();
  n->min = min;
  n->extent = extent;
  return For(n);
}

// Three `For`s held at the base `Stmt` type -- the Rust `Array<Stmt>::iter()`
// must yield all of them, each downcastable to `For`.
Array<Stmt> make_stmt_array() {
  Array<Stmt> arr;
  for (int64_t i = 1; i <= 3; ++i) {
    ObjectPtr<ForObj> n = make_object<ForObj>();
    n->extent = i;
    arr.push_back(For(n));  // For upcasts to Stmt
  }
  return arr;
}
"""


def main() -> None:
    """Build downcast_test.so into the output directory given as argv[1]."""
    if len(sys.argv) != 2:
        print("Usage: python generate_downcast_test_lib.py <output_dir>")
        sys.exit(1)
    out_dir = pathlib.Path(sys.argv[1])
    lib = tvm_ffi.cpp.build_inline(
        name="downcast_test",
        cpp_sources=CPP,
        functions=["make_for", "make_stmt_array"],
    )
    target = out_dir / "downcast_test.so"
    shutil.copy(lib, target)
    print(f"Generated downcast test library at {target}")


if __name__ == "__main__":
    main()
