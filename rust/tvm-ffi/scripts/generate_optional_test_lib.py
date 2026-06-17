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
"""Generate optional_test.so for tests/test_optional.rs.

A reflection object whose fields cover all three ffi::Optional<T> layout categories:

  - a: Optional<int64_t>     -> Category C (std::optional fallback)
  - b: Optional<Array<Any>>  -> Category A (ObjectRef pointer)
  - c: Optional<String>      -> Category B (embedded String/Any)
  - tail: int64_t            -> sentinel to check the following-field offset

Exposes module functions:
  - poc_create(a_has, a_val, b_has, c_has, c_val) -> PoC
  - poc_layout() -> String   (runtime offsets/sizes for cross-check)
"""

import pathlib
import shutil
import sys

import tvm_ffi.cpp

CPP = r"""
#include <tvm/ffi/optional.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <cstdio>

namespace tvm {
namespace ffi {

class PoCObj : public Object {
 public:
  Optional<int64_t> a;      // Category C
  Optional<Array<Any>> b;   // Category A
  Optional<String> c;       // Category B
  int64_t tail{0};          // sentinel

  static constexpr bool _type_mutable = true;
  PoCObj() = default;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("poc.PoCObj", PoCObj, Object);
};

class PoC : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PoC, ObjectRef, PoCObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<PoCObj>()
      .def_rw("a", &PoCObj::a)
      .def_rw("b", &PoCObj::b)
      .def_rw("c", &PoCObj::c)
      .def_rw("tail", &PoCObj::tail);
}

}  // namespace ffi
}  // namespace tvm

// Exported module functions must live in the global namespace.
using namespace tvm::ffi;

PoC poc_create(int64_t a_has, int64_t a_val, int64_t b_has, int64_t c_has, String c_val) {
  ObjectPtr<PoCObj> n = make_object<PoCObj>();
  if (a_has) n->a = a_val;
  if (b_has) n->b = Array<Any>({Any(int64_t(10)), Any(int64_t(20)), Any(int64_t(30))});
  if (c_has) n->c = c_val;
  n->tail = 0x7eeeeeee;
  return PoC(n);
}

String poc_layout() {
  PoCObj o;
  char* base = reinterpret_cast<char*>(&o);
  char buf[640];
  std::snprintf(
      buf, sizeof(buf),
      "sizeof=%zu off_a=%td off_b=%td off_c=%td off_tail=%td "
      "sz_a=%zu sz_b=%zu sz_c=%zu al_a=%zu al_b=%zu al_c=%zu",
      sizeof(PoCObj),
      reinterpret_cast<char*>(&o.a) - base,
      reinterpret_cast<char*>(&o.b) - base,
      reinterpret_cast<char*>(&o.c) - base,
      reinterpret_cast<char*>(&o.tail) - base,
      sizeof(Optional<int64_t>), sizeof(Optional<Array<Any>>), sizeof(Optional<String>),
      alignof(Optional<int64_t>), alignof(Optional<Array<Any>>), alignof(Optional<String>));
  return String(buf);
}
"""


def main() -> None:
    """Build optional_test.so into the output directory given as argv[1]."""
    if len(sys.argv) != 2:
        print("Usage: python generate_optional_test_lib.py <output_dir>")
        sys.exit(1)
    out_dir = pathlib.Path(sys.argv[1])
    lib = tvm_ffi.cpp.build_inline(
        name="optional_test",
        cpp_sources=CPP,
        functions=["poc_create", "poc_layout"],
    )
    target = out_dir / "optional_test.so"
    shutil.copy(lib, target)
    print(f"Generated optional test library at {target}")


if __name__ == "__main__":
    main()
