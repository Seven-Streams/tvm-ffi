/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file int_pair.cc
 * \brief Minimal C++ extension registering an `IntPair` object for the Rust
 *        stubgen example (mirrors the `IntPair` used in docs/packaging/stubgen.rst).
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace rust_stubgen {

namespace ffi = tvm::ffi;

// Forward declaration so a method can take/return the `IntPair` ref wrapper.
class IntPair;

/*! \brief Data object: a pair of 64-bit integers `a` and `b`. */
class IntPairObj : public ffi::Object {
 public:
  int64_t a;
  int64_t b;

  IntPairObj() = default;
  explicit IntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  /*! \brief Sum of the two components. */
  int64_t sum() const { return a + b; }

  /*! \brief Multiply both components in place by `factor`. */
  void scale(int64_t factor) {
    a *= factor;
    b *= factor;
  }

  /*! \brief Return a fresh `IntPair` with the components swapped. */
  IntPair swapped() const;

  /*! \brief Throws on a zero divisor; otherwise returns `a / divisor`. */
  int64_t checked_div(int64_t divisor) const {
    if (divisor == 0) {
      TVM_FFI_THROW(ValueError) << "IntPair::checked_div: division by zero";
    }
    return a / divisor;
  }

  // All fields are writable, so the generated Rust wrapper gets `DerefMut`.
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("rust_stubgen.IntPair", IntPairObj, ffi::Object);
};

/*! \brief Reference wrapper for `IntPairObj`. */
class IntPair : public ffi::ObjectRef {
 public:
  explicit IntPair(int64_t a, int64_t b) { data_ = ffi::make_object<IntPairObj>(a, b); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntPair, ffi::ObjectRef, IntPairObj);
};

// Out-of-line: `IntPair` is now a complete type.
inline IntPair IntPairObj::swapped() const { return IntPair(b, a); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<IntPairObj>()
      .def(refl::init<int64_t, int64_t>())
      .def_rw("a", &IntPairObj::a, "first component")
      .def_rw("b", &IntPairObj::b, "second component")
      .def("sum", &IntPairObj::sum, "a + b")
      .def("scale", &IntPairObj::scale, "multiply both components by factor (in place)")
      .def("swapped", &IntPairObj::swapped, "return a new IntPair with a and b swapped")
      .def("checked_div", &IntPairObj::checked_div, "a / divisor, throws on zero");

  // Lets an `AnyView` holding this object convert back into the `IntPair` ref,
  // which is what the generated Rust bindings rely on for object returns.
  refl::TypeAttrDef<IntPairObj>().def(refl::type_attr::kConvert,
                                      &refl::details::FFIConvertFromAnyViewToObjectRef<IntPair>);
}

}  // namespace rust_stubgen
