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
 * \file expr_lib.cc
 * \brief Minimal C++ Expr type for cpp_rust_test (single int64 value field).
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <iostream>

namespace cpp_rust_test {

namespace ffi = tvm::ffi;

class ExprObj : public ffi::Object {
 public:
  int64_t value;

  explicit ExprObj(int64_t value) : value(value) {}

  ~ExprObj() {
    std::cout << "[cpp_rust_test] ~ExprObj() value=" << value << std::endl;
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("cpp_rust_test.Expr", ExprObj, ffi::Object);
};

class Expr : public ffi::ObjectRef {
 public:
  explicit Expr(int64_t value) { data_ = ffi::make_object<ExprObj>(value); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Expr, ffi::ObjectRef, ExprObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ExprObj>()
      .def(refl::init<int64_t>())
      .def_rw("value", &ExprObj::value, "scalar value");

  refl::TypeAttrDef<ExprObj>().def(refl::type_attr::kConvert,
                                   &refl::details::FFIConvertFromAnyViewToObjectRef<Expr>);

  refl::GlobalDef().def("cpp_rust_test.make_expr", [](int64_t v) { return Expr(v); });
}

}  // namespace cpp_rust_test