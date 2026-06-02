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
 * \brief C++ Expr / Add types for cpp_rust_test.
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

  void SetValue(int64_t new_value) {
    value = new_value;
    std::cout << "[cpp_rust_test] ExprObj::SetValue() value=" << value << std::endl;
  }

  ~ExprObj() {
    std::cout << "[cpp_rust_test] ~ExprObj() value=" << value << std::endl;
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("cpp_rust_test.Expr", ExprObj, ffi::Object);
};

class Expr : public ffi::ObjectRef {
 public:
  explicit Expr(int64_t value) { data_ = ffi::make_object<ExprObj>(value); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Expr, ffi::ObjectRef, ExprObj);
};

class AddObj : public ExprObj {
 public:
  Expr a;
  Expr b;

  AddObj(Expr a, Expr b, int64_t value) : ExprObj(value), a(std::move(a)), b(std::move(b)) {}

  void Update() {
    value = a->value + b->value;
    std::cout << "[cpp_rust_test] AddObj::Update() value=" << value << " (a=" << a->value
              << " + b=" << b->value << ")" << std::endl;
  }

  ~AddObj() {
    std::cout << "[cpp_rust_test] ~AddObj() value=" << value << " a->value=" << a->value
              << " b->value=" << b->value << std::endl;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("cpp_rust_test.Add", AddObj, ExprObj);
};

class Add : public Expr {
 public:
  Add(Expr a, Expr b, int64_t value) : Expr(ffi::UnsafeInit{}) {
    data_ = ffi::make_object<AddObj>(std::move(a), std::move(b), value);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Add, Expr, AddObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ExprObj>()
      .def(refl::init<int64_t>())
      .def_ro("value", &ExprObj::value, "scalar value")
      .def("set_value", &ExprObj::SetValue, "set scalar value");

  refl::TypeAttrDef<ExprObj>().def(refl::type_attr::kConvert,
                                   &refl::details::FFIConvertFromAnyViewToObjectRef<Expr>);

  refl::ObjectDef<AddObj>()
      .def(refl::init<Expr, Expr, int64_t>())
      .def_rw("a", &AddObj::a, "left Expr")
      .def_rw("b", &AddObj::b, "right Expr")
      .def("update", &AddObj::Update, "set value to a.value + b.value");

  refl::TypeAttrDef<AddObj>().def(refl::type_attr::kConvert,
                                   &refl::details::FFIConvertFromAnyViewToObjectRef<Add>);

  refl::GlobalDef()
      .def("cpp_rust_test.make_expr", [](int64_t v) { return Expr(v); })
      .def("cpp_rust_test.make_add",
           [](Expr a, Expr b, int64_t value) { return Add(std::move(a), std::move(b), value); });
}

}  // namespace cpp_rust_test