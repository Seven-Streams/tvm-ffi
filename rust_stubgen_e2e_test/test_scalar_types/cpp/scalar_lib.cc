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
 * \file scalar_lib.cc
 * \brief C++ types testing scalar types (int, float, bool, str, None).
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <iostream>
#include <sstream>

namespace test_scalar_types {

namespace ffi = tvm::ffi;

class ScalarHolderObj : public ffi::Object {
 public:
  int64_t int_val;
  double float_val;
  bool bool_val;
  ffi::String string_val;

  explicit ScalarHolderObj(int64_t int_val = 0, double float_val = 0.0, bool bool_val = false,
                           ffi::String string_val = "")
      : int_val(int_val), float_val(float_val), bool_val(bool_val), string_val(string_val) {}

  static int64_t GetIntConstant() { return 100; }

  static double GetFloatConstant() { return 3.14; }

  static bool GetBoolConstant() { return true; }

  static ffi::String GetStringConstant() { return ffi::String("hello_world"); }

  static ffi::String FormatScalars(int64_t i, double f, bool b, ffi::String s) {
    std::ostringstream oss;
    oss << "int=" << i << ",float=" << f << ",bool=" << (b ? "true" : "false")
        << ",str=" << s.c_str();
    return ffi::String(oss.str());
  }

  void SetValues(int64_t i, double f, bool b, ffi::String s) {
    int_val = i;
    float_val = f;
    bool_val = b;
    string_val = s;
    std::cout << "[test_scalar_types] SetValues called: int=" << i << " float=" << f
              << " bool=" << (b ? "true" : "false") << " str=" << s.c_str() << std::endl;
  }

  ffi::String GetDescription() {
    std::ostringstream oss;
    oss << "ScalarHolder[int=" << int_val << ",float=" << float_val
        << ",bool=" << (bool_val ? "true" : "false") << ",str=" << string_val.c_str() << "]";
    return ffi::String(oss.str());
  }

  ~ScalarHolderObj() {
    std::cout << "[test_scalar_types] ~ScalarHolderObj() int=" << int_val << " float=" << float_val
              << std::endl;
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test_scalar_types.ScalarHolder", ScalarHolderObj, ffi::Object);
};

class ScalarHolder : public ffi::ObjectRef {
 public:
  explicit ScalarHolder(int64_t int_val = 0, double float_val = 0.0, bool bool_val = false,
                        ffi::String string_val = "") {
    data_ = ffi::make_object<ScalarHolderObj>(int_val, float_val, bool_val, string_val);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScalarHolder, ffi::ObjectRef, ScalarHolderObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ScalarHolderObj>()
      .def(refl::init<int64_t, double, bool, ffi::String>())
      .def_rw("int_val", &ScalarHolderObj::int_val, "integer value")
      .def_rw("float_val", &ScalarHolderObj::float_val, "floating point value")
      .def_rw("bool_val", &ScalarHolderObj::bool_val, "boolean value")
      .def_rw("string_val", &ScalarHolderObj::string_val, "string value")
      .def_static("get_int_constant", &ScalarHolderObj::GetIntConstant, "return 100")
      .def_static("get_float_constant", &ScalarHolderObj::GetFloatConstant, "return 3.14")
      .def_static("get_bool_constant", &ScalarHolderObj::GetBoolConstant, "return true")
      .def_static("get_string_constant", &ScalarHolderObj::GetStringConstant, "return hello_world")
      .def_static("format_scalars", &ScalarHolderObj::FormatScalars, "format all scalars")
      .def("set_values", &ScalarHolderObj::SetValues, "set all scalar values")
      .def("get_description", &ScalarHolderObj::GetDescription, "get description of all values");

  refl::TypeAttrDef<ScalarHolderObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<ScalarHolder>);
}

}  // namespace test_scalar_types
