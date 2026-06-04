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
 * \file container_lib.cc
 * \brief C++ types testing container types (Array, Optional, tuple).
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <iostream>
#include <sstream>

namespace test_container_types {

namespace ffi = tvm::ffi;

class ArrayHolderObj : public ffi::Object {
 public:
  ffi::Array<int64_t> int_array;
  ffi::Array<double> float_array;
  ffi::Array<ffi::String> string_array;

  explicit ArrayHolderObj(ffi::Array<int64_t> int_array = ffi::Array<int64_t>(),
                          ffi::Array<double> float_array = ffi::Array<double>(),
                          ffi::Array<ffi::String> string_array = ffi::Array<ffi::String>())
      : int_array(int_array), float_array(float_array), string_array(string_array) {}

  static int64_t SumArray(ffi::Array<int64_t> arr) {
    int64_t sum = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
      sum += arr[i];
    }
    return sum;
  }

  static double AvgArray(ffi::Array<double> arr) {
    if (arr.size() == 0) return 0.0;
    double sum = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
      sum += arr[i];
    }
    return sum / arr.size();
  }

  int64_t GetIntArrayLength() { return int_array.size(); }

  ffi::String JoinStringArray() {
    std::ostringstream oss;
    for (size_t i = 0; i < string_array.size(); ++i) {
      if (i > 0) oss << ",";
      oss << string_array[i].c_str();
    }
    return ffi::String(oss.str());
  }

  void SetArrays(ffi::Array<int64_t> int_arr, ffi::Array<double> float_arr,
                 ffi::Array<ffi::String> str_arr) {
    int_array = int_arr;
    float_array = float_arr;
    string_array = str_arr;
    std::cout << "[test_container_types] SetArrays called with sizes: " << int_arr.size() << ", "
              << float_arr.size() << ", " << str_arr.size() << std::endl;
  }

  ~ArrayHolderObj() {
    std::cout << "[test_container_types] ~ArrayHolderObj() int_array.size=" << int_array.size()
              << std::endl;
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test_container_types.ArrayHolder", ArrayHolderObj, ffi::Object);
};

class ArrayHolder : public ffi::ObjectRef {
 public:
  explicit ArrayHolder(ffi::Array<int64_t> int_array = ffi::Array<int64_t>(),
                       ffi::Array<double> float_array = ffi::Array<double>(),
                       ffi::Array<ffi::String> string_array = ffi::Array<ffi::String>()) {
    data_ = ffi::make_object<ArrayHolderObj>(int_array, float_array, string_array);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ArrayHolder, ffi::ObjectRef, ArrayHolderObj);
};

class OptionalHolderObj : public ffi::Object {
 public:
  ffi::Optional<int64_t> opt_int;
  ffi::Optional<ffi::String> opt_string;

  explicit OptionalHolderObj(ffi::Optional<int64_t> opt_int = ffi::Optional<int64_t>(),
                             ffi::Optional<ffi::String> opt_string = ffi::Optional<ffi::String>())
      : opt_int(opt_int), opt_string(opt_string) {}

  static ffi::Optional<int64_t> CreateOptInt(int64_t value) {
    return ffi::Optional<int64_t>(value);
  }

  static ffi::Optional<int64_t> CreateNoneInt() { return ffi::Optional<int64_t>(); }

  ffi::String DescribeOptionals() {
    std::ostringstream oss;
    oss << "opt_int=" << (opt_int.has_value() ? std::to_string(opt_int.value()) : "None")
        << ",opt_string=" << (opt_string.has_value() ? opt_string.value().c_str() : "None");
    return ffi::String(oss.str());
  }

  ~OptionalHolderObj() { std::cout << "[test_container_types] ~OptionalHolderObj()" << std::endl; }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test_container_types.OptionalHolder", OptionalHolderObj,
                              ffi::Object);
};

class OptionalHolder : public ffi::ObjectRef {
 public:
  explicit OptionalHolder(ffi::Optional<int64_t> opt_int = ffi::Optional<int64_t>(),
                          ffi::Optional<ffi::String> opt_string = ffi::Optional<ffi::String>()) {
    data_ = ffi::make_object<OptionalHolderObj>(opt_int, opt_string);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(OptionalHolder, ffi::ObjectRef, OptionalHolderObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<ArrayHolderObj>()
      .def(refl::init<ffi::Array<int64_t>, ffi::Array<double>, ffi::Array<ffi::String>>())
      .def_rw("int_array", &ArrayHolderObj::int_array, "array of integers")
      .def_rw("float_array", &ArrayHolderObj::float_array, "array of floats")
      .def_rw("string_array", &ArrayHolderObj::string_array, "array of strings")
      .def_static("sum_array", &ArrayHolderObj::SumArray, "sum array of integers")
      .def_static("avg_array", &ArrayHolderObj::AvgArray, "average array of floats")
      .def("get_int_array_length", &ArrayHolderObj::GetIntArrayLength, "get int array length")
      .def("join_string_array", &ArrayHolderObj::JoinStringArray, "join string array")
      .def("set_arrays", &ArrayHolderObj::SetArrays, "set all arrays");

  refl::TypeAttrDef<ArrayHolderObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<ArrayHolder>);

  refl::ObjectDef<OptionalHolderObj>()
      .def(refl::init<ffi::Optional<int64_t>, ffi::Optional<ffi::String>>())
      .def_rw("opt_int", &OptionalHolderObj::opt_int, "optional integer")
      .def_rw("opt_string", &OptionalHolderObj::opt_string, "optional string")
      .def_static("create_opt_int", &OptionalHolderObj::CreateOptInt, "create optional int")
      .def_static("create_none_int", &OptionalHolderObj::CreateNoneInt, "create None int")
      .def("describe_optionals", &OptionalHolderObj::DescribeOptionals, "describe optional values");

  refl::TypeAttrDef<OptionalHolderObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<OptionalHolder>);
}

}  // namespace test_container_types
