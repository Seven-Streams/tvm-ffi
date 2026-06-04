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
 * \file ffi_types_lib.cc
 * \brief C++ types testing core FFI value types: `ffi::Shape` (F2) and
 *        `DataType`/`Device` (F3) in param / return / field positions.
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <vector>

namespace test_ffi_types {

namespace ffi = tvm::ffi;

class FfiTypesHolderObj : public ffi::Object {
 public:
  ffi::Shape shape;
  DLDataType dtype;
  DLDevice device;

  explicit FfiTypesHolderObj(ffi::Shape shape = ffi::Shape(std::vector<int64_t>{}),
                             DLDataType dtype = DLDataType{kDLFloat, 32, 1},
                             DLDevice device = DLDevice{kDLCPU, 0})
      : shape(shape), dtype(dtype), device(device) {}

  // --- F2: ffi::Shape as param / return -------------------------------------
  static int64_t ShapeProduct(ffi::Shape s) {
    int64_t product = 1;
    for (size_t i = 0; i < s.size(); ++i) {
      product *= s[i];
    }
    return product;
  }

  static ffi::Shape MakeShape(int64_t a, int64_t b) {
    return ffi::Shape(std::vector<int64_t>{a, b});
  }

  int64_t ShapeNdim() { return static_cast<int64_t>(shape.size()); }

  // --- F3: DataType / Device as param / return ------------------------------
  static DLDataType EchoDtype(DLDataType dt) { return dt; }

  static int64_t DtypeBits(DLDataType dt) { return dt.bits; }

  static DLDevice EchoDevice(DLDevice d) { return d; }

  static int64_t DeviceId(DLDevice d) { return d.device_id; }

  // --- G: Function (callback) as param / return -----------------------------
  // G1: receive a callback and invoke it.
  static int64_t ApplyFn(ffi::Function fn, int64_t x) { return fn(x).cast<int64_t>(); }

  // G2: return a callback (a closure capturing `n`).
  static ffi::Function MakeAdder(int64_t n) {
    return ffi::Function::FromTyped([n](int64_t x) -> int64_t { return n + x; }, "adder");
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test_ffi_types.FfiTypesHolder", FfiTypesHolderObj, ffi::Object);
};

class FfiTypesHolder : public ffi::ObjectRef {
 public:
  explicit FfiTypesHolder(ffi::Shape shape = ffi::Shape(std::vector<int64_t>{}),
                          DLDataType dtype = DLDataType{kDLFloat, 32, 1},
                          DLDevice device = DLDevice{kDLCPU, 0}) {
    data_ = ffi::make_object<FfiTypesHolderObj>(shape, dtype, device);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FfiTypesHolder, ffi::ObjectRef, FfiTypesHolderObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<FfiTypesHolderObj>()
      .def(refl::init<ffi::Shape, DLDataType, DLDevice>())
      .def_rw("shape", &FfiTypesHolderObj::shape, "tensor shape")
      .def_rw("dtype", &FfiTypesHolderObj::dtype, "data type")
      .def_rw("device", &FfiTypesHolderObj::device, "device")
      .def_static("shape_product", &FfiTypesHolderObj::ShapeProduct, "product of shape dims")
      .def_static("make_shape", &FfiTypesHolderObj::MakeShape, "build a 2-D shape")
      .def_static("echo_dtype", &FfiTypesHolderObj::EchoDtype, "return the DataType unchanged")
      .def_static("dtype_bits", &FfiTypesHolderObj::DtypeBits, "bit width of a DataType")
      .def_static("echo_device", &FfiTypesHolderObj::EchoDevice, "return the Device unchanged")
      .def_static("device_id", &FfiTypesHolderObj::DeviceId, "device id of a Device")
      .def_static("apply_fn", &FfiTypesHolderObj::ApplyFn, "call fn(x) and return the result")
      .def_static("make_adder", &FfiTypesHolderObj::MakeAdder, "return a closure adding n")
      .def("shape_ndim", &FfiTypesHolderObj::ShapeNdim, "number of shape dims");

  refl::TypeAttrDef<FfiTypesHolderObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<FfiTypesHolder>);
}

}  // namespace test_ffi_types
