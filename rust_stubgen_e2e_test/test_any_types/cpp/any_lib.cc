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
 * \file any_lib.cc
 * \brief C++ types testing `Any` in parameter / return / field positions.
 *
 * Per the FFI convention (docs/concepts/any.rst): a top-level `Any` in argument
 * position is passed as the non-owning `AnyView`, while a returned `Any` is
 * owning. The Rust stubgen renders these accordingly (param -> `AnyView`,
 * return/field -> `Any`); this module exercises both directions.
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace test_any_types {

namespace ffi = tvm::ffi;

class AnyHolderObj : public ffi::Object {
 public:
  ffi::Any stored;

  explicit AnyHolderObj(ffi::AnyView stored = ffi::AnyView()) : stored(stored) {}

  // `Any` is opaque: it carries an arbitrary payload across the FFI boundary
  // unchanged. `Echo` returns its input verbatim (H1 param + H2 return) so a
  // test can push several underlying types through and assert round-trip
  // identity -- without C++ ever inspecting what's inside.
  //
  // Per docs/concepts/any.rst, function *parameters* take the non-owning
  // `AnyView` (no refcount / copy overhead), while *return values* are the
  // owning `Any` (transfers ownership to the caller).
  static ffi::Any Echo(ffi::AnyView v) { return v; }

  // Instance method writing the `Any` field, plus an `Any`-returning getter.
  void SetAny(ffi::AnyView v) { stored = v; }

  ffi::Any GetAny() { return stored; }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test_any_types.AnyHolder", AnyHolderObj, ffi::Object);
};

class AnyHolder : public ffi::ObjectRef {
 public:
  explicit AnyHolder(ffi::AnyView stored = ffi::AnyView()) {
    data_ = ffi::make_object<AnyHolderObj>(stored);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AnyHolder, ffi::ObjectRef, AnyHolderObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<AnyHolderObj>()
      .def(refl::init<ffi::AnyView>())
      .def_rw("stored", &AnyHolderObj::stored, "stored Any value")
      .def_static("echo", &AnyHolderObj::Echo, "return the Any unchanged")
      .def("set_any", &AnyHolderObj::SetAny, "store an Any")
      .def("get_any", &AnyHolderObj::GetAny, "retrieve the stored Any");

  refl::TypeAttrDef<AnyHolderObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<AnyHolder>);
}

}  // namespace test_any_types
