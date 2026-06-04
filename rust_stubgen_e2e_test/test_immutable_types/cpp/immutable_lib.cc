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
 * \file immutable_lib.cc
 * \brief C++ types with read-only fields (immutable types).
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <iostream>
#include <sstream>

namespace test_immutable_types {

namespace ffi = tvm::ffi;

class ImmutableVersionObj : public ffi::Object {
 public:
  int64_t major;
  int64_t minor;
  int64_t patch;
  ffi::String label;

  explicit ImmutableVersionObj(int64_t major = 0, int64_t minor = 0, int64_t patch = 0,
                               ffi::String label = "")
      : major(major), minor(minor), patch(patch), label(label) {}

  static int64_t GetCurrentMajor() { return 2; }

  static int64_t GetCurrentMinor() { return 1; }

  static int64_t GetCurrentPatch() { return 0; }

  ffi::String GetVersionString() {
    std::ostringstream oss;
    oss << major << "." << minor << "." << patch << (label.size() > 0 ? "-" : "") << label.c_str();
    return ffi::String(oss.str());
  }

  bool IsGreaterThanVersion(int64_t other_major, int64_t other_minor, int64_t other_patch) {
    if (major != other_major) return major > other_major;
    if (minor != other_minor) return minor > other_minor;
    return patch > other_patch;
  }

  ~ImmutableVersionObj() {
    std::cout << "[test_immutable_types] ~ImmutableVersionObj() version=" << major << "." << minor
              << "." << patch << std::endl;
  }

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO("test_immutable_types.ImmutableVersion", ImmutableVersionObj,
                              ffi::Object);
};

class ImmutableVersion : public ffi::ObjectRef {
 public:
  explicit ImmutableVersion(int64_t major = 0, int64_t minor = 0, int64_t patch = 0,
                            ffi::String label = "") {
    data_ = ffi::make_object<ImmutableVersionObj>(major, minor, patch, label);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ImmutableVersion, ffi::ObjectRef,
                                                ImmutableVersionObj);
};

class ImmutableMetadataObj : public ffi::Object {
 public:
  ffi::String name;
  ffi::String author;
  ffi::String license;
  ffi::Array<ffi::String> keywords;

  explicit ImmutableMetadataObj(ffi::String name = "", ffi::String author = "",
                                ffi::String license = "",
                                ffi::Array<ffi::String> keywords = ffi::Array<ffi::String>())
      : name(name), author(author), license(license), keywords(keywords) {}

  static ffi::String GetDefaultLicense() { return ffi::String("Apache-2.0"); }

  ffi::String ToJSON() {
    std::ostringstream oss;
    oss << "{\"name\":\"" << name.c_str() << "\",\"author\":\"" << author.c_str()
        << "\",\"license\":\"" << license.c_str() << "\"}";
    return ffi::String(oss.str());
  }

  int64_t GetKeywordCount() { return keywords.size(); }

  ~ImmutableMetadataObj() {
    std::cout << "[test_immutable_types] ~ImmutableMetadataObj() name=" << name.c_str()
              << std::endl;
  }

  static constexpr bool _type_mutable = false;
  TVM_FFI_DECLARE_OBJECT_INFO("test_immutable_types.ImmutableMetadata", ImmutableMetadataObj,
                              ffi::Object);
};

class ImmutableMetadata : public ffi::ObjectRef {
 public:
  explicit ImmutableMetadata(ffi::String name = "", ffi::String author = "",
                             ffi::String license = "",
                             ffi::Array<ffi::String> keywords = ffi::Array<ffi::String>()) {
    data_ = ffi::make_object<ImmutableMetadataObj>(name, author, license, keywords);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ImmutableMetadata, ffi::ObjectRef,
                                                ImmutableMetadataObj);
};

class MixedMutabilityObj : public ffi::Object {
 public:
  int64_t immutable_id;
  int64_t mutable_counter;

  explicit MixedMutabilityObj(int64_t immutable_id = 0, int64_t mutable_counter = 0)
      : immutable_id(immutable_id), mutable_counter(mutable_counter) {}

  static int64_t GetNextId() {
    static int64_t next_id = 1000;
    return next_id++;
  }

  void IncrementCounter() {
    mutable_counter++;
    std::cout << "[test_immutable_types] MixedMutability::IncrementCounter() now="
              << mutable_counter << std::endl;
  }

  ~MixedMutabilityObj() {
    std::cout << "[test_immutable_types] ~MixedMutabilityObj() id=" << immutable_id
              << " counter=" << mutable_counter << std::endl;
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test_immutable_types.MixedMutability", MixedMutabilityObj,
                              ffi::Object);
};

class MixedMutability : public ffi::ObjectRef {
 public:
  explicit MixedMutability(int64_t immutable_id = 0, int64_t mutable_counter = 0) {
    data_ = ffi::make_object<MixedMutabilityObj>(immutable_id, mutable_counter);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MixedMutability, ffi::ObjectRef,
                                                MixedMutabilityObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<ImmutableVersionObj>()
      .def(refl::init<int64_t, int64_t, int64_t, ffi::String>())
      .def_ro("major", &ImmutableVersionObj::major, "major version")
      .def_ro("minor", &ImmutableVersionObj::minor, "minor version")
      .def_ro("patch", &ImmutableVersionObj::patch, "patch version")
      .def_ro("label", &ImmutableVersionObj::label, "version label")
      .def_static("get_current_major", &ImmutableVersionObj::GetCurrentMajor,
                  "current major version")
      .def_static("get_current_minor", &ImmutableVersionObj::GetCurrentMinor,
                  "current minor version")
      .def_static("get_current_patch", &ImmutableVersionObj::GetCurrentPatch,
                  "current patch version")
      .def("get_version_string", &ImmutableVersionObj::GetVersionString, "get version string")
      .def("is_greater_than_version", &ImmutableVersionObj::IsGreaterThanVersion,
           "compare versions");

  refl::TypeAttrDef<ImmutableVersionObj>().def(
      refl::type_attr::kConvert,
      &refl::details::FFIConvertFromAnyViewToObjectRef<ImmutableVersion>);

  refl::ObjectDef<ImmutableMetadataObj>()
      .def(refl::init<ffi::String, ffi::String, ffi::String, ffi::Array<ffi::String>>())
      .def_ro("name", &ImmutableMetadataObj::name, "package name")
      .def_ro("author", &ImmutableMetadataObj::author, "package author")
      .def_ro("license", &ImmutableMetadataObj::license, "package license")
      .def_ro("keywords", &ImmutableMetadataObj::keywords, "package keywords")
      .def_static("get_default_license", &ImmutableMetadataObj::GetDefaultLicense,
                  "get default license")
      .def("to_json", &ImmutableMetadataObj::ToJSON, "convert to JSON")
      .def("get_keyword_count", &ImmutableMetadataObj::GetKeywordCount, "get keyword count");

  refl::TypeAttrDef<ImmutableMetadataObj>().def(
      refl::type_attr::kConvert,
      &refl::details::FFIConvertFromAnyViewToObjectRef<ImmutableMetadata>);

  refl::ObjectDef<MixedMutabilityObj>()
      .def(refl::init<int64_t, int64_t>())
      .def_ro("immutable_id", &MixedMutabilityObj::immutable_id, "immutable id field (read-only)")
      .def_rw("mutable_counter", &MixedMutabilityObj::mutable_counter,
              "mutable counter field (read-write)")
      .def_static("get_next_id", &MixedMutabilityObj::GetNextId, "get next available id")
      .def("increment_counter", &MixedMutabilityObj::IncrementCounter, "increment counter");

  refl::TypeAttrDef<MixedMutabilityObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<MixedMutability>);
}

}  // namespace test_immutable_types
