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

cmake_minimum_required(VERSION 3.20)
project(rust_stubgen LANGUAGES CXX)

# Prefer the repo-local virtualenv interpreter when present. Python3_EXECUTABLE matters too: the
# stubgen POST_BUILD step (cmake/Utils/Library.cmake) resolves its interpreter via
# find_package(Python3), and that one must import tvm_ffi.
set(_repo_venv_python "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python")
if (EXISTS "${_repo_venv_python}")
  set(Python_EXECUTABLE "${_repo_venv_python}") # cmake-lint: disable=C0103
  set(Python3_EXECUTABLE "${_repo_venv_python}") # cmake-lint: disable=C0103
endif ()

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT COMMAND_ERROR_IS_FATAL ANY
)
find_package(tvm_ffi CONFIG REQUIRED)

add_library(rust_stubgen SHARED cpp/int_pair.cc)

# Generate the Rust object bindings as a post-build step (STUB_TARGET rust). STUB_INIT ON scaffolds
# a fresh module tree under rust/src/generated. This is the CMake equivalent of the CLI invocation
# in scripts/run_stubgen.sh.
tvm_ffi_configure_target(
  rust_stubgen
  STUB_DIR
  rust/src/generated
  STUB_TARGET
  rust
  STUB_INIT
  ON
  STUB_PREFIX
  "rust_stubgen."
)

set_target_properties(rust_stubgen PROPERTIES PREFIX "" OUTPUT_NAME "rust_stubgen")

if (WIN32)
  set_target_properties(rust_stubgen PROPERTIES SUFFIX ".dll")
elseif (APPLE)
  set_target_properties(rust_stubgen PROPERTIES SUFFIX ".dylib")
else ()
  set_target_properties(rust_stubgen PROPERTIES SUFFIX ".so")
endif ()

set_target_properties(
  rust_stubgen PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
