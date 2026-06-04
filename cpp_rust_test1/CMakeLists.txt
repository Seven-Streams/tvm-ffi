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
project(cpp_rust_test LANGUAGES CXX)

# Prefer the repo-local virtualenv interpreter when present.
set(_repo_venv_python "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
if (EXISTS "${_repo_venv_python}")
  set(Python_EXECUTABLE "${_repo_venv_python}") # cmake-lint: disable=C0103
endif ()

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
find_package(tvm_ffi CONFIG REQUIRED)

add_library(cpp_rust_test_expr SHARED cpp/expr_lib.cc)
tvm_ffi_configure_target(cpp_rust_test_expr STUB_INIT OFF)
set_target_properties(cpp_rust_test_expr PROPERTIES PREFIX "" OUTPUT_NAME "cpp_rust_test_expr")

if (WIN32)
  set_target_properties(cpp_rust_test_expr PROPERTIES SUFFIX ".dll")
elseif (APPLE)
  set_target_properties(cpp_rust_test_expr PROPERTIES SUFFIX ".dylib")
else ()
  set_target_properties(cpp_rust_test_expr PROPERTIES SUFFIX ".so")
endif ()

set(_lib_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(
  cpp_rust_test_expr PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_lib_output_dir}"
                                RUNTIME_OUTPUT_DIRECTORY "${_lib_output_dir}"
)
