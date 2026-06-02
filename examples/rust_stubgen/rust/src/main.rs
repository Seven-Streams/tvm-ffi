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
//! Run the stubgen-generated `IntPair` bindings (see ../../README.md).

mod generated;

use generated::rust_stubgen::IntPair;
use tvm_ffi::{Module, Result};

/// Path of the C++ shared library built by CMake into `../build`.
fn lib_path() -> String {
    let name = if cfg!(target_os = "windows") {
        "rust_stubgen.dll"
    } else if cfg!(target_os = "macos") {
        "librust_stubgen.dylib"
    } else {
        "librust_stubgen.so"
    };
    format!("{}/../build/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn main() -> Result<()> {
    // Load the C++ library so `IntPair` is registered with the FFI type registry.
    // Keep it alive for as long as the bindings are used.
    let _lib = Module::load_from_file(lib_path())?;

    println!("=========== Example 1: construct via ffi_new ===========");
    let mut pair = IntPair::ffi_new(1, 2)?;
    println!("a={}, b={}", pair.a, pair.b);

    println!("=========== Example 2: call a C++ method ===========");
    println!("sum={}", pair.sum()?);

    println!("=========== Example 3: write a field through DerefMut ===========");
    pair.a = 10;
    println!("after pair.a = 10: sum={}", pair.sum()?);

    Ok(())
}
