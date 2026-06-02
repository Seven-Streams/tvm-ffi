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
//! Rust bindings for the `rust_stubgen` C++ example library.
//!
//! The `generated` module is produced by `tvm-ffi-stubgen --target rust` (see
//! `scripts/run_stubgen.sh`). `ensure_loaded` dlopens the C++ shared library
//! once so the reflection registry is populated before any binding is used.

pub mod generated;

use std::sync::OnceLock;

/// Absolute path to the C++ shared library, baked in by `build.rs`.
pub fn lib_path() -> &'static str {
    env!("RUST_STUBGEN_LIB")
}

/// Load the C++ shared library exactly once for the whole process.
///
/// The loaded `Module` is intentionally leaked: keeping the library mapped for
/// the lifetime of the process keeps the FFI type/method registry populated and
/// makes repeated calls cheap and safe across parallel tests.
pub fn ensure_loaded() {
    static LOADED: OnceLock<()> = OnceLock::new();
    LOADED.get_or_init(|| {
        let path = lib_path();
        assert!(
            std::path::Path::new(path).exists(),
            "shared library not found at `{path}`; build it first with `cmake --build examples/rust_stubgen/build`"
        );
        let module = tvm_ffi::Module::load_from_file(path)
            .unwrap_or_else(|e| panic!("failed to load `{path}`: {e}"));
        std::mem::forget(module);
    });
}
