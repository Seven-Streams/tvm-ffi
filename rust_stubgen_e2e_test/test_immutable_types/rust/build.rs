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
//! Build script: locate the tvm-ffi runtime + this test's generated shared library.
use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Name of this test point's C++ shared library (without prefix/suffix).
const LIB_NAME: &str = "test_immutable_types";
/// Env var (exposed to the crate via `env!`) holding the absolute .so path.
const LIB_ENV: &str = "TEST_IMMUTABLE_TYPES_LIB";

fn update_runtime_library_env(lib_dir: &str) {
    let os_env_var = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "PATH",
        Ok("macos") => "DYLD_LIBRARY_PATH",
        Ok("linux") => "LD_LIBRARY_PATH",
        _ => return,
    };
    let current_val = env::var(os_env_var).unwrap_or_default();
    let separator = if os_env_var == "PATH" { ";" } else { ":" };
    let new_val = if current_val.is_empty() {
        lib_dir.to_string()
    } else {
        format!("{current_val}{separator}{lib_dir}")
    };
    println!("cargo:rustc-env={os_env_var}={new_val}");
}

fn default_lib_path(manifest_dir: &PathBuf) -> PathBuf {
    // crate dir is <test>/rust ; the .so is built into <test>/build by CMake.
    let mut path = manifest_dir.join("..").join("build").join(LIB_NAME);
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => path.set_extension("dll"),
        Ok("macos") => path.set_extension("dylib"),
        _ => path.set_extension("so"),
    };
    path
}

fn tvm_ffi_config_bin(manifest_dir: &PathBuf) -> PathBuf {
    if let Ok(path) = env::var("TVM_FFI_CONFIG") {
        return PathBuf::from(path);
    }
    // repo root is three levels up from <test>/rust.
    let venv_bin = manifest_dir
        .join("..")
        .join("..")
        .join("..")
        .join(".venv")
        .join("bin")
        .join("tvm-ffi-config");
    if venv_bin.is_file() {
        return venv_bin;
    }
    PathBuf::from("tvm-ffi-config")
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    let config_bin = tvm_ffi_config_bin(&manifest_dir);

    let config_output = Command::new(&config_bin)
        .arg("--libdir")
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to run {}: {e} (install tvm-ffi: uv pip install -e . from repo root)",
                config_bin.display()
            )
        });
    let lib_dir = String::from_utf8(config_output.stdout)
        .expect("invalid UTF-8 from tvm-ffi-config")
        .trim()
        .to_string();
    update_runtime_library_env(&lib_dir);
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=dylib=tvm_ffi");

    let lib_path = env::var(LIB_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_lib_path(&manifest_dir));
    println!("cargo:rustc-env={LIB_ENV}={}", lib_path.display());
    println!("cargo:rerun-if-env-changed={LIB_ENV}");
    println!("cargo:rerun-if-env-changed=TVM_FFI_CONFIG");
}
