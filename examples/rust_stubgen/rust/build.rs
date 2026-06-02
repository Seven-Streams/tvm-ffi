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
//! Build script: locate the tvm-ffi runtime + this example's shared library.
//!
//! *Linking* libtvm_ffi (`rustc-link-search` + `rustc-link-lib`) is already
//! emitted by the build script of the `tvm-ffi-sys` dependency and is not
//! repeated here. What this script must still provide:
//!
//! - the loader path for `cargo run`/`cargo test`: cargo applies a build
//!   script's `rustc-env` to the spawned process, but only from the package
//!   being run -- the dependency's identical emission does not reach this
//!   binary (see `update_runtime_library_env`);
//! - the absolute path of the example's C++ library, baked in via
//!   `env!("RUST_STUBGEN_LIB")` and passed to `load_module`.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Name of the C++ shared library (without prefix/suffix).
const LIB_NAME: &str = "rust_stubgen";
/// Env var (exposed to the crate via `env!`) holding the absolute library path.
const LIB_ENV: &str = "RUST_STUBGEN_LIB";

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

fn default_lib_path(manifest_dir: &Path) -> PathBuf {
    // crate dir is <example>/rust ; the library is built into <example>/build by CMake.
    let mut path = manifest_dir.join("..").join("build").join(LIB_NAME);
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => path.set_extension("dll"),
        Ok("macos") => path.set_extension("dylib"),
        _ => path.set_extension("so"),
    };
    path
}

fn tvm_ffi_config_bin(manifest_dir: &Path) -> PathBuf {
    if let Ok(path) = env::var("TVM_FFI_CONFIG") {
        return PathBuf::from(path);
    }
    // repo root is three levels up from <example>/rust. `cfg!(windows)` is the
    // host OS here: build scripts always compile for the host.
    let (bin_dir, bin_name) = if cfg!(windows) {
        ("Scripts", "tvm-ffi-config.exe")
    } else {
        ("bin", "tvm-ffi-config")
    };
    let venv_bin = manifest_dir
        .join("..")
        .join("..")
        .join("..")
        .join(".venv")
        .join(bin_dir)
        .join(bin_name);
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
    if !config_output.status.success() || lib_dir.is_empty() {
        panic!(
            "{} --libdir failed: {}",
            config_bin.display(),
            String::from_utf8_lossy(&config_output.stderr)
        );
    }
    update_runtime_library_env(&lib_dir);

    let lib_path = env::var(LIB_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_lib_path(&manifest_dir));
    println!("cargo:rustc-env={LIB_ENV}={}", lib_path.display());
    println!("cargo:rerun-if-env-changed={LIB_ENV}");
    println!("cargo:rerun-if-env-changed=TVM_FFI_CONFIG");
}
