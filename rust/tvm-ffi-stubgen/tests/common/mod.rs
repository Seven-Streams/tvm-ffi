// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tvm_ffi_stubgen::{Args, run};

pub struct TestingLibs {
    pub lib_dir: PathBuf,
    pub dlls: Vec<PathBuf>,
    pub testing_lib: PathBuf,
}

pub struct GeneratedStub {
    pub out_dir: PathBuf,
    pub functions_rs: String,
    pub types_rs: String,
}

pub fn testing_libs() -> Result<TestingLibs, Box<dyn std::error::Error>> {
    let lib_dir = tvm_ffi_libdir()?;
    let dlls = resolve_testing_dlls(&lib_dir).ok_or("unable to locate tvm_ffi testing libraries")?;
    let testing_lib = dlls
        .iter()
        .find(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().contains("tvm_ffi_testing"))
                .unwrap_or(false)
        })
        .cloned()
        .ok_or("tvm_ffi_testing library")?;
    Ok(TestingLibs {
        lib_dir,
        dlls,
        testing_lib,
    })
}

pub fn run_testing_stubgen(tag: &str, init_prefix: Vec<String>) -> Result<GeneratedStub, Box<dyn std::error::Error>> {
    let libs = testing_libs()?;
    let out_dir = unique_temp_dir(&format!("tvm_ffi_stubgen_{}", tag));
    let args = Args {
        out_dir: out_dir.clone(),
        dlls: libs.dlls,
        init_prefix,
        init_crate: format!("tvm_ffi_testing_stub_{}", tag),
        tvm_ffi_path: None,
        overwrite: true,
        no_format: true,
    };
    run(args)?;
    read_generated(&out_dir)
}

pub fn read_generated(out_dir: &Path) -> Result<GeneratedStub, Box<dyn std::error::Error>> {
    let functions_rs = out_dir
        .join("src")
        .join("_tvm_ffi_stubgen_detail")
        .join("functions.rs");
    let types_rs = out_dir
        .join("src")
        .join("_tvm_ffi_stubgen_detail")
        .join("types.rs");
    Ok(GeneratedStub {
        out_dir: out_dir.to_path_buf(),
        functions_rs: fs::read_to_string(functions_rs)?,
        types_rs: fs::read_to_string(types_rs)?,
    })
}

pub fn write_integration_test(
    out_dir: &Path,
    testing_lib: &Path,
    test_source: &str,
    file_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let tests_dir = out_dir.join("tests");
    fs::create_dir_all(&tests_dir)?;
    let body = format!(
        "const TESTING_LIB: &str = {lib_path:?};\n{test_source}",
        lib_path = testing_lib.to_string_lossy()
    );
    fs::write(tests_dir.join(file_name), body)?;
    Ok(())
}

pub fn run_generated_tests(out_dir: &Path, lib_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--manifest-path")
        .arg(out_dir.join("Cargo.toml"))
        .arg("--")
        .arg("--nocapture")
        .current_dir(out_dir);

    let ld_var = if cfg!(target_os = "windows") {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };

    let current_ld = env::var(ld_var).unwrap_or_default();
    let separator = if ld_var == "PATH" { ";" } else { ":" };
    let lib_dir_str = lib_dir.to_string_lossy();
    let new_ld = if current_ld.is_empty() {
        lib_dir_str.to_string()
    } else {
        format!("{}{}{}", lib_dir_str, separator, current_ld)
    };
    cmd.env(ld_var, new_ld);

    if ld_var != "PATH" {
        let path_value = env::var("PATH").unwrap_or_default();
        cmd.env("PATH", path_value);
    }

    let output = cmd.output()?;
    if !output.status.success() {
        eprintln!("generated crate test command failed: {:?}", output.status);
        eprintln!("--- generated test stdout ---");
        eprintln!("{}", String::from_utf8_lossy(&output.stdout));
        eprintln!("--- generated test stderr ---");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return Err("generated crate tests failed".into());
    }
    Ok(())
}

fn resolve_testing_dlls(lib_dir: &Path) -> Option<Vec<PathBuf>> {
    let tvm_ffi = lib_dir.join(lib_filename("tvm_ffi"));
    let tvm_ffi_testing = lib_dir.join(lib_filename("tvm_ffi_testing"));
    if tvm_ffi.exists() && tvm_ffi_testing.exists() {
        Some(vec![tvm_ffi, tvm_ffi_testing])
    } else {
        None
    }
}

fn lib_filename(name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{}.dll", name)
    } else if cfg!(target_os = "macos") {
        format!("lib{}.dylib", name)
    } else {
        format!("lib{}.so", name)
    }
}

pub fn unique_temp_dir(prefix: &str) -> PathBuf {
    let base = env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    base.join(format!("{}_{}_{}", prefix, pid, nanos))
}

pub fn tvm_ffi_libdir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output = Command::new("tvm-ffi-config").arg("--libdir").output()?;
    if !output.status.success() {
        return Err("tvm-ffi-config --libdir failed".into());
    }
    let lib_dir = String::from_utf8(output.stdout)?.trim().to_string();
    if lib_dir.is_empty() {
        return Err("tvm-ffi-config returned empty libdir".into());
    }
    Ok(PathBuf::from(lib_dir))
}
