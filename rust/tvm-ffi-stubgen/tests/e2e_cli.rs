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

mod common;

use common::{run_testing_stubgen, unique_temp_dir};
use std::fs;
use tvm_ffi_stubgen::{Args, run};

#[test]
fn e2e_cli_rejects_nonempty_out_dir_without_overwrite() {
    let libs = common::testing_libs().expect("libs");
    let out_dir = unique_temp_dir("stubgen_cli_no_overwrite");
    fs::create_dir_all(&out_dir).unwrap();
    fs::write(out_dir.join("existing.txt"), "x").unwrap();

    let args = Args {
        out_dir: out_dir.clone(),
        dlls: libs.dlls,
        init_prefix: vec!["testing".to_string()],
        init_crate: "tvm_ffi_testing_stub_cli".to_string(),
        tvm_ffi_path: None,
        overwrite: false,
        no_format: true,
    };
    let err = run(args).unwrap_err();
    assert!(err.to_string().contains("not empty"));
    let _ = fs::remove_dir_all(&out_dir);
}

#[test]
fn e2e_prefix_includes_testing_globals() {
    let gen = run_testing_stubgen("prefix", vec!["testing".to_string()]).expect("stubgen");
    assert!(gen.functions_rs.contains("add_one"));
    assert!(gen.functions_rs.contains("schema_id_int"));
}
