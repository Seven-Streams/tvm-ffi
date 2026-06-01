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

use common::{run_generated_tests, run_testing_stubgen, testing_libs, write_integration_test};

#[test]
fn e2e_auto_init_generates_new() {
    let gen = run_testing_stubgen("auto_init", vec!["testing".to_string()]).expect("stubgen");
    assert!(gen.types_rs.contains("impl TestCxxAutoInitSimple"));
    assert!(gen.types_rs.contains("resolve_type_method(\"testing.TestCxxAutoInitSimple\", \"__ffi_init__\")"));
    assert!(gen.types_rs.contains("impl TestCxxAutoInitChild"));
    assert!(gen.types_rs.contains("resolve_type_method(\"testing.TestCxxAutoInitChild\", \"__ffi_init__\")"));
}

#[test]
fn e2e_auto_init_roundtrip() {
    let libs = testing_libs().expect("libs");
    let gen = run_testing_stubgen("auto_init_rt", vec!["testing".to_string()]).expect("stubgen");
    write_integration_test(
        &gen.out_dir,
        &libs.testing_lib,
        AUTO_INIT_INTEGRATION,
        "auto_init_integration.rs",
    )
    .expect("write integration");
    run_generated_tests(&gen.out_dir, &libs.lib_dir).expect("generated tests");
}

const AUTO_INIT_INTEGRATION: &str = r#"
use tvm_ffi_testing_stub_auto_init_rt as stub;

#[test]
fn auto_init_simple_new() {
    stub::load_library(TESTING_LIB).expect("load");
    let obj = stub::TestCxxAutoInitSimple::new(1, 2).expect("new");
    let _v: stub::TestCxxAutoInitSimple = obj.try_into().expect("downcast");
}

#[test]
fn auto_init_child_includes_parent_fields() {
    stub::load_library(TESTING_LIB).expect("load");
    let obj = stub::TestCxxAutoInitChild::new(10, 20, 30).expect("new parent+child fields");
    let _v: stub::TestCxxAutoInitChild = obj.try_into().expect("downcast");
}
"#;
