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
fn e2e_smoke_generate_and_call() {
    let libs = testing_libs().expect("testing libs");
    let gen = run_testing_stubgen("smoke", vec!["testing".to_string()]).expect("stubgen run");

    assert!(gen.out_dir.join("Cargo.toml").exists());
    assert!(gen.functions_rs.contains("add_one"));
    assert!(gen.types_rs.contains("resolve_type_method"));
    assert!(gen.types_rs.contains("pub fn new("));
    assert!(!gen.types_rs.contains("c_ffi_init"));
    assert!(
        !gen.types_rs.contains("Function::get_global(\"testing.TestIntPair.__ffi_init__\")"),
        "type methods should not use global lookup"
    );

    write_integration_test(
        &gen.out_dir,
        &libs.testing_lib,
        SMOKE_INTEGRATION,
        "smoke_integration.rs",
    )
    .expect("write integration");
    run_generated_tests(&gen.out_dir, &libs.lib_dir).expect("generated crate tests");
}

const SMOKE_INTEGRATION: &str = r#"
use tvm_ffi_testing_stub_smoke as stub;

#[test]
fn smoke_add_one_and_pair() {
    stub::load_library(TESTING_LIB).expect("load");
    assert_eq!(stub::add_one(1).expect("add_one"), 2);
    let pair_obj = stub::TestIntPair::new(3, 4).expect("new");
    let pair: stub::TestIntPair = pair_obj.try_into().expect("downcast");
    let sum: i64 = pair.sum(&[]).expect("sum").try_into().expect("i64");
    assert_eq!(sum, 7);
}

#[test]
fn smoke_echo_packed() {
    stub::load_library(TESTING_LIB).expect("load");
    let _ = stub::echo(&[tvm_ffi::Any::from(1_i64)]).expect("echo");
}

#[test]
fn smoke_cxx_inheritance_roundtrip() {
    stub::load_library(TESTING_LIB).expect("load");
    let derived_obj = stub::TestCxxClassDerived::new(11, 7, 3.5, 1.25).expect("new");
    let derived: stub::TestCxxClassDerived = derived_obj.try_into().expect("downcast");
    let base: stub::TestCxxClassBase = derived.clone().into();
    let base_obj: tvm_ffi::object::ObjectRef = base.clone().into();
    let _: stub::TestCxxClassDerived = base_obj.try_into().expect("roundtrip");
}

#[test]
fn smoke_unregistered_object() {
    stub::load_library(TESTING_LIB).expect("load");
    let obj = stub::make_unregistered_object().expect("make");
    let count = stub::object_use_count(obj.clone()).expect("count");
    assert!(count >= 1);
    let _: stub::TestUnregisteredObject = obj.try_into().expect("downcast");
}
"#;
