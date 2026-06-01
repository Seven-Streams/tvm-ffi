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
fn e2e_unregistered_fallback_wrapper() {
    let gen = run_testing_stubgen("unreg", vec!["testing".to_string()]).expect("stubgen");
    assert!(
        gen.types_rs.contains("define_object_wrapper!(TestUnregisteredObject"),
        "unregistered types use define_object_wrapper"
    );
    assert!(
        gen.types_rs.contains("define_object_wrapper!(TestUnregisteredBaseObject"),
        "unregistered base should also use fallback"
    );
}

#[test]
fn e2e_unregistered_roundtrip() {
    let libs = testing_libs().expect("libs");
    let gen = run_testing_stubgen("unreg_rt", vec!["testing".to_string()]).expect("stubgen");
    write_integration_test(
        &gen.out_dir,
        &libs.testing_lib,
        UNREG_INTEGRATION,
        "unreg_integration.rs",
    )
    .expect("write integration");
    run_generated_tests(&gen.out_dir, &libs.lib_dir).expect("generated tests");
}

const UNREG_INTEGRATION: &str = r#"
use tvm_ffi_testing_stub_unreg_rt as stub;

#[test]
fn unregistered_make_and_downcast() {
    stub::load_library(TESTING_LIB).expect("load");
    let obj = stub::make_unregistered_object().expect("make");
    let _: stub::TestUnregisteredObject = obj.clone().try_into().expect("downcast");
    let count = stub::object_use_count(obj).expect("count");
    assert!(count >= 1);
}
"#;
