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
fn e2e_inheritance_impl_object_hierarchy() {
    let gen = run_testing_stubgen("inherit", vec!["testing".to_string()]).expect("stubgen");
    assert!(
        gen.types_rs.contains(
            "impl_object_hierarchy!(TestCxxClassDerivedDerived: TestCxxClassDerived"
        ) || gen.types_rs.contains(
            "impl_object_hierarchy!(TestCxxClassDerivedDerived: crate::_tvm_ffi_stubgen_detail::types::TestCxxClassDerived"
        ),
        "three-level cxx hierarchy should list direct parent"
    );
    assert!(
        gen.types_rs.contains("impl_object_hierarchy!(TestCxxAutoInitChild: TestCxxAutoInitParent")
            || gen.types_rs
                .contains("TestCxxAutoInitParent"),
        "auto-init child should link to parent in hierarchy"
    );
}

#[test]
fn e2e_inheritance_derived_derived_roundtrip() {
    let libs = testing_libs().expect("libs");
    let gen = run_testing_stubgen("inherit_rt", vec!["testing".to_string()]).expect("stubgen");
    write_integration_test(
        &gen.out_dir,
        &libs.testing_lib,
        INHERIT_INTEGRATION,
        "inherit_integration.rs",
    )
    .expect("write integration");
    run_generated_tests(&gen.out_dir, &libs.lib_dir).expect("generated tests");
}

const INHERIT_INTEGRATION: &str = r#"
use tvm_ffi_testing_stub_inherit_rt as stub;

#[test]
fn cxx_derived_upcast_to_base() {
    stub::load_library(TESTING_LIB).expect("load");
    let obj = stub::TestCxxClassDerived::new(1, 2, 3.0, 4.0).expect("new");
    let derived: stub::TestCxxClassDerived = obj.try_into().expect("downcast");
    let base: stub::TestCxxClassBase = derived.into();
    assert_eq!(base.v_i64().expect("v_i64"), 1);
}
"#;
