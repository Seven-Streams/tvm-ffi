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
fn e2e_schema_typed_globals() {
    let gen = run_testing_stubgen("schema", vec!["testing".to_string()]).expect("stubgen");
    assert!(
        gen.functions_rs.contains("pub fn schema_id_int(") && gen.functions_rs.contains("_0: i64"),
        "schema_id_int should be typed"
    );
    assert!(
        gen.functions_rs.contains("schema_id_opt_int(")
            && gen.functions_rs.contains("Option<i64>"),
        "optional schema global"
    );
    assert!(
        gen.functions_rs.contains("schema_id_arr_int(")
            && gen.functions_rs.contains("tvm_ffi::Array"),
        "array schema global"
    );
    assert!(
        gen.functions_rs.contains("schema_id_map_str_int(")
            && gen.functions_rs.contains("tvm_ffi::Map"),
        "map schema global"
    );
}

#[test]
fn e2e_schema_packed_globals() {
    let gen = run_testing_stubgen("schema_packed", vec!["testing".to_string()]).expect("stubgen");
    assert!(
        gen.functions_rs.contains("pub fn schema_packed(args: &[Any])"),
        "packed global uses Any slice"
    );
    assert!(
        gen.functions_rs.contains("pub fn nop(args: &[Any])")
            || gen.functions_rs.contains("pub fn nop("),
        "nop should be generated"
    );
}

#[test]
fn e2e_schema_all_types_object() {
    let gen = run_testing_stubgen("schema_obj", vec!["testing".to_string()]).expect("stubgen");
    assert!(
        gen.types_rs.contains("SchemaAllTypes"),
        "SchemaAllTypes wrapper should be generated"
    );
    assert!(
        gen.types_rs.contains("resolve_type_method(\"testing.SchemaAllTypes\""),
        "object methods use type metadata"
    );
}

#[test]
fn e2e_schema_roundtrip() {
    let libs = testing_libs().expect("libs");
    let gen = run_testing_stubgen("schema_rt", vec!["testing".to_string()]).expect("stubgen");
    write_integration_test(
        &gen.out_dir,
        &libs.testing_lib,
        SCHEMA_INTEGRATION,
        "schema_integration.rs",
    )
    .expect("write integration");
    run_generated_tests(&gen.out_dir, &libs.lib_dir).expect("generated tests");
}

const SCHEMA_INTEGRATION: &str = r#"
use tvm_ffi_testing_stub_schema_rt as stub;

#[test]
fn schema_globals_roundtrip() {
    stub::load_library(TESTING_LIB).expect("load");
    assert_eq!(stub::schema_id_int(41).expect("int"), 41);
    assert_eq!(
        stub::schema_id_opt_int(Some(7)).expect("opt"),
        Some(7)
    );
}
"#;
