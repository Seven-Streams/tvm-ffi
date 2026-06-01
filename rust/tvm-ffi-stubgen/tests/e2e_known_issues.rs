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

//! Expected-failure / known-bug regression tests.
//!
//! These tests encode **current** behavior documented in `README.md` (ancestor chain
//! truncation when a direct parent is not repr(C)-mappable). When the bug is fixed,
//! these tests should fail and be updated to assert the corrected behavior.

mod common;

use common::run_testing_stubgen;

/// Positive control: mappable parent chain should emit `impl_object_hierarchy!`.
#[test]
fn known_issue_control_mappable_parent_has_hierarchy() {
    let gen = run_testing_stubgen("ki_control", vec!["testing".to_string()]).expect("stubgen");
    assert!(
        gen.types_rs.contains("impl_object_hierarchy!(TestCxxAutoInitChild:")
            && gen.types_rs.contains("TestCxxAutoInitParent"),
        "mappable parent should appear in hierarchy"
    );
}

/// Known gap: fallback/unregistered subclasses do not get `impl_object_hierarchy!` to their
/// registered base type (no `From` between wrapper types). This passes while the limitation
/// remains; it will fail once hierarchy is wired for fallback types.
#[test]
fn known_issue_unregistered_subclass_no_hierarchy_to_base() {
    let gen = run_testing_stubgen("ki_unreg", vec!["testing".to_string()]).expect("stubgen");
    assert!(gen.types_rs.contains("define_object_wrapper!(TestUnregisteredObject"));
    assert!(
        !gen.types_rs.contains(
            "impl_object_hierarchy!(TestUnregisteredObject: TestUnregisteredBaseObject"
        ),
        "current behavior: no hierarchy macro between unregistered child and base"
    );
}

/// Known gap (README TODO): when direct parent is not repr(C)-mappable but a further ancestor
/// is, `ancestor_chain` may collapse to `ObjectRef` only. We cannot trigger A→B(unmapped)→C in
/// testing without new fixtures; this documents the related fallback path instead.
#[test]
fn known_issue_repr_c_child_with_unmapped_parent_uses_object_parent_field() {
    let gen = run_testing_stubgen("ki_reprc", vec!["testing".to_string()]).expect("stubgen");
    // TestUnregisteredObject may still pass repr_c if metadata exists; if fallback only,
    // define_object_wrapper is used. Either way, we should not emit a bogus hierarchy to base.
    if gen.types_rs.contains("struct TestUnregisteredObjectObj") {
        assert!(
            gen.types_rs.contains("__tvm_ffi_object_parent: tvm_ffi::object::Object"),
            "unmapped parent layout falls back to Object parent in Obj struct"
        );
    } else {
        assert!(gen.types_rs.contains("define_object_wrapper!(TestUnregisteredObject"));
    }
}
