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
//! Subtype-aware object downcast tests against a real C++ hierarchy
//! `For : Stmt : Object` (`downcast_test.so`, built by `build.rs` under the
//! `example` feature). Covers proposal.md fixes: (a) real-header-index on
//! marshal-out, (b) `is_instance` downcast, (c) `Array::iter()` no silent truncation.
//!
//! This lives *inside* the crate (not `tests/`) because `#[derive(Object)]`
//! expands to `crate::` paths, which only resolve when the derive is compiled as
//! part of the `tvm-ffi` crate itself. Built only for `cargo test --features example`.

use std::ops::Deref;

use crate::object::{is_instance, ObjectArc, ObjectCore};
use crate::{Any, Array, Module, Object, Tensor};

const LIB_PATH: &str = concat!(env!("OUT_DIR"), "/downcast_test.so");

// Rust mirrors of the C++ types, with matching type keys (exactly what stubgen
// emits). The lazy type-key -> index resolution binds them to the C++-registered
// indices once the library is loaded.
#[repr(C)]
#[derive(crate::derive::Object)]
#[type_key = "test_downcast.Stmt"]
struct StmtObj {
    base: Object,
}
#[repr(C)]
#[derive(crate::derive::ObjectRef, Clone)]
struct Stmt {
    data: ObjectArc<StmtObj>,
}

#[repr(C)]
#[derive(crate::derive::Object)]
#[type_key = "test_downcast.For"]
struct ForObj {
    base: StmtObj,
    min: i64,
    extent: i64,
}
#[repr(C)]
#[derive(crate::derive::ObjectRef, Clone)]
struct For {
    data: ObjectArc<ForObj>,
}
impl Deref for For {
    type Target = ForObj;
    fn deref(&self) -> &ForObj {
        &self.data
    }
}

fn lib() -> Module {
    Module::load_from_file(LIB_PATH).expect("load downcast_test.so")
}

/// `make_for` returns a `For` upcast to `Stmt`, as an `Any`.
fn make_for(min: i64, extent: i64) -> Any {
    lib()
        .get_function("make_for")
        .expect("make_for")
        .call_tuple((min, extent))
        .expect("make_for call")
}

#[test]
fn downcast_any_from_cpp() {
    // The `Any` from C++ carries `For`'s REAL header index (C++ stamps it).
    // (b) exact downcast to the real subtype:
    assert!(For::try_from(make_for(0, 8)).is_ok());
    // (b) base accepts the derived value (`For` is-a `Stmt`):
    assert!(Stmt::try_from(make_for(0, 8)).is_ok());
    // (b) a genuine non-ancestor is still rejected:
    assert!(Tensor::try_from(make_for(0, 8)).is_err());
}

#[test]
fn downcast_after_rust_marshal_out() {
    // (a): hold the value at the BASE static type in Rust, marshal it back out,
    // and downcast to the real subtype. `Any::from(stmt)` goes through the Rust
    // `move_to_any`, which must stamp the REAL header index (`For`), not the
    // static `Stmt` index -- otherwise this downcast would fail.
    let stmt: Stmt = Stmt::try_from(make_for(1, 5)).expect("For is-a Stmt"); // (b)
    let any = Any::from(stmt); // (a) Rust marshal-out stamps the real index
    let f: For = For::try_from(any).expect("base-typed value downcasts to its real For");
    // Object pointer survived the round-trip: read the real fields via Deref.
    assert_eq!(f.min, 1);
    assert_eq!(f.extent, 5);
}

#[test]
fn array_of_base_iterates_all_subtypes() {
    // (b)+(c): an `Array<Stmt>` holding `For`s must iterate ALL elements (before
    // the fix, exact-match made every element fail and `iter()` yielded nothing).
    let any = lib()
        .get_function("make_stmt_array")
        .expect("make_stmt_array")
        .call_tuple(())
        .expect("make_stmt_array call");
    let arr: Array<Stmt> = Array::try_from(any).expect("Array<Stmt>");
    assert_eq!(arr.len(), 3);

    let stmts: Vec<Stmt> = arr.iter().collect(); // no silent truncation
    assert_eq!(stmts.len(), 3);

    // Each element downcasts to its real `For`, in order.
    let extents: Vec<i64> = arr
        .iter()
        .map(|s| {
            For::try_from(Any::from(s))
                .expect("element is a For")
                .extent
        })
        .collect();
    assert_eq!(extents, vec![1, 2, 3]);
}

#[test]
fn is_instance_predicate() {
    // Direct check of the runtime predicate once the hierarchy is registered.
    let _ = lib(); // register test_downcast.Stmt / .For
    let stmt = StmtObj::type_index();
    let for_ = ForObj::type_index();
    let object = Object::type_index();
    assert!(is_instance(for_, stmt)); // For is-a Stmt
    assert!(is_instance(for_, object)); // For is-a Object
    assert!(is_instance(for_, for_)); // reflexive
    assert!(!is_instance(stmt, for_)); // Stmt is NOT a For
    let tensor = crate::object::TypeIndex::kTVMFFITensor as i32;
    assert!(!is_instance(stmt, tensor)); // unrelated (sibling under Object)
}
