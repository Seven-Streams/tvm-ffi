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
//! End-to-end tests for `tvm_ffi::optional` against a real C++ reflection object
//! (`optional_test.so`, built by `build.rs` under the `example` feature) whose
//! fields cover all three `ffi::Optional<T>` layout categories.
//!
//! Run with:  cargo test --features example --test test_optional
#![cfg(feature = "example")]

use tvm_ffi::optional::{Align8, Optional};
use tvm_ffi::tvm_ffi_sys::TVMFFIObject;
use tvm_ffi::{Any, Array, Module, String as FfiString};

const LIB_PATH: &str = concat!(env!("OUT_DIR"), "/optional_test.so");

/// `#[repr(C)]` mirror of the C++ `PoCObj` (exactly what stubgen would emit).
#[repr(C)]
#[allow(dead_code)]
struct PoCObjMirror {
    object: TVMFFIObject,                      // C++ Object header
    a: Optional<i64, Align8, 16>,              // Optional<int64_t>     (Cat C)
    b: Optional<*mut TVMFFIObject, Align8, 8>, // Optional<Array<Any>>  (Cat A)
    c: Optional<u8, Align8, 16>,               // Optional<String>      (Cat B)
    tail: i64,
}

fn lib() -> Module {
    Module::load_from_file(LIB_PATH).expect("load optional_test.so")
}

/// Create a `PoCObj` with the chosen fields present.
fn create(a: Option<i64>, b_present: bool, c: Option<&str>) -> Any {
    let f = lib().get_function("poc_create").expect("poc_create");
    f.call_tuple((
        a.is_some() as i64,
        a.unwrap_or(0),
        b_present as i64,
        c.is_some() as i64,
        FfiString::from(c.unwrap_or("")),
    ))
    .expect("poc_create call")
}

/// Reinterpret the object held by `any` as `(&PoCObjMirror, type_index)`.
///
/// SAFETY: `any` must hold a live `poc.PoCObj`; the borrow lives as long as `any`.
unsafe fn as_mirror(any: &mut Any) -> (&PoCObjMirror, i32) {
    let ti = any.type_index();
    let raw = Any::as_data_ptr(any);
    (&*((*raw).data_union.v_obj as *const PoCObjMirror), ti)
}

#[test]
fn layout_matches_cpp() {
    // C++ reports its own offsets/sizes; the Rust mirror must match exactly.
    let cpp = lib()
        .get_function("poc_layout")
        .unwrap()
        .call_tuple(())
        .unwrap()
        .try_as::<FfiString>()
        .unwrap();
    let kv: std::collections::HashMap<&str, usize> = cpp
        .as_str()
        .split_whitespace()
        .map(|tok| {
            let (k, v) = tok.split_once('=').unwrap();
            (k, v.parse::<usize>().unwrap())
        })
        .collect();

    assert_eq!(kv["sizeof"], std::mem::size_of::<PoCObjMirror>());
    assert_eq!(kv["off_a"], std::mem::offset_of!(PoCObjMirror, a));
    assert_eq!(kv["off_b"], std::mem::offset_of!(PoCObjMirror, b));
    assert_eq!(kv["off_c"], std::mem::offset_of!(PoCObjMirror, c));
    assert_eq!(kv["off_tail"], std::mem::offset_of!(PoCObjMirror, tail));
    // category sizes
    assert_eq!(kv["sz_a"], std::mem::size_of::<Optional<i64, Align8, 16>>());
    assert_eq!(
        kv["sz_b"],
        std::mem::size_of::<Optional<*mut TVMFFIObject, Align8, 8>>()
    );
    assert_eq!(kv["sz_c"], std::mem::size_of::<Optional<u8, Align8, 16>>());
}

#[test]
fn read_some_all_categories() {
    let mut obj = create(Some(99), true, Some("hello, this is a long heap string"));
    let (m, ti) = unsafe { as_mirror(&mut obj) };

    // The field address comes from the *Rust mirror's* offset (m.a/m.b/m.c) — a
    // wrong layout would read garbage. The getter is resolved by field name.
    assert_eq!(unsafe { m.a.read::<i64>(ti, "a") }.unwrap(), Some(99)); // Cat C
    assert_eq!(
        unsafe { m.c.read::<FfiString>(ti, "c") }
            .unwrap()
            .map(|s| s.as_str().to_string()),
        Some("hello, this is a long heap string".to_string())
    ); // Cat B
    let arr = unsafe { m.b.read::<Array<i64>>(ti, "b") }.unwrap().unwrap(); // Cat A
    assert_eq!(arr.iter().collect::<Vec<_>>(), vec![10, 20, 30]);
}

#[test]
fn read_none_all_categories() {
    let mut obj = create(None, false, None);
    let (m, ti) = unsafe { as_mirror(&mut obj) };
    assert_eq!(unsafe { m.a.read::<i64>(ti, "a") }.unwrap(), None);
    assert_eq!(unsafe { m.c.read::<FfiString>(ti, "c") }.unwrap(), None);
    assert!(unsafe { m.b.read::<Array<i64>>(ti, "b") }
        .unwrap()
        .is_none());
}

#[test]
fn write_roundtrip_all_categories() {
    // start from an all-None object, then set each field via the C++ setter.
    let mut obj = create(None, false, None);
    let (m, ti) = unsafe { as_mirror(&mut obj) };

    unsafe {
        m.a.write::<i64>(ti, "a", Some(-7)).unwrap();
        m.c.write::<FfiString>(ti, "c", Some(FfiString::from("written from rust")))
            .unwrap();
        m.b.write::<Array<i64>>(ti, "b", Some(Array::from_iter([4i64, 5, 6])))
            .unwrap();
    }

    assert_eq!(unsafe { m.a.read::<i64>(ti, "a") }.unwrap(), Some(-7));
    assert_eq!(
        unsafe { m.c.read::<FfiString>(ti, "c") }
            .unwrap()
            .map(|s| s.as_str().to_string()),
        Some("written from rust".to_string())
    );
    assert_eq!(
        unsafe { m.b.read::<Array<i64>>(ti, "b") }
            .unwrap()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![4, 5, 6]
    );

    // writing None clears the optional.
    unsafe { m.b.write::<Array<i64>>(ti, "b", None) }.unwrap();
    assert!(unsafe { m.b.read::<Array<i64>>(ti, "b") }
        .unwrap()
        .is_none());
}

#[test]
fn read_does_not_leak_refcount() {
    let mut obj = create(None, true, None);
    let (m, ti) = unsafe { as_mirror(&mut obj) };

    // Each read hands back an owned +1 (move-path contract) and drops cleanly.
    for _ in 0..1000 {
        let got = unsafe { m.b.read::<Array<i64>>(ti, "b") }.unwrap().unwrap();
        assert_eq!(got.len(), 3);
    }
    // Baseline: the field holds 1 ref; our extracted Array holds 1 => exactly 2.
    let arr = unsafe { m.b.read::<Array<i64>>(ti, "b") }.unwrap().unwrap();
    assert_eq!(Any::from(arr).debug_strong_count(), Some(2));
}

#[test]
fn resolve_field_reports_registry_layout() {
    let mut obj = create(Some(1), true, Some("x"));
    let (_m, ti) = unsafe { as_mirror(&mut obj) };
    let fa = tvm_ffi::resolve_field(ti, "a").unwrap();
    assert_eq!(fa.offset as usize, std::mem::offset_of!(PoCObjMirror, a));
    assert_eq!(fa.size, 16);
    assert_eq!(fa.alignment, 8);
    assert!(fa.writable);
    assert!(fa.setter.is_some());
}
