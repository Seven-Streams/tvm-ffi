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

use std::ops::Deref;

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{AnyView, Function, Object, ObjectArc, ObjectCore, String};

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.SchemaAllTypes"]
struct SchemaAllTypesObj {
    base: Object,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct SchemaAllTypes {
    data: ObjectArc<SchemaAllTypesObj>,
}

impl Deref for SchemaAllTypes {
    type Target = SchemaAllTypesObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestCxxAutoInitParent"]
struct AutoInitParentObj {
    base: Object,
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestCxxAutoInitChild"]
struct AutoInitChildObj {
    base: AutoInitParentObj,
}

impl Deref for AutoInitChildObj {
    type Target = AutoInitParentObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct AutoInitChild {
    data: ObjectArc<AutoInitChildObj>,
}

impl Deref for AutoInitChild {
    type Target = AutoInitChildObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[test]
fn reflected_getter_reads_opaque_cpp_object_and_owns_result() {
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );

    let make = Function::from_type_method(SchemaAllTypesObj::type_index(), "make_with").unwrap();
    let object: SchemaAllTypes = make
        .call_packed(&[
            AnyView::from(&7_i64),
            AnyView::from(&2.5_f64),
            AnyView::from(&String::from("owned value")),
        ])
        .unwrap()
        .try_into()
        .unwrap();

    assert_eq!(
        std::mem::size_of::<SchemaAllTypesObj>(),
        std::mem::size_of::<Object>()
    );
    let value: i64 = unsafe { tvm_ffi::object::get_reflected_field_unchecked(&*object, "v_int") }
        .unwrap()
        .try_into()
        .unwrap();
    let text: String =
        unsafe { tvm_ffi::object::get_reflected_field_unchecked(&*object, "v_string") }
            .unwrap()
            .try_into()
            .unwrap();

    assert_eq!(value, 7);
    assert_eq!(AnyView::from(&text).debug_strong_count(), Some(2));
    drop(object);
    assert_eq!(AnyView::from(&text).debug_strong_count(), Some(1));
    assert_eq!(text.as_str(), "owned value");
}

#[test]
fn type_attr_constructor_builds_real_cpp_object() {
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );

    static INIT: std::sync::OnceLock<Function> = std::sync::OnceLock::new();
    let init =
        Function::from_type_attr_cached(&INIT, AutoInitChildObj::type_index(), "__ffi_init__")
            .unwrap();
    let parent_required = 11_i64;
    let parent_default = 13_i64;
    let child_required = 17_i64;
    let child_kw_only = 19_i64;
    let object: AutoInitChild = init
        .call_packed_with_kwargs(
            &[],
            &[
                ("parent_required", AnyView::from(&parent_required)),
                ("parent_default", AnyView::from(&parent_default)),
                ("child_required", AnyView::from(&child_required)),
                ("child_kw_only", AnyView::from(&child_kw_only)),
            ],
        )
        .unwrap()
        .try_into()
        .unwrap();

    let parent_required: i64 =
        unsafe { tvm_ffi::object::get_reflected_field_unchecked(&**object, "parent_required") }
            .unwrap()
            .try_into()
            .unwrap();
    let parent_default: i64 =
        unsafe { tvm_ffi::object::get_reflected_field_unchecked(&**object, "parent_default") }
            .unwrap()
            .try_into()
            .unwrap();
    let child_required: i64 =
        unsafe { tvm_ffi::object::get_reflected_field_unchecked(&*object, "child_required") }
            .unwrap()
            .try_into()
            .unwrap();
    let child_kw_only: i64 =
        unsafe { tvm_ffi::object::get_reflected_field_unchecked(&*object, "child_kw_only") }
            .unwrap()
            .try_into()
            .unwrap();
    assert_eq!(
        (
            parent_required,
            parent_default,
            child_required,
            child_kw_only
        ),
        (11, 13, 17, 19)
    );
}
