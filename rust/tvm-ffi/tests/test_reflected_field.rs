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
use tvm_ffi::object::get_reflected_field;
use tvm_ffi::{AnyView, Function, Object, ObjectArc, ObjectCore};

// Deliberately model only the common object header. This is the shape emitted
// for a foreign type when stubgen cannot prove its complete native layout.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestIntPair"]
struct OpaquePairObj {
    base: Object,
}

#[repr(transparent)]
#[derive(Clone, ObjectRef)]
struct OpaquePair {
    data: ObjectArc<OpaquePairObj>,
}

impl Deref for OpaquePair {
    type Target = OpaquePairObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[test]
fn reflected_getter_reads_fields_from_an_opaque_foreign_layout() {
    // Force the testing shared object into this binary before consulting its
    // static reflection registrations.
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    let init = Function::from_type_method(OpaquePairObj::type_index(), "__ffi_init__").unwrap();
    let a = 3_i64;
    let b = 4_i64;
    let pair: OpaquePair = init
        .call_packed(&[AnyView::from(&a), AnyView::from(&b)])
        .unwrap()
        .try_into()
        .unwrap();

    let actual_a: i64 = get_reflected_field(&*pair, 0).unwrap().try_into().unwrap();
    let actual_b: i64 = get_reflected_field(&*pair, 1).unwrap().try_into().unwrap();
    assert_eq!((actual_a, actual_b), (3, 4));
    assert!(get_reflected_field(&*pair, 2).is_err());
}
