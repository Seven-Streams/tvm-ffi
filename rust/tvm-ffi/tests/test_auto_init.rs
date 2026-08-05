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
use tvm_ffi::{AnyView, Function, Object, ObjectArc, ObjectCore, Result, String};

// The auto-init fixture is intentionally mutable, so model only its stable
// object prefix. This is the same opaque representation selected by stubgen.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestCxxAutoInitKwOnlyDefaults"]
struct AutoInitKwOnlyDefaultsObj {
    base: Object,
    _not_send_sync: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[repr(transparent)]
#[derive(Clone, ObjectRef)]
struct AutoInitKwOnlyDefaults {
    data: ObjectArc<AutoInitKwOnlyDefaultsObj>,
}

impl Deref for AutoInitKwOnlyDefaults {
    type Target = AutoInitKwOnlyDefaultsObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

fn field(object: &AutoInitKwOnlyDefaults, index: usize) -> Result<i64> {
    get_reflected_field(&**object, index)?.try_into()
}

#[test]
fn auto_init_keyword_protocol_preserves_kw_only_and_defaults() -> Result<()> {
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );

    let init = Function::from_type_method(AutoInitKwOnlyDefaultsObj::type_index(), "__ffi_init__")?;
    let kwargs = Function::get_global("ffi.GetKwargsObject")?.call_packed(&[])?;

    let required_keys = [String::from("p_required"), String::from("k_required")];
    let p_required = 7_i64;
    let k_required = 9_i64;
    let object: AutoInitKwOnlyDefaults = init
        .call_packed(&[
            AnyView::from(&kwargs),
            AnyView::from(&required_keys[0]),
            AnyView::from(&p_required),
            AnyView::from(&required_keys[1]),
            AnyView::from(&k_required),
        ])?
        .try_into()?;

    // Registration order: p_required, p_default, k_required, k_default,
    // hidden. The omitted init=True defaults and init=False hidden field must
    // all be filled by the native auto-init implementation.
    assert_eq!(
        (field(&object, 0)?, field(&object, 1)?, field(&object, 2)?),
        (7, 11, 9)
    );
    assert_eq!((field(&object, 3)?, field(&object, 4)?), (22, 33));

    let all_keys = [
        String::from("p_required"),
        String::from("p_default"),
        String::from("k_required"),
        String::from("k_default"),
    ];
    let values = [1_i64, 2_i64, 3_i64, 4_i64];
    let object: AutoInitKwOnlyDefaults = init
        .call_packed(&[
            AnyView::from(&kwargs),
            AnyView::from(&all_keys[0]),
            AnyView::from(&values[0]),
            AnyView::from(&all_keys[1]),
            AnyView::from(&values[1]),
            AnyView::from(&all_keys[2]),
            AnyView::from(&values[2]),
            AnyView::from(&all_keys[3]),
            AnyView::from(&values[3]),
        ])?
        .try_into()?;
    assert_eq!(
        (
            field(&object, 0)?,
            field(&object, 1)?,
            field(&object, 2)?,
            field(&object, 3)?,
            field(&object, 4)?,
        ),
        (1, 2, 3, 4, 33)
    );
    Ok(())
}
