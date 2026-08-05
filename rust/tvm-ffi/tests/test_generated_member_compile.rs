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
use tvm_ffi::{Any, AnyView, Object, ObjectArc, Result};

#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "testing.stubgen.GeneratedParent"]
struct GeneratedParentObj {
    base: Object,
}

impl GeneratedParentObj {
    fn generated_member(&self, dynamic: AnyView<'_>) -> Result<Any> {
        // This is the exact self-packing shape emitted by stubgen. In
        // particular, GeneratedParentObj itself is not AnyCompatible.
        let args = [tvm_ffi::object::as_any_view(self), dynamic];
        let _: &[AnyView<'_>] = &args;
        Ok(Any::new())
    }
}

#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "testing.stubgen.GeneratedChild"]
struct GeneratedChildObj {
    base: GeneratedParentObj,
}

impl Deref for GeneratedChildObj {
    type Target = GeneratedParentObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

#[repr(transparent)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
struct GeneratedChild {
    data: ObjectArc<GeneratedChildObj>,
}

impl Deref for GeneratedChild {
    type Target = GeneratedChildObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

fn inherited_member_typechecks(child: &GeneratedChild, dynamic: AnyView<'_>) -> Result<Any> {
    child.generated_member(dynamic)
}

#[test]
fn generated_member_self_packing_and_inheritance_compile() {
    let _: fn(&GeneratedChild, AnyView<'_>) -> Result<Any> = inherited_member_typechecks;
}
