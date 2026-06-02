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
//! Rust side of cpp_rust_test: load a C++ Expr object and mutate its `value` field.

use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::OnceLock;

use tvm_ffi::derive::ObjectRef as DeriveObjectRef;
use tvm_ffi::object::{Object, ObjectArc, ObjectCore};
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIByteArray, TVMFFIObject, TVMFFITypeKeyToIndex,
};
use tvm_ffi::{ensure, AnyView, Function, Module, Result, VALUE_ERROR};

/// Mirrors C++ `cpp_rust_test::ExprObj` layout (`Object` header + `value`).
#[repr(C)]
struct ExprObj {
    object: Object,
    value: i64,
}

unsafe impl ObjectCore for ExprObj {
    const TYPE_KEY: &'static str = "cpp_rust_test.Expr";

    fn type_index() -> i32 {
        static TYPE_INDEX: OnceLock<i32> = OnceLock::new();
        *TYPE_INDEX.get_or_init(|| {
            let type_key = unsafe { TVMFFIByteArray::from_str(Self::TYPE_KEY) };
            let mut tindex = 0;
            let ret = unsafe { TVMFFITypeKeyToIndex(&type_key, &mut tindex) };
            assert_eq!(
                ret, 0,
                "type key `{}` is not registered; load cpp_rust_test_expr.so first",
                Self::TYPE_KEY
            );
            tindex
        })
    }

    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.object)
    }
}

#[derive(DeriveObjectRef, Clone)]
struct Expr {
    data: ObjectArc<ExprObj>,
}

impl Expr {
    fn value(&self) -> i64 {
        self.data.value
    }

    fn set_value(&mut self, value: i64) {
        self.data.value = value;
    }
}

impl Deref for Expr {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.data
    }
}

impl DerefMut for Expr {
    fn deref_mut(&mut self) -> &mut ExprObj {
        &mut self.data
    }
}

fn expr_lib_path() -> &'static str {
    env!("CPP_RUST_TEST_EXPR_LIB")
}

fn make_expr(value: i64) -> Result<Expr> {
    let func = Function::get_global("cpp_rust_test.make_expr")?;
    let ret = func.call_packed(&[AnyView::from(&value)])?;
    ret.try_into()
}

fn main() -> Result<()> {
    let lib_path = expr_lib_path();
    ensure!(
        Path::new(lib_path).exists(),
        VALUE_ERROR,
        "Expr shared library not found at `{}`. Build it first:\n  \
         cmake -S cpp_rust_test -B cpp_rust_test/build && cmake --build cpp_rust_test/build",
        lib_path
    );

    // dlopen runs TVM_FFI_STATIC_INIT_BLOCK and registers cpp_rust_test.Expr.
    let _module = Module::load_from_file(lib_path)?;

    let mut expr = make_expr(42)?;
    println!("created Expr.value = {}", expr.value());

    // Direct memory mutation of C++ object field from Rust.
    expr.set_value(expr.value() + 8);
    println!("after Rust mutation Expr.value = {}", expr.value());

    ensure!(
        expr.value() == 50,
        VALUE_ERROR,
        "expected value 50, got {}",
        expr.value()
    );

    println!("dropping Expr; C++ ~ExprObj() should print value=50 if Rust wrote the same memory");
    drop(expr);
    println!("cpp_rust_test demo OK");
    Ok(())
}
