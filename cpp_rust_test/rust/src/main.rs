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
//! Rust side of cpp_rust_test: Expr / Add types backed by the same C++ heap objects.

use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::OnceLock;

use tvm_ffi::derive::ObjectRef as DeriveObjectRef;
use tvm_ffi::object::{Object, ObjectArc, ObjectCore};
use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFIObject, TVMFFITypeKeyToIndex};
use tvm_ffi::{ensure, into_typed_fn, AnyView, Function, Module, Result, VALUE_ERROR};

fn lookup_type_index(type_key: &'static str) -> i32 {
    static EXPR_INDEX: OnceLock<i32> = OnceLock::new();
    static ADD_INDEX: OnceLock<i32> = OnceLock::new();
    let cache = match type_key {
        "cpp_rust_test.Expr" => &EXPR_INDEX,
        "cpp_rust_test.Add" => &ADD_INDEX,
        _ => panic!("unknown type key `{type_key}`"),
    };
    *cache.get_or_init(|| {
        let type_key_arg = unsafe { TVMFFIByteArray::from_str(type_key) };
        let mut tindex = 0;
        let ret = unsafe { TVMFFITypeKeyToIndex(&type_key_arg, &mut tindex) };
        assert_eq!(
            ret, 0,
            "type key `{type_key}` is not registered; load cpp_rust_test_expr.so first"
        );
        tindex
    })
}

/// Mirrors C++ `cpp_rust_test::ExprObj` layout (`Object` header + `value`).
#[repr(C)]
struct ExprObj {
    object: Object,
    value: i64,
}

unsafe impl ObjectCore for ExprObj {
    const TYPE_KEY: &'static str = "cpp_rust_test.Expr";

    fn type_index() -> i32 {
        lookup_type_index(Self::TYPE_KEY)
    }

    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.object)
    }
}

#[repr(C)]
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

/// Mirrors C++ `cpp_rust_test::AddObj` (`Object` + `Expr a` + `Expr b` + `value`).
#[repr(C)]
struct AddObj {
    object: Object,
    a: Expr,
    b: Expr,
    value: i64,
}

unsafe impl ObjectCore for AddObj {
    const TYPE_KEY: &'static str = "cpp_rust_test.Add";

    fn type_index() -> i32 {
        lookup_type_index(Self::TYPE_KEY)
    }

    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.object)
    }
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct Add {
    data: ObjectArc<AddObj>,
}

impl Add {
    fn value(&self) -> i64 {
        self.data.value
    }

    fn set_value(&mut self, value: i64) {
        self.data.value = value;
    }

    fn a(&self) -> &Expr {
        &self.data.a
    }

    fn b(&self) -> &Expr {
        &self.data.b
    }

    fn a_mut(&mut self) -> &mut Expr {
        &mut self.data.a
    }
}

impl Deref for Add {
    type Target = AddObj;
    fn deref(&self) -> &AddObj {
        &self.data
    }
}

impl DerefMut for Add {
    fn deref_mut(&mut self) -> &mut AddObj {
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

fn make_add(a: Expr, b: Expr, value: i64) -> Result<Add> {
    let func = Function::get_global("cpp_rust_test.make_add")?;
    let make = into_typed_fn!(func, Fn(Expr, Expr, i64) -> Result<Add>);
    make(a, b, value)
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

    let _module = Module::load_from_file(lib_path)?;

    // --- Expr demo ---
    let mut expr = make_expr(42)?;
    println!("created Expr.value = {}", expr.value());
    expr.set_value(expr.value() + 8);
    println!("after Rust mutation Expr.value = {}", expr.value());
    ensure!(expr.value() == 50, VALUE_ERROR, "expected 50, got {}", expr.value());
    println!("dropping Expr; expect ~ExprObj() value=50");
    drop(expr);

    // --- Add demo (nested Expr fields share C++ heap with Rust views) ---
    let a = make_expr(10)?;
    let b = make_expr(32)?;
    let mut add = make_add(a, b, 0)?;
    println!(
        "created Add: a={}, b={}, value={}",
        add.a().value(),
        add.b().value(),
        add.value()
    );

    add.set_value(add.a().value() + add.b().value());
    println!("after Rust sets Add.value = {}", add.value());
    ensure!(add.value() == 42, VALUE_ERROR, "expected 42, got {}", add.value());

    add.a_mut().set_value(100);
    println!(
        "after Rust mutates Add.a: a={}, b={}, value={}",
        add.a().value(),
        add.b().value(),
        add.value()
    );
    ensure!(
        add.a().value() == 100,
        VALUE_ERROR,
        "expected a=100, got {}",
        add.a().value()
    );

    println!("dropping Add; ~AddObj() then ~ExprObj for a and b");
    drop(add);

    println!("cpp_rust_test demo OK");
    Ok(())
}
