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
//! End-to-end tests for the generated `test_scalar_types` bindings.

use test_scalar_types::ensure_loaded;
use test_scalar_types::generated::test_scalar_types::ScalarHolder;
use tvm_ffi::{Result, String as FFIString};

#[test]
fn static_scalar_constants() -> Result<()> {
    ensure_loaded();
    assert_eq!(ScalarHolder::get_int_constant()?, 100);
    assert_eq!(ScalarHolder::get_float_constant()?, 3.14);
    assert!(ScalarHolder::get_bool_constant()?);
    assert_eq!(ScalarHolder::get_string_constant()?.as_str(), "hello_world");
    Ok(())
}

#[test]
fn static_format_scalars() -> Result<()> {
    ensure_loaded();
    let formatted =
        ScalarHolder::format_scalars(7, 2.5, true, FFIString::from("hi"))?;
    assert_eq!(formatted.as_str(), "int=7,float=2.5,bool=true,str=hi");
    Ok(())
}

#[test]
fn construct_and_read_fields() -> Result<()> {
    ensure_loaded();
    let holder = ScalarHolder::new(42, 1.5, true, FFIString::from("hello"))?;
    assert_eq!(holder.int_val, 42);
    assert_eq!(holder.float_val, 1.5);
    assert!(holder.bool_val);
    assert_eq!(holder.string_val.as_str(), "hello");
    Ok(())
}

#[test]
fn mutate_fields_from_rust() -> Result<()> {
    ensure_loaded();
    let mut holder = ScalarHolder::new(0, 0.0, false, FFIString::from(""))?;
    holder.int_val = 99;
    holder.float_val = 6.25;
    holder.bool_val = true;
    assert_eq!(holder.int_val, 99);
    assert_eq!(holder.float_val, 6.25);
    assert!(holder.bool_val);
    Ok(())
}

#[test]
fn set_values_through_cpp_method() -> Result<()> {
    ensure_loaded();
    let mut holder = ScalarHolder::new(1, 1.0, false, FFIString::from("a"))?;
    holder.set_values(123, 4.5, true, FFIString::from("updated"))?;
    // The C++ method mutated the same heap object; read it back through Rust.
    assert_eq!(holder.int_val, 123);
    assert_eq!(holder.float_val, 4.5);
    assert!(holder.bool_val);
    assert_eq!(holder.string_val.as_str(), "updated");

    let desc = holder.get_description()?;
    assert_eq!(
        desc.as_str(),
        "ScalarHolder[int=123,float=4.5,bool=true,str=updated]"
    );
    Ok(())
}
