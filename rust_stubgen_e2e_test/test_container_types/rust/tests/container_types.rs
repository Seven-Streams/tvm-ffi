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
//! End-to-end tests for the generated `test_container_types` bindings.

use test_container_types::ensure_loaded;
use test_container_types::generated::test_container_types::{ArrayHolder, OptionalHolder};
use tvm_ffi::{Array, Result, String as FFIString};

#[test]
fn array_static_methods() -> Result<()> {
    ensure_loaded();
    let ints: Array<i64> = Array::new(vec![1, 2, 3, 4]);
    assert_eq!(ArrayHolder::sum_array(ints)?, 10);

    let floats: Array<f64> = Array::new(vec![1.0, 2.0, 3.0]);
    assert_eq!(ArrayHolder::avg_array(floats)?, 2.0);

    let empty: Array<f64> = Array::new(vec![]);
    assert_eq!(ArrayHolder::avg_array(empty)?, 0.0);
    Ok(())
}

#[test]
fn array_construct_and_methods() -> Result<()> {
    ensure_loaded();
    let int_array: Array<i64> = Array::new(vec![10, 20, 30]);
    let float_array: Array<f64> = Array::new(vec![1.5, 2.5]);
    let string_array: Array<FFIString> =
        Array::new(vec![FFIString::from("a"), FFIString::from("b"), FFIString::from("c")]);

    let mut holder = ArrayHolder::new(int_array, float_array, string_array)?;
    assert_eq!(holder.get_int_array_length()?, 3);
    assert_eq!(holder.join_string_array()?.as_str(), "a,b,c");

    // Read array fields back through Deref.
    assert_eq!(holder.int_array.len(), 3);
    assert_eq!(holder.int_array.get(1)?, 20);
    Ok(())
}

#[test]
fn array_set_arrays_mutation() -> Result<()> {
    ensure_loaded();
    let mut holder = ArrayHolder::new(Array::new(vec![]), Array::new(vec![]), Array::new(vec![]))?;
    holder.set_arrays(
        Array::new(vec![5, 6]),
        Array::new(vec![0.5]),
        Array::new(vec![FFIString::from("x")]),
    )?;
    assert_eq!(holder.get_int_array_length()?, 2);
    assert_eq!(holder.join_string_array()?.as_str(), "x");
    Ok(())
}

#[test]
fn optional_static_methods() -> Result<()> {
    ensure_loaded();
    assert_eq!(OptionalHolder::create_opt_int(55)?, Some(55));
    assert_eq!(OptionalHolder::create_none_int()?, None);
    Ok(())
}

#[test]
fn optional_construct_and_describe() -> Result<()> {
    ensure_loaded();
    let mut holder = OptionalHolder::new(Some(7), Some(FFIString::from("hi")))?;
    assert_eq!(holder.describe_optionals()?.as_str(), "opt_int=7,opt_string=hi");

    let mut none_holder = OptionalHolder::new(None, None)?;
    assert_eq!(
        none_holder.describe_optionals()?.as_str(),
        "opt_int=None,opt_string=None"
    );
    Ok(())
}
