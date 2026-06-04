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
use test_container_types::generated::test_container_types::{ArrayHolder, NestedHolder, OptionalHolder};
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

// --- E1: nested containers ----------------------------------------------------

#[test]
fn nested_array_param_and_return() -> Result<()> {
    ensure_loaded();
    // Array<Array<i64>> as a parameter.
    let matrix: Array<Array<i64>> =
        Array::new(vec![Array::new(vec![1, 2, 3]), Array::new(vec![4, 5])]);
    assert_eq!(NestedHolder::sum_matrix(matrix)?, 15);

    // Array<Array<i64>> as a return value (3 copies of the row).
    let grid = NestedHolder::replicate(Array::new(vec![7, 8]), 3)?;
    assert_eq!(grid.len(), 3);
    assert_eq!(grid.get(0)?.get(1)?, 8);
    assert_eq!(grid.get(2)?.get(0)?, 7);
    Ok(())
}

#[test]
fn array_of_optionals_param() -> Result<()> {
    ensure_loaded();
    // Array<Optional<i64>> as a parameter; only Some elements are counted.
    let arr: Array<Option<i64>> = Array::new(vec![Some(1), None, Some(3), None, Some(5)]);
    assert_eq!(NestedHolder::count_some(arr)?, 3);
    Ok(())
}

#[test]
fn nested_container_fields() -> Result<()> {
    ensure_loaded();
    let matrix: Array<Array<i64>> = Array::new(vec![Array::new(vec![1, 2]), Array::new(vec![3])]);
    let opt_ints: Array<Option<i64>> = Array::new(vec![Some(9), None]);
    let opt_strs: Option<Array<FFIString>> =
        Some(Array::new(vec![FFIString::from("a"), FFIString::from("b")]));

    let holder = NestedHolder::new(matrix, opt_ints, opt_strs)?;
    // Read nested container fields through Deref.
    assert_eq!(holder.matrix.len(), 2);
    assert_eq!(holder.matrix.get(0)?.get(1)?, 2);
    assert_eq!(holder.matrix.get(1)?.get(0)?, 3);
    assert_eq!(holder.opt_ints.len(), 2);
    assert_eq!(holder.opt_ints.get(0)?, Some(9));
    assert_eq!(holder.opt_ints.get(1)?, None);
    Ok(())
}

#[test]
fn optional_array_field_some_and_none() -> Result<()> {
    ensure_loaded();
    let mut with = NestedHolder::new(
        Array::new(vec![]),
        Array::new(vec![]),
        Some(Array::new(vec![FFIString::from("x"), FFIString::from("y"), FFIString::from("z")])),
    )?;
    assert_eq!(with.opt_strs_len()?, 3);

    let mut without = NestedHolder::new(Array::new(vec![]), Array::new(vec![]), None)?;
    assert_eq!(without.opt_strs_len()?, -1);
    Ok(())
}

// --- E2: Optional<String> as param and return (both Some and None) ------------

#[test]
fn echo_optional_string() -> Result<()> {
    ensure_loaded();
    let some = NestedHolder::echo_opt_string(Some(FFIString::from("hi")))?;
    assert_eq!(some.map(|s| s.as_str().to_string()), Some("hi".to_string()));

    let none = NestedHolder::echo_opt_string(None)?;
    assert!(none.is_none());
    Ok(())
}

// --- I2: directly write a container field via DerefMut ------------------------

#[test]
fn write_container_field_directly() -> Result<()> {
    ensure_loaded();
    let mut holder =
        ArrayHolder::new(Array::new(vec![1, 2]), Array::new(vec![]), Array::new(vec![]))?;
    // Replace the whole Array<i64> field from Rust (not via the C++ setter).
    holder.int_array = Array::new(vec![10, 20, 30, 40]);
    assert_eq!(holder.get_int_array_length()?, 4);
    assert_eq!(holder.int_array.get(2)?, 30);
    Ok(())
}
