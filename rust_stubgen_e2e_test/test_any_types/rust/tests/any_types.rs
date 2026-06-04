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
//! End-to-end tests for the generated `test_any_types` bindings.
//!
//! Per the FFI convention, a top-level `Any` argument is passed as the
//! non-owning `AnyView`, while a returned/stored `Any` is owning. These tests
//! push several underlying types through one `Any` parameter (H1) and read an
//! `Any` back out of return values and a field (H2).

use test_any_types::ensure_loaded;
use test_any_types::generated::test_any_types::AnyHolder;
use tvm_ffi::{AnyView, Result, String as FFIString};

#[test]
fn describe_any_dispatches_on_runtime_type() -> Result<()> {
    ensure_loaded();
    // H1: one `Any`/`AnyView` parameter, several underlying types.
    assert_eq!(AnyHolder::describe_any(AnyView::from(&42i64))?.as_str(), "int=42");
    assert_eq!(AnyHolder::describe_any(AnyView::from(&2.5f64))?.as_str(), "float=2.5");
    assert_eq!(AnyHolder::describe_any(AnyView::from(&true))?.as_str(), "bool=true");

    let s = FFIString::from("hi");
    assert_eq!(AnyHolder::describe_any(AnyView::from(&s))?.as_str(), "str=hi");
    Ok(())
}

#[test]
fn echo_returns_owning_any() -> Result<()> {
    ensure_loaded();
    // H2: the returned `Any` owns its payload; extract it back out.
    let echoed = AnyHolder::echo(AnyView::from(&7i64))?;
    let n: i64 = echoed.try_into()?;
    assert_eq!(n, 7);

    let s = FFIString::from("world");
    let echoed = AnyHolder::echo(AnyView::from(&s))?;
    let back: FFIString = echoed.try_into()?;
    assert_eq!(back.as_str(), "world");
    Ok(())
}

#[test]
fn any_field_roundtrip() -> Result<()> {
    ensure_loaded();
    // Construct with an `Any` field, then read/write it through C++ methods.
    let mut holder = AnyHolder::new(AnyView::from(&100i64))?;
    let got: i64 = holder.get_any()?.try_into()?;
    assert_eq!(got, 100);

    let s = FFIString::from("changed");
    holder.set_any(AnyView::from(&s))?;
    let got: FFIString = holder.get_any()?.try_into()?;
    assert_eq!(got.as_str(), "changed");
    Ok(())
}
