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
//! `Any` is opaque: it carries an arbitrary payload across the FFI boundary
//! unchanged. Per the FFI convention a top-level `Any` argument is passed as the
//! non-owning `AnyView` (H1), while a returned/stored `Any` is owning (H2). The
//! tests below verify *transparent round-trip* — push a value in, get the same
//! value back out — for several underlying types and for an object, without C++
//! ever inspecting what's inside.

use test_any_types::ensure_loaded;
use test_any_types::generated::test_any_types::AnyHolder;
use tvm_ffi::{AnyView, Result, String as FFIString};

#[test]
fn echo_roundtrips_primitive_types() -> Result<()> {
    ensure_loaded();
    // One `Any`/`AnyView` parameter (H1) + an owning `Any` return (H2); the
    // payload must come back byte-for-byte regardless of its underlying type.
    let i: i64 = AnyHolder::echo(AnyView::from(&42i64))?.try_into()?;
    assert_eq!(i, 42);

    let f: f64 = AnyHolder::echo(AnyView::from(&2.5f64))?.try_into()?;
    assert_eq!(f, 2.5);

    let b: bool = AnyHolder::echo(AnyView::from(&true))?.try_into()?;
    assert!(b);

    let s = FFIString::from("hello");
    let back: FFIString = AnyHolder::echo(AnyView::from(&s))?.try_into()?;
    assert_eq!(back.as_str(), "hello");
    Ok(())
}

#[test]
fn echo_roundtrips_an_object() -> Result<()> {
    ensure_loaded();
    // An object (itself an `AnyHolder`) also passes through `Any` untouched:
    // the returned handle refers to the same heap object we put in.
    let original = AnyHolder::new(AnyView::from(&7i64))?;
    let mut echoed: AnyHolder = AnyHolder::echo(AnyView::from(&original))?.try_into()?;
    let inner: i64 = echoed.get_any()?.try_into()?;
    assert_eq!(inner, 7);
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
