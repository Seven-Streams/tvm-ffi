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
//! Runnable demo for the stubgen-generated `IntPair` bindings.
//!
//! Run with `cargo run --example demo` (after building the C++ library and
//! generating the bindings — see `scripts/run_stubgen.sh`).

use rust_stubgen_example::ensure_loaded;
use rust_stubgen_example::generated::rust_stubgen::IntPair;
use tvm_ffi::Result;

fn main() -> Result<()> {
    // dlopen the C++ library once so the reflection registry is populated.
    ensure_loaded();

    // Construct via the reflected `__ffi_init__` constructor.
    let mut p = IntPair::new(3, 4)?;
    println!("IntPair::new(3, 4) -> a={}, b={}", p.a, p.b);
    assert_eq!((p.a, p.b), (3, 4));

    // Call an instance method.
    println!("p.sum() = {}", p.sum()?);
    assert_eq!(p.sum()?, 7);

    // Mutate a field through DerefMut (all C++ fields are writable).
    p.a = 10;
    println!("after p.a = 10: sum() = {}", p.sum()?);
    assert_eq!(p.sum()?, 14);

    // A C++ method that mutates the shared heap object in place.
    p.scale(2)?;
    println!("after p.scale(2): a={}, b={}", p.a, p.b);
    assert_eq!((p.a, p.b), (20, 8));

    // A method that returns a freshly-constructed object.
    let s = p.swapped()?;
    println!("p.swapped() -> a={}, b={}", s.a, s.b);
    assert_eq!((s.a, s.b), (8, 20));

    // Error propagation: a C++ throw surfaces as `Err`, not a panic.
    let err = p.checked_div(0).expect_err("divide-by-zero must error");
    println!("p.checked_div(0) -> Err({}): {}", err.kind().as_str(), err.message());
    assert_eq!(err.kind().as_str(), "ValueError");

    println!("\nAll IntPair binding checks passed.");
    Ok(())
}
