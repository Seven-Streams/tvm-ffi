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
//! End-to-end tests for the generated `test_immutable_types` bindings.

use test_immutable_types::ensure_loaded;
use test_immutable_types::generated::test_immutable_types::{
    ImmutableMetadata, ImmutableVersion, MixedMutability,
};
use tvm_ffi::{Array, Result, String as FFIString};

#[test]
fn version_static_methods() -> Result<()> {
    ensure_loaded();
    assert_eq!(ImmutableVersion::get_current_major()?, 2);
    assert_eq!(ImmutableVersion::get_current_minor()?, 1);
    assert_eq!(ImmutableVersion::get_current_patch()?, 0);
    Ok(())
}

#[test]
fn version_read_only_fields_and_methods() -> Result<()> {
    ensure_loaded();
    // ImmutableVersion is read-only (def_ro): Deref-only, &self receivers.
    let version = ImmutableVersion::new(1, 4, 2, FFIString::from("rc1"))?;
    assert_eq!(version.major, 1);
    assert_eq!(version.minor, 4);
    assert_eq!(version.patch, 2);
    assert_eq!(version.label.as_str(), "rc1");

    assert_eq!(version.get_version_string()?.as_str(), "1.4.2-rc1");
    assert!(version.is_greater_than_version(1, 4, 1)?);
    assert!(!version.is_greater_than_version(1, 4, 2)?);
    assert!(!version.is_greater_than_version(2, 0, 0)?);
    Ok(())
}

#[test]
fn metadata_with_array_field() -> Result<()> {
    ensure_loaded();
    assert_eq!(ImmutableMetadata::get_default_license()?.as_str(), "Apache-2.0");

    let keywords: Array<FFIString> =
        Array::new(vec![FFIString::from("ffi"), FFIString::from("rust")]);
    let meta = ImmutableMetadata::new(
        FFIString::from("tvm-ffi"),
        FFIString::from("apache"),
        FFIString::from("Apache-2.0"),
        keywords,
    )?;
    assert_eq!(meta.name.as_str(), "tvm-ffi");
    assert_eq!(meta.author.as_str(), "apache");
    assert_eq!(meta.license.as_str(), "Apache-2.0");
    assert_eq!(meta.get_keyword_count()?, 2);
    assert_eq!(
        meta.to_json()?.as_str(),
        "{\"name\":\"tvm-ffi\",\"author\":\"apache\",\"license\":\"Apache-2.0\"}"
    );
    Ok(())
}

#[test]
fn mixed_mutability_treated_as_immutable() -> Result<()> {
    ensure_loaded();
    // MixedMutability has one ro + one rw field -> treated as read-only by the
    // generator (Deref only, &self receivers), but the C++ method still mutates.
    let id = MixedMutability::get_next_id()?;
    assert!(id >= 1000);

    let holder = MixedMutability::new(7, 0)?;
    assert_eq!(holder.immutable_id, 7);
    assert_eq!(holder.mutable_counter, 0);

    holder.increment_counter()?;
    holder.increment_counter()?;
    assert_eq!(holder.mutable_counter, 2);
    Ok(())
}
