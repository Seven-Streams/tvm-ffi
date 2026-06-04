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
//! Compile-fail: taking `&mut` through a read-only type requires `DerefMut`,
//! which is not generated for immutable (`def_ro`) types.
use test_immutable_types::generated::test_immutable_types::ImmutableVersion;

fn main() {
    let mut v = ImmutableVersion::new(1, 2, 3, tvm_ffi::String::from("x")).unwrap();
    let _r: &mut i64 = &mut v.major;
}
