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
//! I3: compile-time guarantee that read-only (`def_ro`) types reject mutation.
//!
//! `trybuild` compiles each `tests/ui/*.rs` program and asserts it FAILS to
//! compile, matching the committed `.stderr`. If a toolchain bump changes the
//! diagnostic text, regenerate with `TRYBUILD=overwrite cargo test`.

#[test]
fn readonly_types_reject_mutation() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
