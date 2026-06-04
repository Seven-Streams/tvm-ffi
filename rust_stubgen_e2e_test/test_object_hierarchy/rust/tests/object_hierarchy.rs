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
//! End-to-end tests for the generated `test_object_hierarchy` bindings.

use test_object_hierarchy::ensure_loaded;
use test_object_hierarchy::generated::test_object_hierarchy::{Circle, Rectangle, Shape};
use tvm_ffi::Result;

#[test]
fn shape_static_defaults() -> Result<()> {
    ensure_loaded();
    assert_eq!(Shape::get_default_width()?, 100);
    assert_eq!(Shape::get_default_height()?, 100);
    Ok(())
}

#[test]
fn shape_construct_methods_and_mutation() -> Result<()> {
    ensure_loaded();
    let mut shape = Shape::new(3, 4)?;
    assert_eq!(shape.width, 3);
    assert_eq!(shape.height, 4);
    assert_eq!(shape.get_area()?, 12);
    assert_eq!(shape.get_perimeter()?, 14);
    assert_eq!(shape.get_description()?.as_str(), "Shape[width=3,height=4]");

    // C++ method mutates the shared heap object.
    shape.resize(5, 6)?;
    assert_eq!(shape.width, 5);
    assert_eq!(shape.height, 6);
    assert_eq!(shape.get_area()?, 30);

    // Rust-side field mutation through DerefMut.
    shape.width = 10;
    assert_eq!(shape.get_area()?, 60);
    Ok(())
}

#[test]
fn circle_inherits_shape_fields() -> Result<()> {
    ensure_loaded();
    // CircleObj extends ShapeObj; the constructor sets width=height=radius*2.
    let mut circle = Circle::new(5)?;
    assert_eq!(circle.radius, 5);
    // Inherited fields reachable via the embedded `base: ShapeObj` (double Deref).
    assert_eq!(circle.width, 10);
    assert_eq!(circle.height, 10);
    assert_eq!(circle.get_circle_area()?, 31415927);

    circle.set_radius(8)?;
    assert_eq!(circle.radius, 8);
    assert_eq!(circle.width, 16);
    assert_eq!(circle.height, 16);
    Ok(())
}

#[test]
fn rectangle_inherits_shape_fields() -> Result<()> {
    ensure_loaded();
    let mut rect = Rectangle::new(4, 4, false)?;
    assert_eq!(rect.width, 4);
    assert_eq!(rect.height, 4);
    assert!(!rect.is_square);

    // C++ method recomputes is_square from width == height.
    rect.update_square_flag()?;
    assert!(rect.is_square);

    let mut non_square = Rectangle::new(4, 7, false)?;
    non_square.update_square_flag()?;
    assert!(!non_square.is_square);
    Ok(())
}
