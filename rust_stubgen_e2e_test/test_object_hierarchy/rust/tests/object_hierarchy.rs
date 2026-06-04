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
use test_object_hierarchy::generated::test_object_hierarchy::{
    Box3D, Circle, ColoredBox, Group, Rectangle, Shape, ShapeBatch, Tracked,
};
use tvm_ffi::{Array, Result};

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

// --- D: error propagation (`Result::Err` branch) -----------------------------

#[test]
fn checked_div_ok_and_err() -> Result<()> {
    ensure_loaded();
    let mut shape = Shape::new(6, 4)?; // area = 24
    assert_eq!(shape.checked_div(3)?, 8);

    // Divide-by-zero throws a `ValueError` in C++; it must surface as `Err`,
    // not a panic / abort across the FFI boundary.
    let err = shape.checked_div(0).expect_err("expected divide-by-zero error");
    assert_eq!(err.kind().as_str(), "ValueError");
    assert!(
        err.message().contains("division by zero"),
        "unexpected error message: {}",
        err.message()
    );
    Ok(())
}

// --- A: registered object as a method / static-method parameter ---------------

#[test]
fn object_as_instance_method_param() -> Result<()> {
    ensure_loaded();
    // A1: instance method takes another Shape and compares areas.
    let mut a = Shape::new(2, 6)?; // area 12
    let same = Shape::new(3, 4)?; // area 12
    let diff = Shape::new(3, 5)?; // area 15
    assert!(a.same_size_as(same)?);
    assert!(!a.same_size_as(diff)?);
    Ok(())
}

#[test]
fn object_as_static_method_params() -> Result<()> {
    ensure_loaded();
    // A2: static method takes two Rust-held objects and passes them back to C++.
    let a = Shape::new(2, 3)?; // 6
    let b = Shape::new(4, 5)?; // 20
    assert_eq!(Shape::combined_area(a, b)?, 26);
    Ok(())
}

#[test]
fn array_of_objects_as_param() -> Result<()> {
    ensure_loaded();
    // A4: Array<Shape> argument — exercises the Array<object> codec + element refcount.
    let shapes = Array::new(vec![Shape::new(2, 3)?, Shape::new(4, 5)?, Shape::new(1, 1)?]);
    assert_eq!(ShapeBatch::total_area(shapes)?, 6 + 20 + 1);
    Ok(())
}

// --- B: registered object as a (non-constructor) return value -----------------

#[test]
fn object_returned_from_instance_method() -> Result<()> {
    ensure_loaded();
    // B1: instance method produces a brand-new object; Rust takes over its refcount.
    let mut base = Shape::new(3, 5)?;
    let scaled = base.scaled(2)?;
    assert_eq!(scaled.width, 6);
    assert_eq!(scaled.height, 10);
    // The original is untouched (a new object was returned, not a mutation).
    assert_eq!(base.width, 3);
    assert_eq!(base.height, 5);
    Ok(())
}

#[test]
fn nullable_object_return() -> Result<()> {
    ensure_loaded();
    // B3: Optional<Shape> covering both Some and None branches.
    let non_empty = Shape::new(3, 4)?;
    let some = ShapeBatch::non_empty_or_none(non_empty)?;
    let shape = some.expect("area > 0 should yield Some");
    assert_eq!(shape.width, 3);
    assert_eq!(shape.height, 4);

    let empty = Shape::new(0, 7)?; // area 0
    assert!(ShapeBatch::non_empty_or_none(empty)?.is_none());
    Ok(())
}

#[test]
fn array_of_objects_returned() -> Result<()> {
    ensure_loaded();
    // B4: method returns Array<Shape>; verify element fields after the round-trip.
    let parts = ShapeBatch::split(Shape::new(8, 9)?)?;
    assert_eq!(parts.len(), 2);
    assert_eq!(parts.get(0)?.width, 8);
    assert_eq!(parts.get(0)?.height, 0);
    assert_eq!(parts.get(1)?.width, 0);
    assert_eq!(parts.get(1)?.height, 9);
    Ok(())
}

// --- C: registered object(s) as FIELD types (nested objects) -----------------

#[test]
fn nested_object_fields() -> Result<()> {
    ensure_loaded();
    // Construct with an object field + an Array<object> field.
    let members = Array::new(vec![Shape::new(2, 2)?, Shape::new(3, 3)?]);
    let mut group = Group::new(Shape::new(4, 5)?, members)?;

    // Read the single nested object field via Deref, then its inner fields.
    assert_eq!(group.primary.width, 4);
    assert_eq!(group.primary.height, 5);

    // Read the container-of-objects field.
    assert_eq!(group.members.len(), 2);
    assert_eq!(group.members.get(0)?.width, 2);
    assert_eq!(group.members.get(1)?.height, 3);

    // C++ method sees the same nested objects: 20 + 4 + 9 = 33.
    assert_eq!(group.total_area()?, 33);

    // Write the object field via DerefMut, then confirm C++ observes the swap.
    group.primary = Shape::new(10, 10)?;
    assert_eq!(group.primary.width, 10);
    assert_eq!(group.total_area()?, 100 + 4 + 9);
    Ok(())
}

// --- J: destructor / refcount -------------------------------------------------

#[test]
fn clone_shares_underlying_object() -> Result<()> {
    ensure_loaded();
    // J2 (behavioral): a clone is a second owner of the SAME heap object.
    let mut a = Shape::new(3, 4)?;
    let mut b = a.clone();
    a.width = 99; // mutate through `a`
    assert_eq!(b.width, 99); // observed through `b` => shared object, not a copy

    drop(a); // dropping one owner must NOT free the object
    assert_eq!(b.get_area()?, 99 * 4); // `b` is still valid and usable
    b.height = 5;
    assert_eq!(b.width, 99);
    Ok(())
}

#[test]
fn drop_runs_destructor_exactly_once() -> Result<()> {
    ensure_loaded();
    // J1: the C++ destructor must run exactly once when the last owner drops.
    // `Tracked` keeps a process-global live-instance counter; only this test
    // constructs it, so the deltas are deterministic under parallel execution.
    let before = Tracked::live_count()?;

    let t = Tracked::new()?;
    assert_eq!(Tracked::live_count()?, before + 1, "ctor should create one C++ object");

    let t2 = t.clone();
    assert_eq!(Tracked::live_count()?, before + 1, "clone shares; no second C++ object");

    drop(t);
    assert_eq!(Tracked::live_count()?, before + 1, "one owner left; object still alive");

    drop(t2);
    assert_eq!(Tracked::live_count()?, before, "last drop runs the destructor exactly once");
    Ok(())
}

// --- K1: 3+ level inheritance (Object -> Shape -> Box3D -> ColoredBox) --------

#[test]
fn three_level_inheritance_field_access() -> Result<()> {
    ensure_loaded();
    // ColoredBox is 3 user levels deep; every ancestor's fields are reachable
    // through the multi-level Deref chain.
    let cb = ColoredBox::new(2, 3, 4, 7)?;
    assert_eq!(cb.color, 7); // own field (ColoredBoxObj)
    assert_eq!(cb.depth, 4); // Box3DObj  (1 Deref)
    assert_eq!(cb.width, 2); // ShapeObj  (2 Derefs)
    assert_eq!(cb.height, 3); // ShapeObj  (2 Derefs)
    Ok(())
}

#[test]
fn mid_level_type_has_own_method_and_inherited_fields() -> Result<()> {
    ensure_loaded();
    // Box3D (middle, non-final) can call its own `volume()` and read the
    // inherited Shape fields.
    let mut b = Box3D::new(2, 3, 5)?;
    assert_eq!(b.depth, 5);
    assert_eq!(b.width, 2); // inherited ShapeObj field
    assert_eq!(b.volume()?, 30); // own method: 2 * 3 * 5
    Ok(())
}
