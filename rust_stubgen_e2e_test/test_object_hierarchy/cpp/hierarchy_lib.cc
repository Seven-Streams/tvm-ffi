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
/*!
 * \file hierarchy_lib.cc
 * \brief C++ types testing object hierarchy, constructors, and methods.
 */
#include <tvm/ffi/tvm_ffi.h>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <sstream>

namespace test_object_hierarchy {

namespace ffi = tvm::ffi;

// Forward declaration so ShapeObj can take/return the `Shape` ref wrapper in
// methods exercising object-as-parameter / object-as-return codegen.
class Shape;

class ShapeObj : public ffi::Object {
 public:
  int64_t width;
  int64_t height;

  explicit ShapeObj(int64_t width = 0, int64_t height = 0) : width(width), height(height) {}

  static int64_t GetDefaultWidth() { return 100; }

  static int64_t GetDefaultHeight() { return 100; }

  int64_t GetArea() { return width * height; }

  int64_t GetPerimeter() { return 2 * (width + height); }

  // --- D: error propagation -------------------------------------------------
  // Throws on divide-by-zero; otherwise returns area / denom.
  int64_t CheckedDiv(int64_t denom) {
    if (denom == 0) {
      TVM_FFI_THROW(ValueError) << "CheckedDiv: division by zero";
    }
    return GetArea() / denom;
  }

  // --- A/B: registered object as parameter / return (defined out-of-line) ---
  // Container-of-object variants (Array<Shape>/Optional<Shape>) live on the
  // separate `ShapeBatch` class below, since those templates need a complete
  // `Shape` which is not yet available inside this class body.
  bool SameSizeAs(Shape other);                   // A1 instance takes object param
  static int64_t CombinedArea(Shape a, Shape b);  // A2 static takes object params
  Shape Scaled(int64_t factor);                   // B1 instance returns new object

  void Resize(int64_t new_width, int64_t new_height) {
    width = new_width;
    height = new_height;
    std::cout << "[test_object_hierarchy] Shape::Resize(" << new_width << ", " << new_height << ")"
              << std::endl;
  }

  ffi::String GetDescription() {
    std::ostringstream oss;
    oss << "Shape[width=" << width << ",height=" << height << "]";
    return ffi::String(oss.str());
  }

  ~ShapeObj() {
    std::cout << "[test_object_hierarchy] ~ShapeObj() width=" << width << " height=" << height
              << std::endl;
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test_object_hierarchy.Shape", ShapeObj, ffi::Object);
};

class Shape : public ffi::ObjectRef {
 public:
  explicit Shape(int64_t width = 0, int64_t height = 0) {
    data_ = ffi::make_object<ShapeObj>(width, height);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Shape, ffi::ObjectRef, ShapeObj);
};

// Out-of-line definitions of the methods that mention `Shape` (now complete).
inline bool ShapeObj::SameSizeAs(Shape other) {
  return width * height == other->width * other->height;
}

inline int64_t ShapeObj::CombinedArea(Shape a, Shape b) {
  return a->width * a->height + b->width * b->height;
}

inline Shape ShapeObj::Scaled(int64_t factor) { return Shape(width * factor, height * factor); }

// Hosts the container-of-object methods (A4 / B3 / B4) now that `Shape` is a
// complete type, so `ffi::Array<Shape>` / `ffi::Optional<Shape>` can be used.
class ShapeBatchObj : public ffi::Object {
 public:
  static int64_t TotalArea(ffi::Array<Shape> shapes) {  // A4 Array<object> param
    int64_t total = 0;
    for (Shape s : shapes) {
      total += s->width * s->height;
    }
    return total;
  }

  static ffi::Optional<Shape> NonEmptyOrNone(Shape s) {  // B3 nullable object return
    if (s->width * s->height > 0) {
      return s;
    }
    return std::nullopt;
  }

  static ffi::Array<Shape> Split(Shape s) {  // B4 Array<object> return
    return ffi::Array<Shape>({Shape(s->width, 0), Shape(0, s->height)});
  }

  // D3: static method that throws.
  static int64_t SafeDivide(int64_t a, int64_t b) {
    if (b == 0) {
      TVM_FFI_THROW(ValueError) << "SafeDivide: division by zero";
    }
    return a / b;
  }

  // B2: pure static factory producing a fresh object (Rust takes over refcount).
  static Shape UnitShape() { return Shape(1, 1); }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test_object_hierarchy.ShapeBatch", ShapeBatchObj, ffi::Object);
};

class ShapeBatch : public ffi::ObjectRef {
 public:
  ShapeBatch() { data_ = ffi::make_object<ShapeBatchObj>(); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ShapeBatch, ffi::ObjectRef, ShapeBatchObj);
};

// D2: a constructor that throws on invalid arguments. `new(...)` must surface
// the C++ exception as `Err`, not panic across the FFI boundary.
class ValidatedObj : public ffi::Object {
 public:
  int64_t value;

  explicit ValidatedObj(int64_t value = 0) : value(value) {
    if (value < 0) {
      TVM_FFI_THROW(ValueError) << "Validated: value must be non-negative";
    }
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test_object_hierarchy.Validated", ValidatedObj, ffi::Object);
};

class Validated : public ffi::ObjectRef {
 public:
  explicit Validated(int64_t value = 0) { data_ = ffi::make_object<ValidatedObj>(value); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Validated, ffi::ObjectRef, ValidatedObj);
};

// --- C: registered object(s) as FIELD types (nested objects) -----------------
class GroupObj : public ffi::Object {
 public:
  Shape primary;              // single nested object field
  ffi::Array<Shape> members;  // container-of-object field

  explicit GroupObj(Shape primary, ffi::Array<Shape> members)
      : primary(primary), members(members) {}

  int64_t TotalArea() {
    int64_t total = primary->width * primary->height;
    for (Shape s : members) {
      total += s->width * s->height;
    }
    return total;
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test_object_hierarchy.Group", GroupObj, ffi::Object);
};

class Group : public ffi::ObjectRef {
 public:
  explicit Group(Shape primary, ffi::Array<Shape> members) {
    data_ = ffi::make_object<GroupObj>(primary, members);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Group, ffi::ObjectRef, GroupObj);
};

// --- J: destructor / refcount instrumentation --------------------------------
// A process-global live-instance counter lets a Rust test assert that dropping
// an object runs its C++ destructor exactly once (and that a clone shares one
// underlying object rather than allocating a second). Only the dedicated J test
// constructs `Tracked`, so the count stays deterministic under parallel tests.
class TrackedObj : public ffi::Object {
 public:
  static std::atomic<int64_t>& Counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
  }

  TrackedObj() { Counter().fetch_add(1, std::memory_order_relaxed); }
  ~TrackedObj() { Counter().fetch_sub(1, std::memory_order_relaxed); }

  static int64_t LiveCount() { return Counter().load(std::memory_order_relaxed); }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test_object_hierarchy.Tracked", TrackedObj, ffi::Object);
};

class Tracked : public ffi::ObjectRef {
 public:
  Tracked() { data_ = ffi::make_object<TrackedObj>(); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Tracked, ffi::ObjectRef, TrackedObj);
};

class CircleObj : public ShapeObj {
 public:
  int64_t radius;

  explicit CircleObj(int64_t radius = 0) : ShapeObj(radius * 2, radius * 2), radius(radius) {}

  int64_t GetCircleArea() { return 31415927; }  // Approximation of pi*r^2 * 10^7

  void SetRadius(int64_t new_radius) {
    radius = new_radius;
    width = new_radius * 2;
    height = new_radius * 2;
    std::cout << "[test_object_hierarchy] Circle::SetRadius(" << new_radius << ")" << std::endl;
  }

  ~CircleObj() {
    std::cout << "[test_object_hierarchy] ~CircleObj() radius=" << radius << std::endl;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test_object_hierarchy.Circle", CircleObj, ShapeObj);
};

class Circle : public Shape {
 public:
  explicit Circle(int64_t radius = 0) : Shape(ffi::UnsafeInit{}) {
    data_ = ffi::make_object<CircleObj>(radius);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Circle, Shape, CircleObj);
};

class RectangleObj : public ShapeObj {
 public:
  bool is_square;

  explicit RectangleObj(int64_t width = 0, int64_t height = 0, bool is_square = false)
      : ShapeObj(width, height), is_square(is_square) {}

  void UpdateSquareFlag() {
    is_square = (width == height);
    std::cout << "[test_object_hierarchy] Rectangle::UpdateSquareFlag() is_square="
              << (is_square ? "true" : "false") << std::endl;
  }

  ~RectangleObj() {
    std::cout << "[test_object_hierarchy] ~RectangleObj() is_square="
              << (is_square ? "true" : "false") << std::endl;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test_object_hierarchy.Rectangle", RectangleObj, ShapeObj);
};

class Rectangle : public Shape {
 public:
  explicit Rectangle(int64_t width = 0, int64_t height = 0, bool is_square = false)
      : Shape(ffi::UnsafeInit{}) {
    data_ = ffi::make_object<RectangleObj>(width, height, is_square);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Rectangle, Shape, RectangleObj);
};

// --- K1: 3+ level inheritance (Object -> Shape -> Box3D -> ColoredBox) --------
// Box3D is a NON-final middle type (so it can be subclassed); ColoredBox is the
// final leaf. The Rust side reaches the top-most `Shape` fields through a 3-deep
// Deref chain (ColoredBoxObj -> Box3DObj -> ShapeObj).
class Box3DObj : public ShapeObj {
 public:
  int64_t depth;

  explicit Box3DObj(int64_t width = 0, int64_t height = 0, int64_t depth = 0)
      : ShapeObj(width, height), depth(depth) {}

  int64_t Volume() { return width * height * depth; }

  TVM_FFI_DECLARE_OBJECT_INFO("test_object_hierarchy.Box3D", Box3DObj, ShapeObj);
};

class Box3D : public Shape {
 public:
  explicit Box3D(int64_t width = 0, int64_t height = 0, int64_t depth = 0)
      : Shape(ffi::UnsafeInit{}) {
    data_ = ffi::make_object<Box3DObj>(width, height, depth);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Box3D, Shape, Box3DObj);
};

class ColoredBoxObj : public Box3DObj {
 public:
  int64_t color;

  explicit ColoredBoxObj(int64_t width = 0, int64_t height = 0, int64_t depth = 0,
                         int64_t color = 0)
      : Box3DObj(width, height, depth), color(color) {}

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test_object_hierarchy.ColoredBox", ColoredBoxObj, Box3DObj);
};

class ColoredBox : public Box3D {
 public:
  explicit ColoredBox(int64_t width = 0, int64_t height = 0, int64_t depth = 0, int64_t color = 0)
      : Box3D(ffi::UnsafeInit{}) {
    data_ = ffi::make_object<ColoredBoxObj>(width, height, depth, color);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ColoredBox, Box3D, ColoredBoxObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<ShapeObj>()
      .def(refl::init<int64_t, int64_t>())
      .def_rw("width", &ShapeObj::width, "width of the shape")
      .def_rw("height", &ShapeObj::height, "height of the shape")
      .def_static("get_default_width", &ShapeObj::GetDefaultWidth, "default width")
      .def_static("get_default_height", &ShapeObj::GetDefaultHeight, "default height")
      .def("get_area", &ShapeObj::GetArea, "get area")
      .def("get_perimeter", &ShapeObj::GetPerimeter, "get perimeter")
      .def("resize", &ShapeObj::Resize, "resize the shape")
      .def("get_description", &ShapeObj::GetDescription, "get description")
      .def("checked_div", &ShapeObj::CheckedDiv, "area / denom, throws on zero")
      .def("same_size_as", &ShapeObj::SameSizeAs, "true if same area as other")
      .def("scaled", &ShapeObj::Scaled, "return a new scaled Shape")
      .def_static("combined_area", &ShapeObj::CombinedArea, "sum of two shapes' areas");

  refl::TypeAttrDef<ShapeObj>().def(refl::type_attr::kConvert,
                                    &refl::details::FFIConvertFromAnyViewToObjectRef<Shape>);

  refl::ObjectDef<ShapeBatchObj>()
      .def(refl::init<>())
      .def_static("total_area", &ShapeBatchObj::TotalArea, "sum of areas in an Array<Shape>")
      .def_static("non_empty_or_none", &ShapeBatchObj::NonEmptyOrNone,
                  "Some(s) if area>0 else None")
      .def_static("split", &ShapeBatchObj::Split, "split a shape into two")
      .def_static("safe_divide", &ShapeBatchObj::SafeDivide, "a / b, throws on zero")
      .def_static("unit_shape", &ShapeBatchObj::UnitShape, "return a fresh 1x1 Shape");

  refl::TypeAttrDef<ShapeBatchObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<ShapeBatch>);

  refl::ObjectDef<ValidatedObj>()
      .def(refl::init<int64_t>())
      .def_rw("value", &ValidatedObj::value, "non-negative value");

  refl::TypeAttrDef<ValidatedObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<Validated>);

  refl::ObjectDef<GroupObj>()
      .def(refl::init<Shape, ffi::Array<Shape>>())
      .def_rw("primary", &GroupObj::primary, "primary nested shape")
      .def_rw("members", &GroupObj::members, "array of member shapes")
      .def("total_area", &GroupObj::TotalArea, "primary area + members areas");

  refl::TypeAttrDef<GroupObj>().def(refl::type_attr::kConvert,
                                    &refl::details::FFIConvertFromAnyViewToObjectRef<Group>);

  refl::ObjectDef<TrackedObj>()
      .def(refl::init<>())
      .def_static("live_count", &TrackedObj::LiveCount, "number of live TrackedObj instances");

  refl::TypeAttrDef<TrackedObj>().def(refl::type_attr::kConvert,
                                      &refl::details::FFIConvertFromAnyViewToObjectRef<Tracked>);

  refl::ObjectDef<CircleObj>()
      .def(refl::init<int64_t>())
      .def_rw("radius", &CircleObj::radius, "radius of the circle")
      .def("get_circle_area", &CircleObj::GetCircleArea, "get circle area approximation")
      .def("set_radius", &CircleObj::SetRadius, "set radius");

  refl::TypeAttrDef<CircleObj>().def(refl::type_attr::kConvert,
                                     &refl::details::FFIConvertFromAnyViewToObjectRef<Circle>);

  refl::ObjectDef<RectangleObj>()
      .def(refl::init<int64_t, int64_t, bool>())
      .def_rw("is_square", &RectangleObj::is_square, "whether the rectangle is a square")
      .def("update_square_flag", &RectangleObj::UpdateSquareFlag, "update square flag");

  refl::TypeAttrDef<RectangleObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<Rectangle>);

  refl::ObjectDef<Box3DObj>()
      .def(refl::init<int64_t, int64_t, int64_t>())
      .def_rw("depth", &Box3DObj::depth, "depth of the box")
      .def("volume", &Box3DObj::Volume, "width * height * depth");

  refl::TypeAttrDef<Box3DObj>().def(refl::type_attr::kConvert,
                                    &refl::details::FFIConvertFromAnyViewToObjectRef<Box3D>);

  refl::ObjectDef<ColoredBoxObj>()
      .def(refl::init<int64_t, int64_t, int64_t, int64_t>())
      .def_rw("color", &ColoredBoxObj::color, "color code of the box");

  refl::TypeAttrDef<ColoredBoxObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<ColoredBox>);
}

}  // namespace test_object_hierarchy
