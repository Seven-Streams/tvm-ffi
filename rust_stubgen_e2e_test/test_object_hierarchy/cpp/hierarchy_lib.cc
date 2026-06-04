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

#include <cstdint>
#include <iostream>
#include <sstream>

namespace test_object_hierarchy {

namespace ffi = tvm::ffi;

class ShapeObj : public ffi::Object {
 public:
  int64_t width;
  int64_t height;

  explicit ShapeObj(int64_t width = 0, int64_t height = 0) : width(width), height(height) {}

  static int64_t GetDefaultWidth() { return 100; }

  static int64_t GetDefaultHeight() { return 100; }

  int64_t GetArea() { return width * height; }

  int64_t GetPerimeter() { return 2 * (width + height); }

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
      .def("get_description", &ShapeObj::GetDescription, "get description");

  refl::TypeAttrDef<ShapeObj>().def(refl::type_attr::kConvert,
                                    &refl::details::FFIConvertFromAnyViewToObjectRef<Shape>);

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
}

}  // namespace test_object_hierarchy
