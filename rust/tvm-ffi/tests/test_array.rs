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
use tvm_ffi::*;

/// Helper to create a Tensor with a specific float value and shape
fn create_tensor(val: f32, shape: &[i64]) -> Tensor {
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, shape, dtype, device);
    if let Ok(slice) = tensor.data_as_slice_mut::<f32>() {
        slice[0] = val;
    }
    tensor
}

/// Helper to extract the first float value from a Tensor
fn get_val(tensor: &Tensor) -> f32 {
    tensor
        .data_as_slice::<f32>()
        .expect("Type mismatch or null")[0]
}

#[test]
fn test_array_core_and_iteration() {
    let t1 = create_tensor(10.0, &[1, 2]);
    let t2 = create_tensor(20.0, &[3, 4, 5]);

    let array = Array::new(vec![t1.clone(), t2.clone()]);

    // Core Accessors
    assert_eq!(array.len(), 2);
    assert!(!array.is_empty());

    // Value Integrity
    assert_eq!(get_val(&array.get(0).unwrap()), 10.0);
    assert_eq!(array.get(0).unwrap().ndim(), 2);
    assert_eq!(array.get(1).unwrap().ndim(), 3);

    // Iteration
    let vals: Vec<f32> = array.iter().map(|t| get_val(&t)).collect();
    assert_eq!(vals, vec![10.0, 20.0]);
}

#[test]
fn test_array_view_is_borrowed_and_get_is_owning() {
    let text = String::from("a-long-array-element-that-is-reference-counted");
    let base = AnyView::from(&text).debug_strong_count().unwrap();
    let array = Array::new(vec![text.clone()]);
    assert_eq!(AnyView::from(&text).debug_strong_count(), Some(base + 1));

    {
        let view = array.view(0).unwrap();
        // A view borrows the array's element and does not acquire ownership.
        assert_eq!(view.debug_strong_count(), Some(base + 1));
        assert_eq!(view.try_as::<String>().as_deref(), Some(text.as_str()));
    }
    assert!(array.view(1).is_err());

    // `get` returns an owning value, which remains valid after the array drops.
    let owned = array.get(0).unwrap();
    assert_eq!(AnyView::from(&text).debug_strong_count(), Some(base + 2));
    drop(array);
    assert_eq!(owned.as_str(), text.as_str());
    assert_eq!(AnyView::from(&text).debug_strong_count(), Some(base + 1));
    drop(owned);
    assert_eq!(AnyView::from(&text).debug_strong_count(), Some(base));
}

#[test]
fn test_array_any_conversions() {
    let array = Array::new(vec![
        create_tensor(1.0, &[1]),
        create_tensor(2.0, &[1]),
        create_tensor(3.0, &[1]),
    ]);

    // Test Any/AnyView Roundtrip (Verifies AnyCompatible and Trait Bounds)
    let any = Any::from(array);
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIArray as i32);

    let back: Array<Tensor> = Array::try_from(any).expect("Any -> Array failed");
    assert_eq!(back.len(), 3);
    assert_eq!(get_val(&back.get(2).unwrap()), 3.0);

    let view = AnyView::from(&back);
    let back_from_view: Array<Tensor> = Array::try_from(view).expect("AnyView -> Array failed");
    assert_eq!(back_from_view.len(), 3);
}

#[test]
fn test_array_any_value_preserves_heterogeneous_elements() {
    let array = Array::new(vec![
        AnyValue::new(42i64),
        AnyValue::new(String::from("dynamic text")),
        AnyValue::new(Shape::from(vec![2, 3])),
        AnyValue::default(),
    ]);

    assert_eq!(array.get(0).unwrap().try_as::<i64>(), Some(42));
    assert_eq!(
        array.get(1).unwrap().try_as::<String>().as_deref(),
        Some("dynamic text")
    );
    assert_eq!(
        array.get(2).unwrap().try_as::<Shape>().unwrap().as_slice(),
        &[2, 3]
    );
    assert_eq!(array.get(3).unwrap().try_as::<Option<i64>>(), Some(None));

    let roundtrip: Array<AnyValue> =
        Array::try_from(Any::from(array)).expect("Any -> Array<AnyValue> failed");
    assert_eq!(roundtrip.len(), 4);
    assert_eq!(roundtrip.get(0).unwrap().try_as::<i64>(), Some(42));
    assert_eq!(
        roundtrip
            .get(2)
            .unwrap()
            .try_as::<Shape>()
            .unwrap()
            .as_slice(),
        &[2, 3]
    );
}

#[test]
fn test_array_recursive_type_checking() {
    // 1. Create an Array of Shapes
    let shape_array = Array::new(vec![Shape::from(vec![1, 2]), Shape::from(vec![3])]);

    // 2. Wrap it in Any
    let any_val = Any::from(shape_array);

    // 3. Try to convert Any (containing Shapes) into Array<Tensor>
    // This should FAIL because T::check_any_strict (Tensor) will fail on Shape elements
    let tensor_cast = Array::<Tensor>::try_from(any_val.clone());
    assert!(
        tensor_cast.is_err(),
        "Should not be able to cast Array<Shape> to Array<Tensor>"
    );

    // 4. Verify valid cast works
    let shape_cast = Array::<Shape>::try_from(any_val);
    assert!(
        shape_cast.is_ok(),
        "Should be able to cast back to correct type"
    );
}

#[test]
fn test_array_parametric_heterogeneity() {
    // Verify Array works with different ObjectRefCore types
    let shape_array = Array::new(vec![Shape::from(vec![1, 2, 3]), Shape::from(vec![10])]);
    assert_eq!(shape_array.get(0).unwrap().as_slice(), &[1, 2, 3]);
    assert_eq!(shape_array.get(1).unwrap().as_slice(), &[10]);

    let function_array = Array::new(vec![
        Function::get_global("ffi.String").unwrap(),
        Function::get_global("ffi.Bytes").unwrap(),
    ]);
    assert_eq!(
        into_typed_fn!(
            function_array.get(0).unwrap(),
            Fn(String) -> Result<String>
        )("hello".into())
        .unwrap(),
        "hello"
    );
    assert_eq!(
        into_typed_fn!(
            function_array.get(1).unwrap(),
            Fn(Bytes) -> Result<Bytes>
        )([1, 2, 3].into())
        .unwrap(),
        &[1, 2, 3]
    );
}
