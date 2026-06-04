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
//! End-to-end tests for the generated `test_ffi_types` bindings: core FFI value
//! types `ffi::Shape` (F2) and `DataType` / `Device` (F3) as param/return/field.

use test_ffi_types::ensure_loaded;
use test_ffi_types::generated::test_ffi_types::FfiTypesHolder;
use tvm_ffi::{
    AnyView, CPUNDAlloc, DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, Function, Result,
    Shape, Tensor,
};

#[test]
fn shape_param_and_return() -> Result<()> {
    ensure_loaded();
    // F2: Shape as a parameter.
    assert_eq!(FfiTypesHolder::shape_product(Shape::from(vec![2, 3, 4]))?, 24);
    // Shape as a return value, fed straight back in as a parameter.
    let s = FfiTypesHolder::make_shape(3, 5)?;
    assert_eq!(FfiTypesHolder::shape_product(s)?, 15);
    Ok(())
}

#[test]
fn datatype_param_and_return() -> Result<()> {
    ensure_loaded();
    // F3: DataType as a parameter and as an (echoed) return value.
    let dt = DLDataType { code: DLDataTypeCode::kDLFloat as u8, bits: 64, lanes: 1 };
    assert_eq!(FfiTypesHolder::dtype_bits(dt)?, 64);

    let echoed = FfiTypesHolder::echo_dtype(dt)?;
    assert_eq!(echoed.code, DLDataTypeCode::kDLFloat as u8);
    assert_eq!(echoed.bits, 64);
    assert_eq!(echoed.lanes, 1);
    Ok(())
}

#[test]
fn device_param_and_return() -> Result<()> {
    ensure_loaded();
    // F3: Device as a parameter and as an (echoed) return value.
    let dev = DLDevice::new(DLDeviceType::kDLCPU, 3);
    assert_eq!(FfiTypesHolder::device_id(dev)?, 3);

    let echoed = FfiTypesHolder::echo_device(dev)?;
    assert_eq!(echoed.device_type, DLDeviceType::kDLCPU);
    assert_eq!(echoed.device_id, 3);
    Ok(())
}

#[test]
fn ffi_type_fields() -> Result<()> {
    ensure_loaded();
    // Shape / DataType / Device as object fields, read back through Deref.
    let dt = DLDataType { code: DLDataTypeCode::kDLInt as u8, bits: 16, lanes: 1 };
    let dev = DLDevice::new(DLDeviceType::kDLCPU, 1);
    let mut holder = FfiTypesHolder::new(Shape::from(vec![5, 6]), dt, dev)?;
    assert_eq!(holder.shape_ndim()?, 2);
    assert_eq!(holder.dtype.bits, 16);
    assert_eq!(holder.device.device_id, 1);
    Ok(())
}

// --- G: Function (callback) as param / return ---------------------------------

#[test]
fn function_as_param() -> Result<()> {
    ensure_loaded();
    // G1: pass a Rust closure as a `Function`; C++ invokes it via fn(x).
    let doubler = Function::from_typed(|x: i64| -> Result<i64> { Ok(x * 2) });
    assert_eq!(FfiTypesHolder::apply_fn(doubler, 21)?, 42);
    Ok(())
}

#[test]
fn function_as_return() -> Result<()> {
    ensure_loaded();
    // G2: receive a `Function` produced by C++ and call it from Rust.
    let adder = FfiTypesHolder::make_adder(10)?;
    let r: i64 = adder.call_packed(&[AnyView::from(&5i64)])?.try_into()?;
    assert_eq!(r, 15);
    Ok(())
}

// --- F1: Tensor (DLPack) as param / return ------------------------------------

#[test]
fn tensor_param_and_return() -> Result<()> {
    ensure_loaded();
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, &[2, 3, 4], dtype, device);

    // Tensor as a parameter: C++ reads ndim / numel / dtype.
    assert_eq!(FfiTypesHolder::tensor_ndim(tensor.clone())?, 3);
    assert_eq!(FfiTypesHolder::tensor_numel(tensor.clone())?, 24);
    assert_eq!(FfiTypesHolder::tensor_dtype(tensor.clone())?.bits, 32);

    // Tensor as a return value (echoed): the round-trip preserves shape/dtype.
    let echoed = FfiTypesHolder::echo_tensor(tensor)?;
    assert_eq!(echoed.shape(), &[2, 3, 4]);
    assert_eq!(echoed.ndim(), 3);
    assert_eq!(echoed.dtype().code, DLDataTypeCode::kDLFloat as u8);
    Ok(())
}
