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

// The C++ global function table deliberately has no internal mutex. Serialize
// every test in this binary that reads or mutates it while the cache tests
// temporarily remove and re-register globals.
static GLOBAL_FUNCTION_TABLE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_global_function_table_for_test() -> std::sync::MutexGuard<'static, ()> {
    GLOBAL_FUNCTION_TABLE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn test_function_dummpy_c_api() {
    let ret = unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() };
    assert_eq!(ret, 0);
}

#[test]
fn test_function_get_global_required() {
    let _registry_guard = lock_global_function_table_for_test();
    let fecho = Function::get_global("testing.echo").unwrap();
    let a = 1;
    let args = [AnyView::from(&a)];
    let result = fecho.call_packed(&args).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 1);
}

fn remove_global_for_test(name: &str) -> bool {
    let name = String::from(name);
    let result = Function::get_global("ffi.FunctionRemoveGlobal")
        .unwrap()
        .call_packed(&[AnyView::from(&name)])
        .unwrap();
    bool::try_from(result).unwrap()
}

#[test]
fn test_function_get_global_cached_retries_after_missing_global() {
    let _registry_guard = lock_global_function_table_for_test();
    static CACHE: std::sync::OnceLock<Function> = std::sync::OnceLock::new();

    const NAME: &str = "testing.rust.get_global_cached.retry";
    remove_global_for_test(NAME);

    assert!(Function::get_global_cached(&CACHE, NAME).is_err());

    Function::register_global(NAME, Function::from_typed(|| -> Result<i32> { Ok(17) })).unwrap();
    let function = Function::get_global_cached(&CACHE, NAME).unwrap();
    let result = function.call_packed(&[]).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 17);

    assert!(remove_global_for_test(NAME));
}

#[test]
fn test_function_get_global_cached_is_shared_across_threads() {
    let _registry_guard = lock_global_function_table_for_test();
    static CACHE: std::sync::OnceLock<Function> = std::sync::OnceLock::new();

    const NAME: &str = "testing.rust.get_global_cached.shared";
    remove_global_for_test(NAME);
    Function::register_global(NAME, Function::from_typed(|| -> Result<i32> { Ok(1) })).unwrap();

    let function = Function::get_global_cached(&CACHE, NAME).unwrap();
    assert_eq!(
        i32::try_from(function.call_packed(&[]).unwrap()).unwrap(),
        1
    );
    assert!(remove_global_for_test(NAME));
    Function::register_global(NAME, Function::from_typed(|| -> Result<i32> { Ok(2) })).unwrap();

    // The cached Function is Send + Sync and shared by every generated call
    // site, so another thread observes the same successful lookup.
    let other_thread = std::thread::spawn(move || {
        let function = Function::get_global_cached(&CACHE, NAME).unwrap();
        i32::try_from(function.call_packed(&[]).unwrap()).unwrap()
    });
    assert_eq!(other_thread.join().unwrap(), 1);

    assert!(remove_global_for_test(NAME));
}

#[test]
fn test_function_from_type_method_finds_auto_init_type_attr() {
    // SAFETY: the byte array is used only while this string literal is alive.
    let type_key =
        unsafe { tvm_ffi_sys::TVMFFIByteArray::from_str("testing.TestCxxAutoInitSimple") };
    let mut type_index = -1;
    let result = unsafe { tvm_ffi_sys::TVMFFITypeKeyToIndex(&type_key, &mut type_index) };
    assert_eq!(result, 0);

    // This fixture intentionally has no explicit refl::init TypeMethod. Its
    // __ffi_init__ is created at runtime and exists only in a type-attr column.
    let init = Function::from_type_method(type_index, "__ffi_init__").unwrap();
    let x = 10_i64;
    let y = 20_i64;
    let object = init
        .call_packed(&[AnyView::from(&x), AnyView::from(&y)])
        .unwrap();
    assert_eq!(object.type_index(), type_index);
}

#[test]
fn test_function_from_packed() {
    let value = 2;
    let v2 = 4;
    let check_and_add_value = Function::from_packed(move |args: &[AnyView]| -> Result<Any> {
        ensure!(
            args.len() == 1,
            VALUE_ERROR,
            "Expected 1 argument, got {}",
            args.len()
        );
        let v0 = i32::try_from(args[0])?;
        ensure!(v0 == value, VALUE_ERROR, "Expected {}, got {}", value, v0);
        Ok(Any::from(v0 + v2))
    });
    let args = [AnyView::from(&value)];
    let result = check_and_add_value.call_packed(&args).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 6);
}

#[test]
fn test_function_from_typed() {
    let offset = 2;
    // test one argument
    let sum1 = Function::from_typed(move |x: i32| -> Result<i32> { Ok(x + offset) });
    let result = sum1.call_packed(&[AnyView::from(&1)]).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 1 + offset);
    // test two arguments
    let sum2 = Function::from_typed(move |x: i32, y: i32| -> Result<i32> { Ok(x + y) });
    let result = sum2
        .call_packed(&[AnyView::from(&1), AnyView::from(&2)])
        .unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 3);
    // test three arguments
    let sum3f = Function::from_typed(move |x: i32, y: i32, z: f32| -> Result<f32> {
        Ok((x + y) as f32 + z)
    });
    let result = sum3f
        .call_packed(&[AnyView::from(&1), AnyView::from(&2), AnyView::from(&3)])
        .unwrap();
    assert_eq!(f32::try_from(result).unwrap(), 6.0);
}

#[test]
fn test_function_call_tuple() {
    let offset = 2;
    // test one argument
    let sum1 = Function::from_typed(move |x: i32| -> Result<i32> { Ok(x + offset) });
    let result = sum1.call_tuple((1,)).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 1 + offset);
    // test pass by reference
    let result = sum1.call_tuple_with_len::<1, _>((&1,)).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 1 + offset);
    let typed_fn = |x: &i32| -> Result<i32> { Ok(sum1.call_tuple((x,))?.try_into()?) };
    let result = typed_fn(&1);
    assert_eq!(result.unwrap(), 1 + offset);
}

#[test]
fn test_function_into_typed_fn() {
    let offset = 2;
    let typed_sum1 = into_typed_fn!(
        Function::from_typed(move |x: i32| -> Result<i32> { Ok(x + offset) }),
        Fn(&i32) -> Result<i32>);
    assert_eq!(typed_sum1(&1).unwrap(), 1 + offset);
    // try to box the resulting function
    let sum2 = Function::from_typed(move |x: i32, y: i32| -> Result<i32> { Ok(x + y) });
    let typed_sum2 = Box::new(into_typed_fn!(sum2, Fn(&i32, i32) -> Result<i32>));
    assert_eq!(typed_sum2(&1, 2).unwrap(), 3);

    // test three arguments
    let sum3 = Function::from_typed(move |x: i32, y: i32, z: f32| -> Result<f32> {
        Ok((x + y) as f32 + z)
    });
    let typed_sum3 = Box::new(into_typed_fn!(sum3, Fn(&i32, i32, f32) -> Result<f32>));
    assert_eq!(typed_sum3(&1, 2, 3.0).unwrap(), 6.0);
}

#[test]
fn test_function_echo_tensor_typed() {
    let _registry_guard = lock_global_function_table_for_test();
    let echo = into_typed_fn!(
        Function::get_global("testing.echo").unwrap(),
        Fn(&Tensor) -> Result<Tensor>
    );
    let data: &[f32] = &[1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_slice(data, &[1, 2, 2]).unwrap();
    // write tensor content here
    let result = echo(&tensor).unwrap();
    assert_eq!(result.data_ptr(), tensor.data_ptr());
    assert_eq!(AnyView::from(&result).debug_strong_count(), Some(2));
    let result_data = result.data_as_slice::<f32>().unwrap();
    assert_eq!(result_data.len(), 4);
    assert_eq!(result_data[0], 1.0);
    assert_eq!(result_data[1], 2.0);
    assert_eq!(result_data[2], 3.0);
    assert_eq!(result_data[3], 4.0);
}

fn testing_add_one(x: i32) -> Result<i32> {
    Ok(x + 1)
}
tvm_ffi_dll_export_typed_func!(testing_add_one, testing_add_one);

#[test]
fn test_function_from_extern_c() {
    // SAFETY: null handle is valid for testing_add_one which doesn't use the handle.
    let add_one =
        unsafe { Function::from_extern_c(std::ptr::null_mut(), __tvm_ffi_testing_add_one, None) };
    let typed_add_one = into_typed_fn!(add_one, Fn(i32) -> Result<i32>);
    assert_eq!(typed_add_one(1).unwrap(), 2);
}

#[test]
fn test_function_echo_string_bytes() {
    let _registry_guard = lock_global_function_table_for_test();
    let echo = Function::get_global("testing.echo").unwrap();
    let echo_str = into_typed_fn!(
        echo.clone(),
        Fn(&str) -> Result<String>
    );
    let result = echo_str("hello").unwrap();
    assert_eq!(result, "hello");
    let echo_bytes = into_typed_fn!(
        echo.clone(),
        Fn(&[u8]) -> Result<Bytes>
    );
    let result = echo_bytes(b"hello").unwrap();
    assert_eq!(result, b"hello");
}

#[test]
fn test_function_apply() {
    let _registry_guard = lock_global_function_table_for_test();
    let add_one = Function::from_typed(|x: i32| -> Result<i32> { Ok(x + 1) });
    let fapply = into_typed_fn!(
        Function::get_global("testing.apply").unwrap(),
        Fn(Function, i32) -> Result<i32>
    );
    let result = fapply(add_one, 3).unwrap();
    assert_eq!(result, 4);
}

fn test_add_one_tensor(x: tvm_ffi::Tensor, y: tvm_ffi::Tensor) -> Result<()> {
    let x_data = x.data_as_slice::<f32>()?;
    let y_data = y.data_as_slice_mut::<f32>()?;
    for i in 0..x_data.len() {
        y_data[i] = x_data[i] + 1.0;
    }
    Ok(())
}

tvm_ffi_dll_export_typed_func!(test_add_one_tensor, test_add_one_tensor);

#[test]
fn test_function_call_tensor_fn() {
    // SAFETY: null handle is valid for test_add_one_tensor which doesn't use the handle.
    let add_one = unsafe {
        Function::from_extern_c(std::ptr::null_mut(), __tvm_ffi_test_add_one_tensor, None)
    };
    let typed_add_one = into_typed_fn!(add_one, Fn(&Tensor, &Tensor) -> Result<()>);
    let x_data: &[f32] = &[0.0, 1.0, 2.0, 3.0];
    let x = Tensor::from_slice(x_data, &[2, 2]).unwrap();
    let y_data: &[f32] = &[0.0, 0.0, 0.0, 0.0];
    let y = Tensor::from_slice(y_data, &[2, 2]).unwrap();
    typed_add_one(&x, &y).unwrap();
    let y_data = y.data_as_slice::<f32>().unwrap();
    assert_eq!(y_data[0], 1.0);
    assert_eq!(y_data[1], 2.0);
    assert_eq!(y_data[2], 3.0);
    assert_eq!(y_data[3], 4.0);
}
