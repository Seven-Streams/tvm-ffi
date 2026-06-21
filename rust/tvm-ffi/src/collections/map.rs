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
//! Read-only binding to the C++ `ffi::Map` container.
//!
//! Unlike [`Array`](super::array::Array), a `Map`'s storage is a hash table with
//! a layout (see `map_base.h`) that Rust does not replicate. The binding treats
//! the [`MapObj`] body as opaque and performs every operation through the global
//! reflection functions registered in `src/ffi/container.cc` (`ffi.Map`,
//! `ffi.MapSize`, `ffi.MapGetItem`, `ffi.MapCount`, `ffi.MapForwardIterFunctor`),
//! mirroring the Python binding in `python/tvm_ffi/container.py`.
//!
//! The C++ `Map` is copy-on-write immutable, so this binding is read-only:
//! mutating it would copy into a fresh map rather than update the shared object.
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::any::TryFromTemp;
use crate::derive::Object;
use crate::function::Function;
use crate::object::{Object, ObjectArc};
use crate::{Any, AnyCompatible, AnyView, ObjectRefCore};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIObject};

/// Container object behind a [`Map`].
///
/// Only the object header is modeled; the C++ `MapObj` continues with
/// hash-table state (`map_base.h`) that this binding never reads. A `MapObj` is
/// always allocated on the C++ side -- Rust only ever wraps a returned handle,
/// so no Rust-side allocation path is provided.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Map"]
#[type_index(TypeIndex::kTVMFFIMap)]
pub struct MapObj {
    pub object: Object,
}

/// A read-only view of a C++ `ffi::Map<K, V>`.
///
/// `Map` is a reference-counted handle to a shared map object; cloning it shares
/// the same underlying map (it does not deep-copy). Element types are checked
/// lazily: a `Map<K, V>` accepts any map-typed [`Any`], and a wrong `K`/`V` only
/// surfaces as a conversion error from [`get`](Map::get) / [`keys`](Map::keys) /
/// [`values`](Map::values) / [`items`](Map::items).
#[repr(C)]
pub struct Map<K, V> {
    data: ObjectArc<MapObj>,
    _marker: PhantomData<(K, V)>,
}

impl<K, V> Clone for Map<K, V> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _marker: PhantomData,
        }
    }
}

impl<K, V> Debug for Map<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn short(full: &str) -> &str {
            full.split("::").last().unwrap_or(full)
        }
        write!(
            f,
            "Map<{}, {}>",
            short(std::any::type_name::<K>()),
            short(std::any::type_name::<V>())
        )
    }
}

unsafe impl<K, V> ObjectRefCore for Map<K, V> {
    type ContainerType = MapObj;

    fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
        &this.data
    }

    fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
        this.data
    }

    fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }
}

/// Move an owned [`Any`] into a concrete `AnyCompatible` value.
///
/// Goes through [`TryFromTemp`], which is the only generic `TryFrom<Any>` path
/// (the concrete `TryFrom<Any>` impls are per-type), so it works for any
/// element type `T` a `Map` may hold.
fn any_into<T: AnyCompatible>(value: Any) -> crate::Result<T> {
    let temp: TryFromTemp<T> = TryFromTemp::try_from(value)?;
    Ok(TryFromTemp::into_value(temp))
}

/// Value types a [`Map`] can yield from its accessors.
///
/// Implemented for every `V: AnyCompatible` (converted to that concrete type)
/// and for [`Any`] itself (returned opaque, to be downcast), so a `Map<K, Any>`
/// -- e.g. the `DictAttrs.__dict__ : Map<str, Any>` IR keystone -- is readable,
/// even though `Any` is not `AnyCompatible`. Mirrors [`crate::ArrayElement`].
/// Keys stay `AnyCompatible` (you look up by a concrete key). Not for hand impl.
pub trait MapValue: Sized {
    /// Convert a value read out of the map (always delivered as an [`Any`]) into
    /// the requested value type.
    #[doc(hidden)]
    fn map_value_from_any(value: Any) -> crate::Result<Self>;
}

impl<V: AnyCompatible> MapValue for V {
    #[inline]
    fn map_value_from_any(value: Any) -> crate::Result<Self> {
        any_into(value)
    }
}

impl MapValue for Any {
    #[inline]
    fn map_value_from_any(value: Any) -> crate::Result<Self> {
        Ok(value)
    }
}

impl<K: AnyCompatible, V: MapValue> Map<K, V> {
    /// Builds a map from key/value pairs by calling the global `ffi.Map`
    /// constructor (packed as `k0, v0, k1, v1, ...`). Later duplicate keys win,
    /// matching the C++ constructor. Construction needs `V: AnyCompatible` (the
    /// values are marshalled in); reading back only needs `V: MapValue`.
    pub fn from_pairs<I>(pairs: I) -> crate::Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        V: AnyCompatible,
    {
        // Keep the keys/values alive while their non-owning `AnyView`s are in the
        // packed-args slice; `ffi.Map` copies (inc-refs) them into the new map.
        let pairs: Vec<(K, V)> = pairs.into_iter().collect();
        let mut views: Vec<AnyView> = Vec::with_capacity(pairs.len() * 2);
        for (k, v) in &pairs {
            views.push(AnyView::from(k));
            views.push(AnyView::from(v));
        }
        let any = Function::get_global("ffi.Map")?.call_packed(&views)?;
        any_into(any)
    }

    /// Returns the number of entries.
    pub fn len(&self) -> crate::Result<usize> {
        Ok(any_into::<i64>(self.invoke("ffi.MapSize", None)?)? as usize)
    }

    /// Returns `true` when the map has no entries.
    pub fn is_empty(&self) -> crate::Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Returns `true` when `key` is present.
    pub fn contains_key(&self, key: &K) -> crate::Result<bool> {
        Ok(any_into::<i64>(self.invoke("ffi.MapCount", Some(key))?)? != 0)
    }

    /// Returns the value for `key`, or `None` when it is absent.
    ///
    /// Uses `ffi.MapCount` + `ffi.MapGetItem` so absence is reported as `None`
    /// rather than surfacing the C++ `MISSING` sentinel.
    pub fn get(&self, key: &K) -> crate::Result<Option<V>> {
        if any_into::<i64>(self.invoke("ffi.MapCount", Some(key))?)? == 0 {
            return Ok(None);
        }
        Ok(Some(V::map_value_from_any(self.invoke("ffi.MapGetItem", Some(key))?)?))
    }

    /// Collects the keys (iteration order is the C++ map's, which is unspecified).
    pub fn keys(&self) -> crate::Result<Vec<K>> {
        self.collect_entries(|f| any_into::<K>(Self::iter_step(f, 0)?))
    }

    /// Collects the values (same unspecified order as [`keys`](Map::keys)).
    pub fn values(&self) -> crate::Result<Vec<V>> {
        self.collect_entries(|f| V::map_value_from_any(Self::iter_step(f, 1)?))
    }

    /// Collects the `(key, value)` pairs.
    pub fn items(&self) -> crate::Result<Vec<(K, V)>> {
        self.collect_entries(|f| {
            let k = any_into::<K>(Self::iter_step(f, 0)?)?;
            let v = V::map_value_from_any(Self::iter_step(f, 1)?)?;
            Ok((k, v))
        })
    }

    /// Calls a global reflection function with `self` (and an optional key) as
    /// the leading packed argument(s).
    fn invoke(&self, name: &str, key: Option<&K>) -> crate::Result<Any> {
        let f = Function::get_global(name)?;
        // View the map by raw object pointer (it is a single `kTVMFFIMap` pointer
        // regardless of K/V), so `Map<_, Any>` -- where `Map` is not itself
        // `AnyCompatible` -- can still pass itself as the leading argument. The
        // borrow of `self` keeps the object alive for the view's lifetime.
        let self_view = unsafe {
            AnyView::from_raw_object(
                TypeIndex::kTVMFFIMap as i32,
                ObjectArc::as_raw(&self.data) as *mut TVMFFIObject,
            )
        };
        match key {
            Some(k) => f.call_packed(&[self_view, AnyView::from(k)]),
            None => f.call_packed(&[self_view]),
        }
    }

    /// Drives the stateful `ffi.MapForwardIterFunctor`, calling `read` once per
    /// entry. The functor starts on the first entry; reading past `len()` would
    /// dereference the end iterator, so the loop runs exactly `len()` times and
    /// only advances *between* entries.
    fn collect_entries<T, F>(&self, mut read: F) -> crate::Result<Vec<T>>
    where
        F: FnMut(&Function) -> crate::Result<T>,
    {
        let n = self.len()?;
        let mut out = Vec::with_capacity(n);
        if n == 0 {
            return Ok(out);
        }
        let functor: Function = any_into(self.invoke("ffi.MapForwardIterFunctor", None)?)?;
        for i in 0..n {
            out.push(read(&functor)?);
            if i + 1 < n {
                // Command 2 advances and returns whether a next entry exists; we
                // stop before the final advance, so the result is always `true`.
                let _: bool = any_into(Self::iter_step(&functor, 2)?)?;
            }
        }
        Ok(out)
    }

    /// Invokes the iterator functor with a command: 0 = current key, 1 = current
    /// value, 2 = advance.
    fn iter_step(functor: &Function, command: i64) -> crate::Result<Any> {
        functor.call_packed(&[AnyView::from(&command)])
    }
}

// --- Any Type System Conversions ---

unsafe impl<K, V> AnyCompatible for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    fn type_str() -> String {
        format!("Map<{}, {}>", K::type_str(), V::type_str())
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        // Container-level check only. The C++ map is a heterogeneous Any->Any
        // store, so element types are not re-validated here (a read-only handle
        // checks them lazily on access); this keeps casts O(1).
        data.type_index == TypeIndex::kTVMFFIMap as i32
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIMap as i32;
        data.data_union.v_obj = ObjectArc::as_raw(Self::data(src)) as *mut TVMFFIObject;
        data.small_str_len = 0;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIMap as i32;
        data.data_union.v_obj = ObjectArc::into_raw(Self::into_data(src)) as *mut TVMFFIObject;
        data.small_str_len = 0;
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const MapObj;
        crate::object::unsafe_::inc_ref(ptr as *mut TVMFFIObject);
        Self::from_data(ObjectArc::from_raw(ptr))
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const MapObj;
        let obj = Self::from_data(ObjectArc::from_raw(ptr));

        data.type_index = TypeIndex::kTVMFFINone as i32;
        data.data_union.v_int64 = 0;

        obj
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIMap as i32 {
            Ok(Self::copy_from_any_view_after_check(data))
        } else {
            Err(())
        }
    }
}

impl<K, V> TryFrom<Any> for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    type Error = crate::error::Error;

    fn try_from(value: Any) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}

impl<'a, K, V> TryFrom<AnyView<'a>> for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    type Error = crate::error::Error;

    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}
