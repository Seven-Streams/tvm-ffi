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
//! Layout-mirror for C++ [`ffi::Optional<T>`] struct fields.
//!
//! [`Optional<T, A, N>`] occupies the **same size and alignment** as a C++
//! `ffi::Optional<T>` so a `#[repr(C)]` Rust mirror of a reflected object can
//! embed an optional field at the correct offset. It is **view-only**: it is
//! never constructed, owned, mutated, or dropped by Rust — the object always
//! lives on (and is created/destroyed by) the C++ side, and Rust only ever sees
//! it through a borrow into a live object.
//!
//! All access is delegated to C++ by reusing the reflection field
//! getter/setter (see [`resolve_field`]); reads/writes marshal through a native
//! `Option<V>` at the boundary. This is "method (b)" from the design doc.
//!
//! Layout facts (must come from the C++ reflection registry, never guessed):
//!
//! | C++ `Optional<T>` category        | size / align (x86_64, libstdc++) |
//! |-----------------------------------|----------------------------------|
//! | ObjectRef (`Optional<Array>` ...) | 8 / 8                            |
//! | `String` / `Bytes`                | 16 / 8                          |
//! | `std::optional<scalar>` fallback  | implementation-defined (e.g. i64 → 16/8, i32 → 8/4, DataType → 6/2) |
//!
//! Because Rust cannot parameterize `#[repr(align(N))]` by a const generic, the
//! alignment is carried by a zero-sized marker type `A` (one of [`Align1`] ..
//! [`Align16`]); the size is the const generic `N`.

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::marker::PhantomData;

use crate::any::Any;
use crate::error::{Error, Result, TYPE_ERROR};
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIFieldGetter, TVMFFIFieldSetter, TVMFFIGetTypeInfo};

// --- alignment marker ZSTs (carry alignment; bypass const-generic align limitation) ---
macro_rules! align_marker {
    ($name:ident, $n:literal) => {
        #[doc = concat!("Zero-sized alignment marker for ", stringify!($n), "-byte alignment.")]
        #[repr(align($n))]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;
    };
}
align_marker!(Align1, 1);
align_marker!(Align2, 2);
align_marker!(Align4, 4);
align_marker!(Align8, 8);
align_marker!(Align16, 16);

/// Opaque, view-only mirror of a C++ `ffi::Optional<T>` struct field.
///
/// * `T` — logical value type (for documentation and `Send` propagation only).
/// * `A` — alignment marker (one of [`Align1`] ..= [`Align16`]).
/// * `N` — byte size, equal to `sizeof(ffi::Optional<T>)` from the reflection registry.
///
/// The storage is private and wrapped in `UnsafeCell`, so:
/// * Rust cannot construct or directly read/write the bytes (view-only); and
/// * a C++ setter invoked through a shared `&self` is sound (interior mutability).
///
/// Auto traits: always `!Sync` (via `UnsafeCell`); `Send` follows `T`.
/// No `Drop`/`Clone`/`Copy`: destruction and duplication belong to C++.
#[repr(C)]
#[allow(dead_code)]
pub struct Optional<T, A, const N: usize> {
    _align: A,
    bytes: UnsafeCell<[u8; N]>,
    _marker: PhantomData<T>,
}

impl<T, A, const N: usize> Optional<T, A, N> {
    /// Raw address of this optional's storage — i.e. a pointer to the in-place
    /// C++ `ffi::Optional<T>`. Valid through `&self` because the storage is an
    /// `UnsafeCell`.
    #[inline]
    pub fn as_ffi_ptr(&self) -> *mut c_void {
        self.bytes.get() as *mut c_void
    }

    /// Read this optional in place via its reflection getter, returning a native
    /// `Option<V>` (a `None` C++ optional maps to `None`).
    ///
    /// The getter hands back an **owned** value (it already `inc_ref`-ed via the
    /// `Any(*field)` copy on the C++ side), so we take it by move with no extra
    /// `inc_ref`; the returned `Option<V>` owns that reference.
    ///
    /// # Safety
    /// `getter` must be the reflection getter registered for *this* field of the
    /// enclosing object, and `self` must point into a live object of that type.
    pub(crate) unsafe fn read_with_getter<V: AnyCompatible>(
        &self,
        getter: TVMFFIFieldGetter,
    ) -> Result<Option<V>> {
        let mut out = TVMFFIAny::new();
        crate::check_safe_call!(getter(self.as_ffi_ptr(), &mut out))?;
        // `out` owns one reference. `TryFrom<Any> for Option<V>` consumes it via
        // the move-path on a strict match (None-any -> None, value-any -> Some).
        let any = Any::from_raw_ffi_any(out);
        Option::<V>::try_from(any)
    }

    /// Write this optional in place via its reflection setter, from a native
    /// `Option<V>` (`None` writes a C++ `nullopt`). The C++ setter destructs the
    /// previous value, so the field must already be constructed (it always is —
    /// the object was created by C++).
    ///
    /// # Safety
    /// `setter` must be the reflection setter registered for *this* field, and
    /// `self` must point into a live object of that type.
    pub(crate) unsafe fn write_with_setter<V: AnyCompatible>(
        &self,
        setter: TVMFFIFieldSetter,
        value: Option<V>,
    ) -> Result<()> {
        let mut any = Any::from(value);
        crate::check_safe_call!(setter(self.as_ffi_ptr(), Any::as_data_ptr(&mut any)))?;
        Ok(())
    }

    /// Convenience read that resolves the getter from the enclosing object's
    /// reflection table by `field_name`. Prefer caching [`resolve_field`] in the
    /// generated accessor for hot paths.
    ///
    /// # Safety
    /// `self` must be the `field_name` field of a live object whose type index is
    /// `parent_type_index`.
    pub unsafe fn read<V: AnyCompatible>(
        &self,
        parent_type_index: i32,
        field_name: &str,
    ) -> Result<Option<V>> {
        let fa = resolve_field(parent_type_index, field_name)?;
        self.read_with_getter(fa.getter)
    }

    /// Convenience write counterpart of [`read`](Self::read).
    ///
    /// # Safety
    /// Same as [`read`](Self::read).
    pub unsafe fn write<V: AnyCompatible>(
        &self,
        parent_type_index: i32,
        field_name: &str,
        value: Option<V>,
    ) -> Result<()> {
        let fa = resolve_field(parent_type_index, field_name)?;
        let setter = fa.setter.ok_or_else(|| {
            Error::new(
                TYPE_ERROR,
                &format!("field `{field_name}` has no plain (function-pointer) setter"),
                "",
            )
        })?;
        self.write_with_setter(setter, value)
    }

    /// Cached read used by generated accessors: resolves the field once via
    /// `cell` (a per-call-site `OnceLock`), then reads. Avoids the per-call
    /// field-table scan that [`read`](Self::read) does.
    ///
    /// # Safety
    /// Same as [`read`](Self::read).
    pub unsafe fn read_cached<V: AnyCompatible>(
        &self,
        cell: &std::sync::OnceLock<FieldAccess>,
        parent_type_index: i32,
        field_name: &str,
    ) -> Result<Option<V>> {
        let fa = resolve_field_cached(cell, parent_type_index, field_name)?;
        self.read_with_getter(fa.getter)
    }

    /// Cached write counterpart of [`read_cached`](Self::read_cached).
    ///
    /// # Safety
    /// Same as [`read`](Self::read).
    pub unsafe fn write_cached<V: AnyCompatible>(
        &self,
        cell: &std::sync::OnceLock<FieldAccess>,
        parent_type_index: i32,
        field_name: &str,
        value: Option<V>,
    ) -> Result<()> {
        let fa = resolve_field_cached(cell, parent_type_index, field_name)?;
        let setter = fa.setter.ok_or_else(|| {
            Error::new(
                TYPE_ERROR,
                &format!("field `{field_name}` has no plain (function-pointer) setter"),
                "",
            )
        })?;
        self.write_with_setter(setter, value)
    }
}

impl<T, A, const N: usize> std::fmt::Debug for Optional<T, A, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ffi::Optional<{}>(opaque, {} bytes)",
            std::any::type_name::<T>(),
            N
        )
    }
}

// --- reflection field resolution ---

const FIELD_FLAG_SETTER_IS_FUNCTION_OBJ: i64 = 1 << 11;

/// Resolved reflection access info for one field, looked up by name.
#[derive(Clone, Copy)]
pub struct FieldAccess {
    /// In-place getter: `(field_addr, *mut TVMFFIAny) -> status`.
    pub getter: TVMFFIFieldGetter,
    /// In-place setter, when stored as a plain function pointer (the default).
    /// `None` if the field exposes no setter or uses a function-object setter.
    pub setter: Option<TVMFFIFieldSetter>,
}

/// Look up a field's reflection access info by name on the given type index.
///
/// This walks `TVMFFIGetTypeInfo(type_index).fields`. The result is `Copy` and
/// holds raw function pointers, so callers may cache it (e.g. in a `OnceLock`).
pub fn resolve_field(type_index: i32, name: &str) -> Result<FieldAccess> {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            return Err(Error::new(
                TYPE_ERROR,
                &format!("no type info registered for type_index {type_index}"),
                "",
            ));
        }
        let info = &*info;
        // Index with `.add(i)` rather than `from_raw_parts`, which is UB on a null
        // `fields` pointer even when `num_fields == 0`.
        for i in 0..info.num_fields {
            let f = &*info.fields.add(i as usize);
            if f.name.as_str() != name {
                continue;
            }
            let getter = f.getter.ok_or_else(|| {
                Error::new(TYPE_ERROR, &format!("field `{name}` has no getter"), "")
            })?;
            let setter = if f.setter.is_null() || (f.flags & FIELD_FLAG_SETTER_IS_FUNCTION_OBJ) != 0
            {
                None
            } else {
                // The default ABI stores the setter as a TVMFFIFieldSetter function
                // pointer cast to *mut c_void; reverse that cast.
                Some(std::mem::transmute::<*mut c_void, TVMFFIFieldSetter>(
                    f.setter,
                ))
            };
            return Ok(FieldAccess { getter, setter });
        }
    }
    Err(Error::new(
        TYPE_ERROR,
        &format!("type_index {type_index} has no field named `{name}`"),
        "",
    ))
}

/// Cached front of [`resolve_field`] for generated accessors: the field-table
/// scan runs once, then the `Copy` [`FieldAccess`] is reused. `FieldAccess`
/// holds only function pointers (so it is `Sync`), so a process-wide `OnceLock`
/// suffices — no per-thread re-scan (unlike `Function::from_type_method_cached`).
pub(crate) fn resolve_field_cached(
    cell: &std::sync::OnceLock<FieldAccess>,
    type_index: i32,
    name: &str,
) -> Result<FieldAccess> {
    if let Some(fa) = cell.get() {
        return Ok(*fa);
    }
    let fa = resolve_field(type_index, name)?;
    let _ = cell.set(fa);
    Ok(fa)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `Align_k` marker + `[u8; N]` storage must yield exactly (size N, align
    // k) for each distinct (align, size) pair; layout is independent of marker T
    // (measured C++ values in plan.md).
    #[test]
    fn layout_matches_cpp_categories() {
        macro_rules! check {
            ($t:ty, $a:ty, $n:literal, $sz:literal, $al:literal) => {{
                assert_eq!(std::mem::size_of::<Optional<$t, $a, $n>>(), $sz);
                assert_eq!(std::mem::align_of::<Optional<$t, $a, $n>>(), $al);
            }};
        }
        check!(i64, Align8, 16, 16, 8); // Optional<i64>/<f64>/<String> ...
        check!(i32, Align4, 8, 8, 4); // Optional<i32>
        check!(u8, Align1, 2, 2, 1); // Optional<bool>
        check!(u32, Align4, 12, 12, 4); // Optional<DLDevice>
        check!(u16, Align2, 6, 6, 2); // Optional<DLDataType>
        check!(*mut c_void, Align8, 8, 8, 8); // Optional<ObjectRef>
    }

    // An Optional field embedded in a #[repr(C)] mirror must land at the offset C
    // gives, and must not perturb the trailing field.
    #[test]
    fn embeds_in_repr_c_struct() {
        #[repr(C)]
        #[allow(dead_code)]
        struct Mirror {
            head: u64,
            opt: Optional<i64, Align8, 16>,
            tail: u32,
        }
        assert_eq!(std::mem::offset_of!(Mirror, opt), 8);
        assert_eq!(std::mem::offset_of!(Mirror, tail), 24);
        assert_eq!(std::mem::align_of::<Mirror>(), 8);
    }
}
