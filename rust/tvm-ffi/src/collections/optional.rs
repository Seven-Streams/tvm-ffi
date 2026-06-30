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
//! In-place `Optional<T>` mirror of C++ `ffi::Optional<T>` for POD `T`.
//!
//! This is the Rust counterpart of the `std::optional`-backed
//! `ffi::Optional<T>` specialization (`include/tvm/ffi/optional.h`, the
//! non-`ObjectRef` / non-`String` / non-`Bytes` fallback). It is
//! **memory-layout-compatible** with that C++ type: a `#[repr(C)]`
//! `{ value: T, engaged: bool }` reproduces, byte for byte, the
//! `std::optional<T>` layout used by libstdc++, libc++ and MSVC STL — the
//! payload at offset `0`, the engaged flag at offset `size_of::<T>()`, padded
//! to `align_of::<T>()`. This was verified against real `ffi::Optional<T>`
//! instances under both libstdc++ and libc++.
//!
//! Because the layout is known, values are decoded **in place**:
//! [`Optional::get`] reads the flag and payload directly from the field bytes
//! with no FFI call and no allocation, and [`Optional::set`] writes them back.
//! This is the in-place-decode design, in contrast to routing every access
//! through the reflection getter/setter.
//!
//! # Scope
//! Only the `std::optional` fallback specialization is mirrored here, i.e. POD
//! `T` that is neither an `ObjectRef`, a [`String`](crate::String), nor
//! [`Bytes`](crate::Bytes):
//! - `ffi::Optional<SomeRef>` is a nullable object pointer — use
//!   `Option<SomeRef>` (`nullptr` == `None`) instead.
//! - `ffi::Optional<String>` / `ffi::Optional<Bytes>` carry a null sentinel
//!   inside the string container and are out of scope for this module.
//!
//! # Thread-safety
//! [`Optional<T>`] uses interior mutability so a shared `&Optional<T>` aliasing
//! a C++-owned field can be written in place; it is therefore `!Sync`.

use std::cell::UnsafeCell;
use std::fmt::{self, Debug};
use std::mem::MaybeUninit;

/// Marker for POD `T` whose C++ `ffi::Optional<T>` uses the `std::optional`
/// fallback with a representation that byte-matches `T` in Rust.
///
/// # Safety
/// Implementing this asserts that, for the reflected C++ field of type
/// `ffi::Optional<T>`:
/// - the C++ side selects the `std::optional<T>` fallback (i.e. `T` is not an
///   `ObjectRef`, `String`, or `Bytes`),
/// - `T` is trivially copyable with no destructor, and
/// - `T`'s Rust in-memory representation is identical to the C++ field type
///   (e.g. `i32` ↔ `int32_t`, `f64` ↔ `double`, `bool` ↔ `bool`).
///
/// Sealed via the private `Sealed` supertrait: only the scalar set in
/// `optional_pod!` below can implement it, so the contract can't break downstream.
pub unsafe trait OptionalPod: Copy + private::Sealed {}

mod private {
    /// Seals `OptionalPod`: unreachable outside this module.
    pub trait Sealed {}
}

/// Layout-mirror of `std::optional<T>`: `{ T value @0; bool engaged @sizeof(T) }`.
#[repr(C)]
struct OptionalCell<T: OptionalPod> {
    value: MaybeUninit<T>,
    engaged: bool,
}

/// In-place mirror of C++ `ffi::Optional<T>` for POD `T`.
///
/// Layout-compatible with the C++ type; see the [module docs](self).
#[repr(transparent)]
pub struct Optional<T: OptionalPod> {
    cell: UnsafeCell<OptionalCell<T>>,
}

impl<T: OptionalPod> Optional<T> {
    /// Builds an engaged optional holding `value`.
    #[inline]
    pub fn some(value: T) -> Self {
        // Only payload+flag are written; padding isn't part of the ABI.
        Self {
            cell: UnsafeCell::new(OptionalCell {
                value: MaybeUninit::new(value),
                engaged: true,
            }),
        }
    }

    /// Builds a disengaged optional (`nullopt`).
    #[inline]
    pub fn none() -> Self {
        // Zeroed (not `uninit`) payload keeps the byte-image tests reading init bytes.
        Self {
            cell: UnsafeCell::new(OptionalCell {
                value: MaybeUninit::zeroed(),
                engaged: false,
            }),
        }
    }

    /// Decodes the value in place. No FFI call, no allocation.
    #[inline]
    pub fn get(&self) -> Option<T> {
        // SAFETY: the cell is always fully initialized; the payload is only read
        // after confirming `engaged`, matching C++ `has_value()` gating.
        let cell = unsafe { &*self.cell.get() };
        if cell.engaged {
            Some(unsafe { cell.value.assume_init() })
        } else {
            None
        }
    }

    /// Returns whether a value is present.
    #[inline]
    pub fn has_value(&self) -> bool {
        // SAFETY: `engaged` is always initialized.
        unsafe { (*self.cell.get()).engaged }
    }

    /// Returns whether the optional is `nullopt`.
    #[inline]
    pub fn is_none(&self) -> bool {
        !self.has_value()
    }

    /// Overwrites the value in place through a shared reference.
    ///
    /// Mirrors C++ assignment: `Some(v)` engages and stores `v`; `None`
    /// disengages without touching the payload bytes, as `std::optional::reset`
    /// does for trivial `T`.
    #[inline]
    pub fn set(&self, value: Option<T>) {
        // SAFETY: interior mutation through `UnsafeCell`; the caller must not
        // race (the type is `!Sync`).
        let cell = unsafe { &mut *self.cell.get() };
        match value {
            Some(v) => {
                cell.value = MaybeUninit::new(v);
                cell.engaged = true;
            }
            None => cell.engaged = false,
        }
    }
}

impl<T: OptionalPod> Default for Optional<T> {
    /// `nullopt`, matching the C++ default constructor.
    #[inline]
    fn default() -> Self {
        Self::none()
    }
}

impl<T: OptionalPod> Clone for Optional<T> {
    #[inline]
    fn clone(&self) -> Self {
        match self.get() {
            Some(v) => Self::some(v),
            None => Self::none(),
        }
    }
}

impl<T: OptionalPod> From<Option<T>> for Optional<T> {
    #[inline]
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => Self::some(v),
            None => Self::none(),
        }
    }
}

impl<T: OptionalPod> From<Optional<T>> for Option<T> {
    #[inline]
    fn from(value: Optional<T>) -> Self {
        value.get()
    }
}

impl<T: OptionalPod + Debug> Debug for Optional<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => write!(f, "Optional::Some({v:?})"),
            None => f.write_str("Optional::None"),
        }
    }
}

// Registers each supported scalar from one list: the seal, the `OptionalPod` impl,
// and a compile-time guard that `Optional<T>` matches the `std::optional<T>`
// footprint (`size == round_up(size_of::<T>()+1, align)`). One list keeps the impl
// and its layout check from drifting.
macro_rules! optional_pod {
    ($($t:ty),* $(,)?) => { $(
        impl private::Sealed for $t {}
        // SAFETY: fixed-width scalar; repr matches the C++ field and uses the
        // `std::optional` fallback (layout proven by the `const` block below).
        unsafe impl OptionalPod for $t {}
        const _: () = {
            let tsz = core::mem::size_of::<$t>();
            let tal = core::mem::align_of::<$t>();
            let expect = (tsz + 1).div_ceil(tal) * tal;
            assert!(core::mem::align_of::<Optional<$t>>() == tal);
            assert!(core::mem::size_of::<Optional<$t>>() == expect);
        };
    )* };
}
optional_pod!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    /// `(size, align)` pairs verified against real `ffi::Optional<T>` instances
    /// (libstdc++ and libc++): `size = round_up(sizeof(T)+1, alignof(T))`.
    #[test]
    fn layout_matches_cpp_optional() {
        fn sa<T: OptionalPod>() -> (usize, usize) {
            (
                std::mem::size_of::<Optional<T>>(),
                std::mem::align_of::<Optional<T>>(),
            )
        }
        assert_eq!(sa::<i8>(), (2, 1));
        assert_eq!(sa::<bool>(), (2, 1));
        assert_eq!(sa::<i16>(), (4, 2));
        assert_eq!(sa::<i32>(), (8, 4));
        assert_eq!(sa::<f32>(), (8, 4));
        assert_eq!(sa::<i64>(), (16, 8));
        assert_eq!(sa::<f64>(), (16, 8));
    }

    /// Payload+flag bytes `[0, size_of::<T>()]`; padding is excluded (not ABI, not
    /// guaranteed initialized).
    fn image<T: OptionalPod>(opt: &Optional<T>) -> Vec<u8> {
        let p = opt as *const Optional<T> as *const u8;
        let n = std::mem::size_of::<T>() + 1; // payload + flag, no padding
        // SAFETY: payload and flag are always initialized; padding is not read.
        unsafe { std::slice::from_raw_parts(p, n).to_vec() }
    }

    /// The scalar's own bytes, to assert the optional's payload matches it verbatim.
    fn raw_bytes<T: OptionalPod>(v: &T) -> Vec<u8> {
        let p = v as *const T as *const u8;
        // SAFETY: size_of::<T>() bytes of an initialized `Copy` scalar.
        unsafe { std::slice::from_raw_parts(p, std::mem::size_of::<T>()).to_vec() }
    }

    #[test]
    fn byte_image_some_i32_matches_cpp() {
        // C++ probe (both STLs): some(0x12345678) => 78 56 34 12 | 01 ..
        let o = Optional::<i32>::some(0x1234_5678);
        let b = image(&o);
        assert_eq!(&b[0..4], &0x1234_5678_i32.to_le_bytes());
        assert_eq!(b[4], 1, "engaged flag must sit at offset size_of::<i32>()");
    }

    #[test]
    fn byte_image_some_i64_matches_cpp() {
        let o = Optional::<i64>::some(0x1122_3344_5566_7788);
        let b = image(&o);
        assert_eq!(&b[0..8], &0x1122_3344_5566_7788_i64.to_le_bytes());
        assert_eq!(b[8], 1, "engaged flag must sit at offset size_of::<i64>()");
    }

    #[test]
    fn byte_image_none_clears_flag() {
        let o = Optional::<i32>::none();
        let b = image(&o);
        assert_eq!(b[4], 0, "engaged flag must be clear for nullopt");
    }

    /// Payload@0 and flag@`size_of::<T>()` for every supported type, not just i32/i64.
    #[test]
    fn flag_offset_all_supported_types() {
        fn check<T: OptionalPod + PartialEq + std::fmt::Debug>(val: T) {
            let ty = std::any::type_name::<T>();
            let sz = std::mem::size_of::<T>();
            let some = Optional::<T>::some(val);
            let b = image(&some);
            assert_eq!(&b[0..sz], &raw_bytes(&val)[..], "payload@0 for {ty}");
            assert_eq!(b[sz], 1, "engaged flag must sit at offset size_of for {ty}");
            assert_eq!(some.get(), Some(val), "engaged roundtrip for {ty}");
            let none = Optional::<T>::none();
            assert_eq!(image(&none)[sz], 0, "flag must be clear for none for {ty}");
            assert!(none.get().is_none(), "disengaged roundtrip for {ty}");
        }
        check::<bool>(true);
        check::<i8>(0x12);
        check::<i16>(0x1234);
        check::<i32>(0x1234_5678);
        check::<i64>(0x1122_3344_5566_7788);
        check::<u8>(0xAB);
        check::<u16>(0xABCD);
        check::<u32>(0xABCD_EF01);
        check::<u64>(0xABCD_EF01_2345_6789);
        check::<f32>(1.5);
        check::<f64>(2.5);
    }

    #[test]
    fn roundtrip_get_set() {
        let o = Optional::<i32>::some(42);
        assert_eq!(o.get(), Some(42));
        assert!(o.has_value());

        // in-place mutation through a shared reference
        o.set(None);
        assert_eq!(o.get(), None);
        assert!(o.is_none());

        o.set(Some(-7));
        assert_eq!(o.get(), Some(-7));
    }

    #[test]
    fn conversions_and_default() {
        assert_eq!(Optional::<f64>::from(Some(2.5)).get(), Some(2.5));
        assert_eq!(Optional::<f64>::from(None).get(), None);
        let back: Option<i16> = Optional::<i16>::some(9).into();
        assert_eq!(back, Some(9));
        assert_eq!(Optional::<u8>::default().get(), None);
        assert_eq!(Optional::<bool>::some(true).get(), Some(true));
    }

    #[test]
    fn clone_preserves_state() {
        let some = Optional::<i64>::some(123);
        assert_eq!(some.clone().get(), Some(123));
        let none = Optional::<i64>::none();
        assert_eq!(none.clone().get(), None);
    }
}
