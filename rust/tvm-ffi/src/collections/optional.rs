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
//! In-place mirrors of C++ `ffi::Optional<T>` (`include/tvm/ffi/optional.h`) —
//! the specializations that store their value inline, so field bytes decode
//! directly (no FFI call, no alloc, no reflection getter/setter):
//! - [`Optional`] — POD scalar `T`. `#[repr(C)] { value: T, engaged: bool }`
//!   reproduces the `std::optional<T>` layout byte for byte (payload@0, flag@
//!   `size_of::<T>()`, padded to `align_of::<T>()`); verified vs libstdc++/libc++.
//! - [`OptionalStr`] — `String`: the 16-byte cell, `type_index == kTVMFFINone` =
//!   `nullopt` (C++ stores the sentinel in-cell). `Bytes` would follow the same.
//!
//! `ffi::Optional<SomeRef>` is a nullable object pointer, not stored inline — use
//! `Option<SomeRef>` (`nullptr` == `None`) instead.
//!
//! # Thread-safety
//! [`Optional`] is `!Sync`: interior mutability lets a shared `&Optional<T>`
//! aliasing a C++-owned field write in place (`set(&self)`). [`OptionalStr`]
//! instead uses `set(&mut self)` — it hands out `as_str` borrows into a
//! refcounted cell, so a shared-ref setter would dangle a live `&str`.

use crate::String;
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
pub unsafe trait OptionalPod: Copy {}

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

// Registers each supported scalar from one list: the `OptionalPod` impl and a
// compile-time guard that `Optional<T>` matches the `std::optional<T>` footprint
// (`size == round_up(size_of::<T>()+1, align)`). One list keeps the impl and its
// layout check from drifting.
macro_rules! optional_pod {
    ($($t:ty),* $(,)?) => { $(
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

/// In-place mirror of C++ `ffi::Optional<String>`: the 16-byte string cell
/// itself, with `type_index == kTVMFFINone` meaning `nullopt` (the C++
/// String/Bytes spec stores the sentinel in-cell, not as a separate flag).
/// Reuses [`String`]'s `Clone`/`Drop`, whose refcounting is a no-op on the
/// `nullopt` cell (`type_index` below `kTVMFFIStaticObjectBegin`).
#[repr(transparent)]
#[derive(Clone)]
pub struct OptionalStr {
    // Never handed out or accessed while disengaged (a `nullopt` cell is not a
    // valid string).
    inner: String,
}

// Must stay 16 bytes to overlay C++ `ffi::Optional<String>` (parity with the POD
// guard in `optional_pod!`).
const _: () = assert!(std::mem::size_of::<OptionalStr>() == 16);

impl OptionalStr {
    /// An engaged optional holding `value`.
    #[inline]
    pub fn some(value: String) -> Self {
        Self { inner: value }
    }

    /// A disengaged optional (`nullopt`).
    #[inline]
    pub fn none() -> Self {
        Self {
            inner: String::none_cell(),
        }
    }

    /// Whether a value is present.
    #[inline]
    pub fn has_value(&self) -> bool {
        !self.inner.is_none_cell()
    }

    /// Whether the optional is `nullopt`.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.inner.is_none_cell()
    }

    /// Borrows the engaged string as `&str`, or `None` when `nullopt`.
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        if self.has_value() {
            Some(self.inner.as_str())
        } else {
            None
        }
    }

    /// Takes the value out, consuming self.
    #[inline]
    pub fn get(self) -> Option<String> {
        let OptionalStr { inner } = self; // no `Drop` impl, so the move is allowed
        if inner.is_none_cell() {
            None
        } else {
            Some(inner)
        }
    }

    /// Overwrites the value in place, dropping the previous one first (dec-ref'd
    /// if it was a heap string).
    ///
    /// `&mut self`, not `&self` like POD [`Optional::set`]: `as_str` hands out
    /// borrows into a refcounted cell, so a shared-ref setter could drop the
    /// backing string under a live `&str`.
    #[inline]
    pub fn set(&mut self, value: Option<String>) {
        // Assignment drops the old `String` (dec_ref if heap) before moving in the new.
        self.inner = match value {
            Some(s) => s,
            None => String::none_cell(),
        };
    }
}

impl Default for OptionalStr {
    /// `nullopt`, matching the C++ default constructor.
    #[inline]
    fn default() -> Self {
        Self::none()
    }
}

impl From<Option<String>> for OptionalStr {
    #[inline]
    fn from(value: Option<String>) -> Self {
        match value {
            Some(s) => Self::some(s),
            None => Self::none(),
        }
    }
}

impl From<OptionalStr> for Option<String> {
    #[inline]
    fn from(value: OptionalStr) -> Self {
        value.get()
    }
}

impl Debug for OptionalStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "OptionalStr::Some({s:?})"),
            None => f.write_str("OptionalStr::None"),
        }
    }
}

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

    // 16-byte cell (type_index@0, small_str_len@4, union@8); no padding.
    fn str_image(o: &OptionalStr) -> [u8; 16] {
        let p = o as *const OptionalStr as *const u8;
        let mut b = [0u8; 16];
        // SAFETY: OptionalStr is a fully-initialized 16-byte TVMFFIAny cell.
        unsafe { std::ptr::copy_nonoverlapping(p, b.as_mut_ptr(), 16) };
        b
    }

    #[test]
    fn optional_str_byte_image_matches_cpp() {
        // C++ probe: Optional<String> none => 16 zero bytes;
        //            some("hi")           => 0b 00 00 00 | 02 00 00 00 | 68 69 00 ...
        assert_eq!(str_image(&OptionalStr::none()), [0u8; 16]);
        let some = OptionalStr::some(String::from("hi"));
        let b = str_image(&some);
        assert_eq!(&b[0..4], &[0x0b, 0, 0, 0], "type_index = kTVMFFISmallStr");
        assert_eq!(&b[4..8], &[0x02, 0, 0, 0], "small_str_len = 2");
        assert_eq!(&b[8..10], b"hi", "inline payload");
    }

    #[test]
    fn optional_str_roundtrip_and_conversions() {
        // small (inline) string
        let s = OptionalStr::some(String::from("hi"));
        assert!(s.has_value());
        assert_eq!(s.as_str(), Some("hi"));
        assert_eq!(s.get().as_deref(), Some("hi"));

        // nullopt
        let n = OptionalStr::none();
        assert!(n.is_none());
        assert_eq!(n.as_str(), None);
        assert_eq!(n.get(), None);

        // conversions + default
        let from_some: OptionalStr = Some(String::from("x")).into();
        assert_eq!(from_some.as_str(), Some("x"));
        let back: Option<String> = OptionalStr::none().into();
        assert!(back.is_none());
        assert!(OptionalStr::default().is_none());
    }

    #[test]
    fn optional_str_heap_clone_no_double_free() {
        // long (heap) string exercises refcounted Clone/Drop through the wrapper.
        let long = String::from("a-very-long-heap-allocated-string-value");
        let a = OptionalStr::some(long);
        let b = a.clone();
        assert_eq!(a.as_str(), Some("a-very-long-heap-allocated-string-value"));
        assert_eq!(b.as_str(), Some("a-very-long-heap-allocated-string-value"));
        // both drop here: two dec_refs balancing the clone's inc_ref, no leak/UAF.
    }

    #[test]
    fn optional_str_set_in_place() {
        // Start engaged with a heap string, then replace it: `set` must drop
        // (dec_ref) the old heap string before moving the new one in.
        let mut o = OptionalStr::some(String::from("first-long-heap-allocated-value"));
        assert_eq!(o.as_str(), Some("first-long-heap-allocated-value"));

        o.set(Some(String::from("second-long-heap-allocated-value")));
        assert_eq!(o.as_str(), Some("second-long-heap-allocated-value"));

        // disengage (drops the heap string), then re-engage
        o.set(None);
        assert!(o.is_none());
        assert_eq!(o.as_str(), None);

        o.set(Some(String::from("x")));
        assert_eq!(o.as_str(), Some("x"));
    }
}
