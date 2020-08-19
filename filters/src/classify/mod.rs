// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filters that map a signal onto a fixed set of discrete values (e.g. `on`, `off`).

#![allow(clippy::use_self)]
#![allow(clippy::wildcard_imports)]

use generic_array::typenum::*;
use generic_array::{ArrayLength, GenericArray};

pub mod debounce;
pub mod schmitt;
pub mod threshold;

pub mod peaks;
pub mod slopes;

/// A trait describing a classification value.
pub trait Classification<T, N>: Sized
where
    N: ArrayLength<T>,
{
    /// The available classes.
    fn classes() -> GenericArray<T, N>;
}

macro_rules! classification_impl {
    ($head:ty, $($tail:ty),+ => [$a:expr, $b:expr]) => {
        classification_impl!($head => [$a, $b]);
        classification_impl!($($tail),+ => [$a, $b]);
    };
    ($t:ty => [$a:expr, $b:expr]) => {
        impl Classification<Self, U2> for $t {
            fn classes() -> GenericArray<Self, U2> {
                arr![Self; $a, $b]
            }
        }
    };
    ($head:ty, $($tail:ty),+ => [$a:expr, $b:expr, $c:expr]) => {
        classification_impl!($head => [$a, $b, $c]);
        classification_impl!($($tail),+ => [$a, $b, $c]);
    };
    ($t:ty => [$a:expr, $b:expr, $c:expr]) => {
        impl Classification<Self, U3> for $t {
            fn classes() -> GenericArray<Self, U3> {
                arr![Self; $a, $b, $c]
            }
        }
    };
}

classification_impl!(bool => [false, true]);

classification_impl!(f32, f64 => [-1.0, 1.0]);
classification_impl!(f32, f64 => [1.0, 0.0, -1.0]);

classification_impl!(u8, u16, u32, u64 => [0, 1]);
classification_impl!(i8, i16, i32, i64 => [-1, 1]);
classification_impl!(i8, i16, i32, i64 => [1, 0, -1]);
