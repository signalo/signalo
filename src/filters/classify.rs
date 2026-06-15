// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Classification filters; continuous signals to discrete states.
//!
//! These filters convert a continuous-valued (analog) signal into a discrete set of
//! output classes. This is the bridge between raw sensor data and boolean/event-driven
//! logic: threshold crossings, edge detection, debouncing, and zero-crossing detection.
//!
//! # The `Classification` trait
//!
//! Each filter produces output of some type `T` that must implement
//! [`Classification`]`<T, N>`. The trait declares a fixed set of
//! `N` valid output classes (e.g. `[false, true]` for `bool`, `[-1, 0, 1]` for integer
//! sign). This lets classification filters be generic over their output type while
//! remaining type-safe.
//!
//! # When to use which filter
//!
//! | Filter                         | Purpose                                                    |
//! | ------------------------------ | ---------------------------------------------------------- |
//! | `threshold::Threshold`         | Simple binary classification at a fixed threshold level    |
//! | `schmitt::Schmitt`             | Hysteresis threshold (Schmitt trigger); two thresholds    |
//! | `debounce::Debounce`           | Noise rejection: require a state to persist before change  |
//! | `peaks::Peaks`                 | Local extrema detection (peaks and valleys)                |
//! | `slopes::Slopes`               | Rising/falling edge detection with hysteresis              |
//! | `zero_crossing::ZeroCrossing`  | Detect sign-change crossings through zero                  |
//!
//! - **Threshold** is the simplest classifier: compare against a single value. Use for
//!   basic limit switches, level detection, and binary on/off states.
//! - **Schmitt** adds hysteresis: the turn-on threshold is higher than the turn-off
//!   threshold. This prevents chatter when a noisy signal hovers near the boundary.
//!   Essential when driving relays, actuators, or state machines from analog sensors.
//! - **Debounce** requires a state to persist for a minimum number of consecutive samples
//!   before the output changes. Use for mechanical switch inputs or any binary signal
//!   with transient glitches.
//! - **Peaks** detects local maxima and minima within a sliding window. Useful for
//!   feature extraction in sensor streams (heartbeat peaks, vibration maxima).
//! - **Slopes** classifies segments as rising, falling, or flat based on sustained
//!   trends. Built on the Schmitt trigger for hysteresis in the derivative domain.
//! - **`ZeroCrossing`** detects when a signal changes sign (positive to negative or vice
//!   versa). Useful in AC signal analysis, phase detection, and frequency counting.
//!
//! # See also
//!
//! - [`super::rank::min`] / [`super::rank::max`]: envelope detection as a pre-processing
//!   step before threshold or Schmitt classification.
//! - [`super::estimate`]: state-space observers that can produce smoothed estimates for
//!   feeding into classifiers.

#![allow(clippy::use_self)]
#![allow(clippy::wildcard_imports)]

pub mod debounce;

pub mod schmitt;

pub mod threshold;

pub mod peaks;

pub mod slopes;

pub mod zero_crossing;

/// A trait describing a classification value.
pub trait Classification<T, const N: usize>: Sized {
    /// The available classes.
    fn classes() -> [T; N];
}

macro_rules! classification_impl {
    ($head:ty, $($tail:ty),+ => [$a:expr, $b:expr]) => {
        classification_impl!($head => [$a, $b]);
        classification_impl!($($tail),+ => [$a, $b]);
    };
    ($t:ty => [$a:expr, $b:expr]) => {
        impl Classification<Self, 2> for $t {
            fn classes() -> [Self; 2] {
                [$a, $b]
            }
        }
    };
    ($head:ty, $($tail:ty),+ => [$a:expr, $b:expr, $c:expr]) => {
        classification_impl!($head => [$a, $b, $c]);
        classification_impl!($($tail),+ => [$a, $b, $c]);
    };
    ($t:ty => [$a:expr, $b:expr, $c:expr]) => {
        impl Classification<Self, 3> for $t {
            fn classes() -> [Self; 3] {
                [$a, $b, $c]
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
