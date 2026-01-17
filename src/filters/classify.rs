// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Filters that map a signal onto a fixed set of discrete values (e.g. `on`, `off`).

#![allow(clippy::use_self)]
#![allow(clippy::wildcard_imports)]

/// Debounce filter for noise reduction in digital signals.
///
/// Filters out short pulses and noise spikes by requiring signals to remain stable for a
/// configurable duration before state changes are recognized.
pub mod debounce;

/// Schmitt trigger hysteresis filter.
///
/// Implements hysteresis with separate threshold values for rising and falling edges,
/// preventing oscillation around a single threshold and improving noise immunity.
pub mod schmitt;

/// Threshold detector for binary signal classification.
///
/// Classifies input values as above or below a threshold, producing binary output useful for
/// level detection, event triggering, and signal digitization.
pub mod threshold;

/// Peak detection filter.
///
/// Identifies local maximum values in signals, useful for finding peaks in waveforms,
/// detecting events, and ridge extraction.
pub mod peaks;

/// Slope detection filter.
///
/// Classifies signal regions based on slope direction (rising, falling, or flat),
/// useful for trend analysis and phase detection.
pub mod slopes;

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
