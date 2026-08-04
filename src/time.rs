// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Time-tagged signal values for continuous-time processing.
//!
//! This module provides [`Timed`], the shared value type that pairs a signal
//! value with the time delta preceding it. Continuous-time pipelines carry
//! irregularly-spaced samples, so each sample records the interval elapsed
//! since its predecessor rather than an absolute timestamp or a global sample
//! rate.
//!
//! The `dt` field is relative: it measures the gap between the current sample
//! and the previous one. Units are caller-consistent — a pipeline may work in
//! samples or seconds — and the crate itself stores no sample rate.
//!
//! # Examples
//!
//! ```
//! use signalo::Timed;
//!
//! let sample = Timed::new(1.0_f32, 0.25_f32);
//!
//! assert_eq!(sample.value, 1.0);
//! assert_eq!(sample.dt, 0.25);
//! ```

/// A signal value paired with the time delta preceding it.
///
/// `dt` is the interval elapsed since the previous sample (relative, not absolute).
/// Units are caller-consistent (samples or seconds); the crate stores no sample rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Timed<T, Tm = T> {
    /// The signal value.
    pub value: T,
    /// Interval elapsed since the previous sample.
    pub dt: Tm,
}

impl<T, Tm> Timed<T, Tm> {
    /// Creates a new `Timed` from a value and its preceding time delta.
    pub const fn new(value: T, dt: Tm) -> Self {
        Self { value, dt }
    }
}

/// Per-sample blend factor for a time-constant smoother: `1 − exp(−dt / tau)`.
///
/// A `tau` of zero yields `1 − exp(−∞) = 1` (the sample replaces the state); a negative or zero
/// `dt` yields a factor in `[0, 1)` accordingly. Units of `dt` and `tau` must match.
pub(crate) fn smoothing_alpha<T: num_traits::Float>(dt: T, tau: T) -> T {
    T::one() - (-dt / tau).exp()
}

/// Per-sample decay factor for a time-constant decay: `exp(−dt / tau)`.
pub(crate) fn decay_factor<T: num_traits::Float>(dt: T, tau: T) -> T {
    (-dt / tau).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timed_new_exposes_fields() {
        let s = Timed::new(3.0_f32, 0.5_f32);
        assert_eq!(s.value, 3.0);
        assert_eq!(s.dt, 0.5);
        assert_eq!(
            s,
            Timed {
                value: 3.0,
                dt: 0.5
            }
        );
    }
}
