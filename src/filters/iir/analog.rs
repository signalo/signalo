// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Analog (s-domain) second-order section design data.
//!
//! An [`AnalogBiquad`] represents a normalized continuous-time second-order transfer
//! function in the Laplace variable `s`:
//!
//! ```text
//! H(s) = (b[2] s² + b[1] s + b[0]) / (a[2] s² + a[1] s + a[0])
//! ```
//!
//! This is design-time data, not a runnable filter: unlike [`biquad::Biquad`](super::biquad),
//! [`AnalogBiquad`] holds no state and has no [`Filter`](crate::traits::Filter) implementation,
//! because an s-domain transfer function cannot be evaluated sample-by-sample directly. Classic
//! analog prototypes (Butterworth, Chebyshev, Bessel, and so on) are naturally expressed in the
//! s-domain, where their pole/zero placement is simplest to reason about.
//!
//! To turn an [`AnalogBiquad`] into something that can process a signal, discretize it (for
//! example via the bilinear transform) into a digital [`biquad::Config`](super::biquad::Config),
//! which drives the [`biquad::Biquad`](super::biquad::Biquad) filter.

/// A normalized analog (s-domain) second-order section:
/// `H(s) = (b[2] s² + b[1] s + b[0]) / (a[2] s² + a[1] s + a[0])`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalogBiquad<T> {
    /// Numerator coefficients `[b0, b1, b2]`.
    pub b: [T; 3],
    /// Denominator coefficients `[a0, a1, a2]`.
    pub a: [T; 3],
}

impl<T> AnalogBiquad<T> {
    /// Creates a section from numerator `b = [b0,b1,b2]` and denominator `a = [a0,a1,a2]`.
    pub const fn new(b: [T; 3], a: [T; 3]) -> Self {
        Self { b, a }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analog_biquad_new_stores_coeffs() {
        let s = AnalogBiquad::new([1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(s.b, [1.0, 2.0, 3.0]);
        assert_eq!(s.a, [4.0, 5.0, 6.0]);
    }
}
