// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Coefficient factory traits for biquad filter design.
//!
//! This module provides traits and implementations for computing biquad filter coefficients
//! using standard DSP design equations. Coefficient calculators are implemented as zero-size
//! types that produce pure functions, allowing filter designers to compute coefficients at
//! construction time without runtime state.
//!
//! # Coefficient Layout
//!
//! All coefficient functions return a 5-element array `[T; 5]` arranged as:
//! - `[0]` = b0 (feedforward numerator)
//! - `[1]` = b1 (feedforward numerator)
//! - `[2]` = b2 (feedforward numerator)
//! - `[3]` = a1 (feedback denominator)
//! - `[4]` = a2 (feedback denominator)
//!
//! These coefficients are used in the biquad difference equation:
//! ```text
//! y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
//! ```
//!
//! # Transcendental Functions
//!
//! Coefficient calculations require `sin`, `cos`, and `sqrt` functions. These are provided by
//! the `num_traits::Float` trait, which requires the `std` feature or a compatible math library.
//! Functions using these operations are gated with `#[cfg(feature = "std")]` or require
//! `T: Float` bounds.
//!
//! # References
//!
//! - **Audio EQ Cookbook** by Robert Bristow-Johnson — Standard reference for biquad
//!   coefficient formulae used in audio signal processing.

#[cfg(feature = "std")]
use num_traits::Float;

/// Shared constants for biquad coefficient calculations.
#[cfg(feature = "std")]
fn float_constants<T: Float>() -> (T, T, T) {
    let pi = T::from(core::f64::consts::PI).unwrap();
    let two = T::from(2.0).unwrap();
    let sqrt2 = T::from(2.0_f64.sqrt()).unwrap();
    (pi, two, sqrt2)
}

/// Butterworth filter coefficient calculator.
///
/// Provides design equations for Butterworth lowpass, highpass, and bandpass
/// biquad filter coefficients using standard DSP formulae.
///
/// # Requirements
///
/// - Requires `T: Float` for trigonometric and arithmetic operations.
/// - Requires feature "std" or compatible floating-point trait implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Butterworth;

impl Butterworth {
    /// Compute Butterworth lowpass coefficients for a filter at `freq` Hz given a `sample_rate` in Hz.
    ///
    /// Returns `[b0, b1, b2, a1, a2]` normalized by `a0`. Uses `Q = 1/√2` (maximally flat
    /// Butterworth response).
    ///
    /// # Panics
    ///
    /// Panics if `T` cannot represent the constants `PI`, `2.0`, or `√2`. This is infallible
    /// for standard `f32` and `f64` types.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "std")] {
    /// # use signalo::filters::biquad::coefficients::Butterworth;
    /// let coeffs = Butterworth::lowpass(44100.0, 1000.0);
    /// assert!(coeffs[0] > 0.0); // b0 should be positive
    /// # }
    /// ```
    #[cfg(feature = "std")]
    #[allow(clippy::unwrap_used)]
    pub fn lowpass<T: Float>(sample_rate: T, freq: T) -> [T; 5] {
        let (pi, two, sqrt2) = float_constants::<T>();

        let omega = two * pi * freq / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();

        let q = T::one() / sqrt2;
        let alpha = sin_omega / (two * q);

        let one_minus_cos = T::one() - cos_omega;
        let b0 = one_minus_cos / two;
        let b1 = one_minus_cos;
        let b2 = one_minus_cos / two;

        let one_plus_alpha = T::one() + alpha;
        let a0 = one_plus_alpha;
        let a1 = -two * cos_omega;
        let a2 = T::one() - alpha;

        [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
    }

    /// Compute Butterworth highpass coefficients for a filter at `freq` Hz given a `sample_rate` in Hz.
    ///
    /// Returns `[b0, b1, b2, a1, a2]` normalized by `a0`. Uses `Q = 1/√2` (maximally flat
    /// Butterworth response).
    ///
    /// # Panics
    ///
    /// Panics if `T` cannot represent the constants `PI`, `2.0`, or `√2`. This is infallible
    /// for standard `f32` and `f64` types.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "std")] {
    /// # use signalo::filters::biquad::coefficients::Butterworth;
    /// let coeffs = Butterworth::highpass(44100.0, 1000.0);
    /// assert!(coeffs[0] > 0.0); // b0 should be positive
    /// # }
    /// ```
    #[cfg(feature = "std")]
    #[allow(clippy::unwrap_used)]
    pub fn highpass<T: Float>(sample_rate: T, freq: T) -> [T; 5] {
        let (pi, two, sqrt2) = float_constants::<T>();

        let omega = two * pi * freq / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();

        let q = T::one() / sqrt2;
        let alpha = sin_omega / (two * q);

        let one_plus_cos = T::one() + cos_omega;
        let b0 = one_plus_cos / two;
        let b1 = -(one_plus_cos);
        let b2 = one_plus_cos / two;

        let one_plus_alpha = T::one() + alpha;
        let a0 = one_plus_alpha;
        let a1 = -two * cos_omega;
        let a2 = T::one() - alpha;

        [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
    }

    /// Compute Butterworth bandpass coefficients for a filter centered at `center` Hz with
    /// quality factor `q`, given a `sample_rate` in Hz.
    ///
    /// Implements the *constant 0 dB peak gain* variant from the Audio EQ Cookbook (`b0 = α`).
    /// For the constant skirt-gain variant (`b0 = Q·α`) use a different design.
    ///
    /// Returns `[b0, b1, b2, a1, a2]` normalized by `a0`.
    ///
    /// # Panics
    ///
    /// Panics if `T` cannot represent the constants `PI` or `2.0`. This is infallible
    /// for standard `f32` and `f64` types.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "std")] {
    /// # use signalo::filters::biquad::coefficients::Butterworth;
    /// let coeffs = Butterworth::bandpass(44100.0, 1000.0, 1.0);
    /// assert!(coeffs[0] > 0.0); // b0 should be positive
    /// # }
    /// ```
    #[cfg(feature = "std")]
    #[allow(clippy::unwrap_used)]
    pub fn bandpass<T: Float>(sample_rate: T, center: T, q: T) -> [T; 5] {
        let (pi, two, _sqrt2) = float_constants::<T>();

        let omega = two * pi * center / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();

        let alpha = sin_omega / (two * q);

        let b0 = alpha;
        let b1 = T::zero();
        let b2 = -alpha;

        let one_plus_alpha = T::one() + alpha;
        let a0 = one_plus_alpha;
        let a1 = -two * cos_omega;
        let a2 = T::one() - alpha;

        [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use nearly_eq::assert_nearly_eq;

    #[cfg(feature = "std")]
    use super::*;

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_lowpass_dc_gain() {
        // Butterworth lowpass at Nyquist/4 should have DC gain ≈ 1.0
        let sample_rate = 44100.0f64;
        let freq = sample_rate / 4.0; // Nyquist / 4

        let coeffs = Butterworth::lowpass(sample_rate, freq);
        let [b0, b1, b2, a1, a2] = coeffs;

        // DC gain = (b0 + b1 + b2) / (1 + a1 + a2)
        let dc_gain = (b0 + b1 + b2) / (1.0 + a1 + a2);

        // Should be close to 1.0
        assert_nearly_eq!(dc_gain, 1.0, 1e-6);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_highpass_nyquist_gain() {
        // Butterworth highpass at Nyquist / 2 should have high frequency gain ≈ 1.0
        let sample_rate = 44100.0f64;
        let freq = sample_rate / 4.0; // Nyquist / 2

        let coeffs = Butterworth::highpass(sample_rate, freq);
        let [b0, b1, b2, a1, a2] = coeffs;

        // Gain at Nyquist (z=-1): (b0 - b1 + b2) / (1 - a1 + a2)
        let nyquist_gain = (b0 - b1 + b2) / (1.0 - a1 + a2);

        // Should be close to 1.0
        assert_nearly_eq!(nyquist_gain, 1.0, 1e-6);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_bandpass_coefficients_non_zero() {
        // Bandpass coefficients should be reasonable values
        let sample_rate = 44100.0f64;
        let center = 1000.0;
        let q = 1.0;

        let coeffs = Butterworth::bandpass(sample_rate, center, q);
        let [b0, b1, b2, a1, a2] = coeffs;

        // b0 and b2 should be non-zero and nearly equal (symmetric)
        assert!(b0 > 0.0);
        assert_nearly_eq!(b0, -b2, 1e-10);

        // b1 should be zero
        assert_nearly_eq!(b1, 0.0, 1e-10);

        // Denominator should be stable
        assert!((1.0 + a1 + a2).abs() > 0.0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_highpass_nyquist_gain_nonzero_omega() {
        // Pick a frequency where cos(omega) != 0 so denominator
        // formula matters (unlike the existing test at sr/4).
        let sample_rate = 44100.0f64;
        let freq = sample_rate / 6.0; // omega = pi/3, cos = 0.5

        let coeffs = Butterworth::highpass(sample_rate, freq);
        let [b0, b1, b2, a1, a2] = coeffs;

        // Gain at Nyquist (z=-1): (b0 - b1 + b2) / (1 - a1 + a2)
        let nyquist_gain = (b0 - b1 + b2) / (1.0 - a1 + a2);

        assert_nearly_eq!(nyquist_gain, 1.0, 1e-6);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_bandpass_dc_and_nyquist_gain() {
        let sample_rate = 44100.0f64;
        let center = 1000.0;
        let q = 1.0;

        let coeffs = Butterworth::bandpass(sample_rate, center, q);
        let [b0, b1, b2, a1, a2] = coeffs;

        // DC gain (z=1): (b0 + b1 + b2) / (1 + a1 + a2) = 0 for bandpass
        let dc_gain = (b0 + b1 + b2) / (1.0 + a1 + a2);
        assert_nearly_eq!(dc_gain, 0.0, 1e-10);

        // Nyquist gain (z=-1): (b0 - b1 + b2) / (1 - a1 + a2) = 0 for bandpass
        let nyquist_gain = (b0 - b1 + b2) / (1.0 - a1 + a2);
        assert_nearly_eq!(nyquist_gain, 0.0, 1e-10);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_lowpass_cutoff_gain() {
        let sample_rate = 44100.0f64;
        let freq = 1000.0;
        let [b0, b1, b2, a1, a2] = Butterworth::lowpass(sample_rate, freq);

        let omega = 2.0 * std::f64::consts::PI * freq / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        assert_nearly_eq!(gain, 1.0_f64 / 2.0_f64.sqrt(), 1e-10);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_highpass_cutoff_gain() {
        let sample_rate = 44100.0f64;
        let freq = 1000.0;
        let [b0, b1, b2, a1, a2] = Butterworth::highpass(sample_rate, freq);

        let omega = 2.0 * std::f64::consts::PI * freq / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        assert_nearly_eq!(gain, 1.0_f64 / 2.0_f64.sqrt(), 1e-10);
    }
}
