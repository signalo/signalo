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
//! This layout matches the field order of [`super::Config`] and supports direct conversion via
//! `Config::from([b0, b1, b2, a1, a2])`.
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
    let pi = T::from(core::f64::consts::PI).expect("π is representable");
    let two = T::from(2.0).expect("2 is representable");
    let sqrt2 = T::from(2.0).expect("2 is representable").sqrt();
    (pi, two, sqrt2)
}

/// Butterworth filter coefficient calculator.
///
/// Provides design equations for Butterworth lowpass, highpass, cookbook bandpass,
/// and cookbook bandstop (notch) biquad filter coefficients using standard DSP formulae.
///
/// All methods take `(sample_rate, frequency[, q])` in that order — sample rate first,
/// then the characteristic frequency. This matches the convention of most DSP textbooks.
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
    /// In debug builds, panics if `sample_rate <= 0`, `freq <= 0`, or `freq >= sample_rate / 2`.
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
        debug_assert!(sample_rate > T::zero(), "sample_rate must be positive");
        debug_assert!(freq > T::zero(), "freq must be positive");
        debug_assert!(
            freq < sample_rate / T::from(2.0).unwrap(),
            "freq must be below Nyquist (sample_rate / 2)"
        );

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
    /// In debug builds, panics if `sample_rate <= 0`, `freq <= 0`, or `freq >= sample_rate / 2`.
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
        debug_assert!(sample_rate > T::zero(), "sample_rate must be positive");
        debug_assert!(freq > T::zero(), "freq must be positive");
        debug_assert!(
            freq < sample_rate / T::from(2.0).unwrap(),
            "freq must be below Nyquist (sample_rate / 2)"
        );

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

    /// Compute bandpass coefficients for a filter centered at `center` Hz with quality factor `q`,
    /// given a `sample_rate` in Hz.
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
    /// In debug builds, panics if `sample_rate <= 0`, `center <= 0`, `center >= sample_rate / 2`,
    /// or `q <= 0`.
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
        debug_assert!(sample_rate > T::zero(), "sample_rate must be positive");
        debug_assert!(center > T::zero(), "center must be positive");
        debug_assert!(
            center < sample_rate / T::from(2.0).unwrap(),
            "center must be below Nyquist (sample_rate / 2)"
        );
        debug_assert!(q > T::zero(), "q must be positive");

        let (pi, two, _) = float_constants::<T>();

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

    /// Compute bandstop (notch) coefficients for a filter centered at `center` Hz
    /// with quality factor `q`, given a `sample_rate` in Hz.
    ///
    /// Implements the *notch* design from the Audio EQ Cookbook: a 2nd-order
    /// IIR with a pair of complex-conjugate zeros lying exactly on the unit
    /// circle at ±ω₀. DC and Nyquist gains are unity; the response is exactly
    /// zero at `center`.
    ///
    /// For a strict 4th-order Butterworth band-reject response, cascade two
    /// biquads via [`super::cascade::BiquadCascade`]; this single-biquad form
    /// matches the convention used by [`Self::bandpass`].
    ///
    /// Returns `[b0, b1, b2, a1, a2]` normalized by `a0`.
    ///
    /// # Panics
    ///
    /// Panics if `T` cannot represent the constants `PI` or `2.0`. This is
    /// infallible for standard `f32` and `f64` types.
    ///
    /// In debug builds, panics if `sample_rate <= 0`, `center <= 0`,
    /// `center >= sample_rate / 2`, or `q <= 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "std")] {
    /// # use signalo::filters::biquad::coefficients::Butterworth;
    /// let coeffs = Butterworth::bandstop(44100.0, 1000.0, 1.0);
    /// assert!(coeffs[0] > 0.0); // b0 should be positive
    /// # }
    /// ```
    #[cfg(feature = "std")]
    #[allow(clippy::unwrap_used)]
    pub fn bandstop<T: Float>(sample_rate: T, center: T, q: T) -> [T; 5] {
        debug_assert!(sample_rate > T::zero(), "sample_rate must be positive");
        debug_assert!(center > T::zero(), "center must be positive");
        debug_assert!(
            center < sample_rate / T::from(2.0).unwrap(),
            "center must be below Nyquist (sample_rate / 2)"
        );
        debug_assert!(q > T::zero(), "q must be positive");

        let (pi, two, _) = float_constants::<T>();

        let omega = two * pi * center / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();

        let alpha = sin_omega / (two * q);

        let b0 = T::one();
        let b1 = -two * cos_omega;
        let b2 = T::one();

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
    use alloc::vec::Vec;

    #[cfg(feature = "std")]
    use approx::assert_abs_diff_eq;

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
        assert_abs_diff_eq!(dc_gain, 1.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_highpass_nyquist_gain() {
        // Butterworth highpass at Nyquist / 2 (i.e. sr / 4) should have high-frequency gain ≈ 1.0
        let sample_rate = 44100.0f64;
        let freq = sample_rate / 4.0; // Nyquist / 2 = sample_rate / 4

        let coeffs = Butterworth::highpass(sample_rate, freq);
        let [b0, b1, b2, a1, a2] = coeffs;

        // Gain at Nyquist (z=-1): (b0 - b1 + b2) / (1 - a1 + a2)
        let nyquist_gain = (b0 - b1 + b2) / (1.0 - a1 + a2);

        // Should be close to 1.0
        assert_abs_diff_eq!(nyquist_gain, 1.0, epsilon = 1e-6);
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
        assert_abs_diff_eq!(b0, -b2, epsilon = 1e-10);

        // b1 should be zero
        assert_abs_diff_eq!(b1, 0.0, epsilon = 1e-10);

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

        assert_abs_diff_eq!(nyquist_gain, 1.0, epsilon = 1e-6);
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
        assert_abs_diff_eq!(dc_gain, 0.0, epsilon = 1e-10);

        // Nyquist gain (z=-1): (b0 - b1 + b2) / (1 - a1 + a2) = 0 for bandpass
        let nyquist_gain = (b0 - b1 + b2) / (1.0 - a1 + a2);
        assert_abs_diff_eq!(nyquist_gain, 0.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_lowpass_cutoff_gain() {
        let sample_rate = 44100.0f64;
        let freq = 1000.0;
        let [b0, b1, b2, a1, a2] = Butterworth::lowpass(sample_rate, freq);

        let omega = 2.0 * core::f64::consts::PI * freq / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        assert_abs_diff_eq!(gain, 1.0_f64 / 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_highpass_cutoff_gain() {
        let sample_rate = 44100.0f64;
        let freq = 1000.0;
        let [b0, b1, b2, a1, a2] = Butterworth::highpass(sample_rate, freq);

        let omega = 2.0 * core::f64::consts::PI * freq / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        assert_abs_diff_eq!(gain, 1.0_f64 / 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_lowpass_cutoff_gain_f32() {
        let sample_rate = 44100.0f32;
        let freq = 1000.0f32;
        let [b0, b1, b2, a1, a2] = Butterworth::lowpass(sample_rate, freq);

        let omega = 2.0f32 * core::f32::consts::PI * freq / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        assert_abs_diff_eq!(gain, 1.0f32 / 2.0f32.sqrt(), epsilon = 1e-4);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_highpass_cutoff_gain_f32() {
        let sample_rate = 44100.0f32;
        let freq = 1000.0f32;
        let [b0, b1, b2, a1, a2] = Butterworth::highpass(sample_rate, freq);

        let omega = 2.0f32 * core::f32::consts::PI * freq / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        assert_abs_diff_eq!(gain, 1.0f32 / 2.0f32.sqrt(), epsilon = 1e-4);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_bandpass_center_gain_f32() {
        let sample_rate = 44100.0f32;
        let center = 1000.0f32;
        let q = 1.0f32;
        let [b0, b1, b2, a1, a2] = Butterworth::bandpass(sample_rate, center, q);

        let omega = 2.0f32 * core::f32::consts::PI * center / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        // Constant 0 dB peak gain bandpass has gain = 1.0 at center frequency
        assert_abs_diff_eq!(gain, 1.0f32, epsilon = 1e-4);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_lowpass_impulse_response_3db_cutoff() {
        use crate::filters::biquad::{Biquad, Config};
        use crate::traits::{Filter, WithConfig};

        let sample_rate = 44100.0f64;
        let freq = 1000.0;
        let coeffs = Butterworth::lowpass(sample_rate, freq);
        let [b0, b1, b2, a1, a2] = coeffs;

        let mut filter = Biquad::with_config(Config { b0, b1, b2, a1, a2 });

        // Drive an impulse and accumulate the frequency response via DFT at cutoff
        let n = 4096usize;
        let mut response: Vec<f64> = Vec::with_capacity(n);
        response.push(filter.filter(1.0));

        for _ in 1..n {
            response.push(filter.filter(0.0));
        }

        // Evaluate DFT at freq bin corresponding to cutoff
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let (re, im) = {
            let n_f = n as f64;
            let k = (freq / sample_rate * n_f).round() as usize;
            let k_f = k as f64;
            response
                .iter()
                .enumerate()
                .fold((0.0f64, 0.0f64), |(re, im), (i, &h)| {
                    let angle = -2.0 * core::f64::consts::PI * k_f * (i as f64) / n_f;
                    (re + h * angle.cos(), im + h * angle.sin())
                })
        };

        let gain = re.hypot(im);

        assert_abs_diff_eq!(gain, 1.0 / 2.0f64.sqrt(), epsilon = 1e-3);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_highpass_impulse_response_3db_cutoff() {
        use crate::filters::biquad::{Biquad, Config};
        use crate::traits::{Filter, WithConfig};

        let sample_rate = 44100.0f64;
        let freq = 1000.0;
        let coeffs = Butterworth::highpass(sample_rate, freq);
        let [b0, b1, b2, a1, a2] = coeffs;

        let mut filter = Biquad::with_config(Config { b0, b1, b2, a1, a2 });

        let n = 4096usize;
        let mut response: Vec<f64> = Vec::with_capacity(n);
        response.push(filter.filter(1.0));

        for _ in 1..n {
            response.push(filter.filter(0.0));
        }

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let (re, im) = {
            let n_f = n as f64;
            let k = (freq / sample_rate * n_f).round() as usize;
            let k_f = k as f64;
            response
                .iter()
                .enumerate()
                .fold((0.0f64, 0.0f64), |(re, im), (i, &h)| {
                    let angle = -2.0 * core::f64::consts::PI * k_f * (i as f64) / n_f;
                    (re + h * angle.cos(), im + h * angle.sin())
                })
        };

        let gain = re.hypot(im);

        assert_abs_diff_eq!(gain, 1.0 / 2.0f64.sqrt(), epsilon = 1e-3);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_bandstop_coefficients_symmetry() {
        let sample_rate = 44100.0f64;
        let center = 1000.0;
        let q = 1.0;

        let coeffs = Butterworth::bandstop(sample_rate, center, q);
        let [b0, b1, b2, a1, a2] = coeffs;

        // b0 ≈ b2, b1 ≈ a1 (Cookbook notch symmetry)
        assert_abs_diff_eq!(b0, b2, epsilon = 1e-10);
        assert_abs_diff_eq!(b1, a1, epsilon = 1e-10);

        // Denominator non-zero
        assert!((1.0 + a1 + a2).abs() > 0.0);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_bandstop_dc_and_nyquist_gain() {
        let sample_rate = 44100.0f64;
        let center = 1000.0;
        let q = 1.0;

        let coeffs = Butterworth::bandstop(sample_rate, center, q);
        let [b0, b1, b2, a1, a2] = coeffs;

        // DC gain (z=1): (b0 + b1 + b2) / (1 + a1 + a2) ≈ 1
        let dc_gain = (b0 + b1 + b2) / (1.0 + a1 + a2);
        assert_abs_diff_eq!(dc_gain, 1.0, epsilon = 1e-10);

        // Nyquist gain (z=-1): (b0 - b1 + b2) / (1 - a1 + a2) ≈ 1
        let nyquist_gain = (b0 - b1 + b2) / (1.0 - a1 + a2);
        assert_abs_diff_eq!(nyquist_gain, 1.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_bandstop_center_notch_f64() {
        let sample_rate = 44100.0f64;
        let center = 1000.0;
        let q = 1.0;

        let coeffs = Butterworth::bandstop(sample_rate, center, q);
        let [b0, b1, b2, a1, a2] = coeffs;

        let omega = 2.0 * core::f64::consts::PI * center / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        // Magnitude should be ≈ 0 at center frequency (true notch)
        assert_abs_diff_eq!(gain, 0.0, epsilon = 1e-10);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_butterworth_bandstop_center_notch_f32() {
        let sample_rate = 44100.0f32;
        let center = 1000.0f32;
        let q = 1.0f32;

        let coeffs = Butterworth::bandstop(sample_rate, center, q);
        let [b0, b1, b2, a1, a2] = coeffs;

        let omega = 2.0f32 * core::f32::consts::PI * center / sample_rate;
        let (s, c) = omega.sin_cos();
        let num_re = b0 + b1 * c + b2 * (c * c - s * s);
        let num_im = -b1 * s + b2 * (-2.0 * s * c);
        let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
        let den_im = -a1 * s + a2 * (-2.0 * s * c);
        let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

        // Magnitude should be ≈ 0 at center frequency (true notch)
        assert_abs_diff_eq!(gain, 0.0f32, epsilon = 1e-4);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_bandstop_impulse_response_notch_via_dft() {
        use crate::filters::biquad::{Biquad, Config};
        use crate::traits::{Filter, WithConfig};

        let sample_rate = 44100.0f64;
        let center = 1000.0;
        let q = 1.0;

        let coeffs = Butterworth::bandstop(sample_rate, center, q);
        let [b0, b1, b2, a1, a2] = coeffs;

        let mut filter = Biquad::with_config(Config { b0, b1, b2, a1, a2 });

        let n = 4096usize;
        let mut response: Vec<f64> = Vec::with_capacity(n);
        response.push(filter.filter(1.0));

        for _ in 1..n {
            response.push(filter.filter(0.0));
        }

        // DFT bin at center should have magnitude ≈ 0
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let (re, im) = {
            let n_f = n as f64;
            let k = (center / sample_rate * n_f).round() as usize;
            let k_f = k as f64;
            response
                .iter()
                .enumerate()
                .fold((0.0f64, 0.0f64), |(re, im), (i, &h)| {
                    let angle = -2.0 * core::f64::consts::PI * k_f * (i as f64) / n_f;
                    (re + h * angle.cos(), im + h * angle.sin())
                })
        };

        let gain = re.hypot(im);
        assert_abs_diff_eq!(gain, 0.0, epsilon = 1e-2);

        // DFT bin far from center should have magnitude ≈ 1.0
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let (re_low, im_low) = {
            let n_f = n as f64;
            let freq_low = center / 10.0;
            let k = (freq_low / sample_rate * n_f).round() as usize;
            let k_f = k as f64;
            response
                .iter()
                .enumerate()
                .fold((0.0f64, 0.0f64), |(re, im), (i, &h)| {
                    let angle = -2.0 * core::f64::consts::PI * k_f * (i as f64) / n_f;
                    (re + h * angle.cos(), im + h * angle.sin())
                })
        };

        let passband_gain = re_low.hypot(im_low);
        assert_abs_diff_eq!(passband_gain, 1.0, epsilon = 1e-2);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_bandstop_3db_bandwidth() {
        let sample_rate = 44100.0f64;
        let center = 1000.0;
        let q = 5.0;

        let coeffs = Butterworth::bandstop(sample_rate, center, q);
        let [b0, b1, b2, a1, a2] = coeffs;

        // Test at f = center ± center/(2Q) — the ±3 dB edges
        let bw_half = center / (2.0 * q);

        for &offset in &[-bw_half, bw_half] {
            let f = center + offset;
            let omega = 2.0 * core::f64::consts::PI * f / sample_rate;
            let (s, c) = omega.sin_cos();
            let num_re = b0 + b1 * c + b2 * (c * c - s * s);
            let num_im = -b1 * s + b2 * (-2.0 * s * c);
            let den_re = 1.0 + a1 * c + a2 * (c * c - s * s);
            let den_im = -a1 * s + a2 * (-2.0 * s * c);
            let gain = num_re.hypot(num_im) / den_re.hypot(den_im);

            assert_abs_diff_eq!(gain, 1.0 / 2.0f64.sqrt(), epsilon = 3e-2);
        }
    }
}
