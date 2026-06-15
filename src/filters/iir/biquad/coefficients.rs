// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Coefficient factory traits for computing biquad filter coefficients from standard DSP design
//! equations (low-pass, high-pass, band-pass, notch, peaking, etc.).
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
//! This layout matches the field order of `Config` and supports direct conversion via
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
//! the `num_traits::Float` trait, which requires the `std` or `libm` feature for a math library.
//! Functions using these operations are gated with `#[cfg(any(feature = "libm", feature = "std"))]`.
//!
//! # References
//!
//! - **Audio EQ Cookbook** by Robert Bristow-Johnson; Standard reference for biquad
//!   coefficient formulae used in audio signal processing.

use num_traits::Float;

/// Shared constants for biquad coefficient calculations.
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
/// All methods take `(sample_rate, frequency[, q])` in that order; sample rate first,
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
    /// # use signalo::filters::iir::biquad::coefficients::Butterworth;
    /// let coeffs = Butterworth::lowpass(44100.0, 1000.0);
    /// assert!(coeffs[0] > 0.0); // b0 should be positive
    /// ```
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
    /// # use signalo::filters::iir::biquad::coefficients::Butterworth;
    /// let coeffs = Butterworth::highpass(44100.0, 1000.0);
    /// assert!(coeffs[0] > 0.0); // b0 should be positive
    /// ```
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
    /// # use signalo::filters::iir::biquad::coefficients::Butterworth;
    /// let coeffs = Butterworth::bandpass(44100.0, 1000.0, 1.0);
    /// assert!(coeffs[0] > 0.0); // b0 should be positive
    /// ```
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
    /// # use signalo::filters::iir::biquad::coefficients::Butterworth;
    /// let coeffs = Butterworth::bandstop(44100.0, 1000.0, 1.0);
    /// assert!(coeffs[0] > 0.0); // b0 should be positive
    /// ```
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
mod tests;
