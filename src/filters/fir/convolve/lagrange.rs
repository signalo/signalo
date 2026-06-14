// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Lagrange fractional-delay FIR filters.
//!
//! A fractional-delay filter shifts a signal by a non-integer number of
//! samples using Lagrange polynomial interpolation. For a desired delay
//! `δ ∈ [0, M−1]` and `M` taps, the coefficients are:
//!
//! ```text
//! h[k] = ∏_{j=0, j≠k}^{M−1} (δ − j) / (k − j),    k = 0 … M−1
//! ```
//!
//! The filter is exact for polynomial signals up to degree `M−1`.
//!
//! # Use cases
//!
//! - Sensor fusion with timestamp misalignment.
//! - Resampling and sample-rate conversion (as a building block).
//! - Delay compensation in beamforming or antenna arrays.
//!
//! # Trade-offs vs. windowed-sinc
//!
//! Lagrange is cheap and exact for low-order polynomial signals, but has
//! wider transition bands than windowed-sinc. Prefer Lagrange when
//! per-sample latency matters more than frequency-domain control.
//!
//! # Numerical caveats
//!
//! The best numerical behaviour is for `δ` near the centre tap
//! (approximately `(M−1)/2`). Ill-conditioning increases near the edges
//! (`δ ≈ 0` or `δ ≈ M−1`).
//!
//! # Related
//!
//! - [`crate::filters::delay::Delay`] for integer delay.
//! - [`crate::filters::fir::convolve::windowed_sinc`] for frequency-domain
//!   fractional-delay alternatives via windowed-sinc resampling.
//!
//! # Half-sample delay
//!
//! `half_sample_delay()` is pre-defined for even `M ∈ {4, 6, 8, 10}`. Odd `M`
//! yields `(M−1)/2 ∈ ℤ` — that is an *integer* delay and should use
//! [`Delay`](crate::filters::delay::Delay) (or
//! [`Convolve::with_config`](crate::filters::fir::convolve::Convolve::with_config)
//! with a unit impulse at the relevant tap).
//!
//! `h[k]` pairs with the newest sample, so `δ = 0` outputs `x[n]` and
//! `δ = M − 1` outputs `x[n − (M − 1)]`. The Lagrange formula in this
//! module produces coefficients in exactly that order — no reversal is
//! needed at storage time.
//!

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

use crate::traits::WithConfig;

use super::{Config, Convolve};

/// Trait for Lagrange fractional-delay FIR filters.
///
/// The const generic `M` is the filter length (tap count).
/// The polynomial order is `M − 1` and the delay `δ` is in `[0, M−1]`.
#[cfg(any(feature = "libm", feature = "std"))]
pub trait FractionalDelay<T>: Sized {
    /// Returns a convolution filter with Lagrange interpolation coefficients
    /// for fractional delay `delta`.
    ///
    /// `delta` must be in `[0, M−1]`. The best numerical behaviour is for
    /// `delta` near `(M−1)/2`.
    fn lagrange(delta: T) -> Self;
}

#[cfg(any(feature = "libm", feature = "std"))]
impl<T: Float, const M: usize> FractionalDelay<T> for Convolve<T, M> {
    fn lagrange(delta: T) -> Self {
        assert!(
            M >= 2,
            "Lagrange fractional delay requires M >= 2 \
             (M = 1 is the trivial identity; use `Delay` instead)"
        );
        assert!(delta >= T::zero(), "delta must be non-negative");
        assert!(
            delta <= T::from(M - 1).expect("M too large for T::from"),
            "delta must be <= M-1"
        );

        let mut h = [T::zero(); M];

        for (k, hk) in h.iter_mut().enumerate() {
            let mut num = T::one();
            let mut den = T::one();
            let k_t = T::from(k).expect("k too large for T::from");

            for j in 0..M {
                if j != k {
                    let j_t = T::from(j).expect("j too large for T::from");
                    num = num * (delta - j_t);
                    den = den * (k_t - j_t);
                }
            }

            *hk = num / den;
        }

        Self::with_config(Config { coefficients: h })
    }
}

// MARK: Half-sample-delay table constructors (pure no_std)

macro_rules! half_sample_delay_impl {
    ($width:expr => [$($num:literal / $den:literal),* $(,)?]) => {
        // Denominators are powers of two; ratios are exact in IEEE 754.
        impl Convolve<f32, $width> {
            /// Returns a convolution filter pre-configured with Lagrange
            /// half-sample-delay coefficients for `δ = (M−1)/2`.
            ///
            /// The half-sample delay is a common special case useful for
            /// resampling and inter-sample timing correction. This
            /// constructor is pure `no_std`.
            ///
            /// With the project's newest-first ordering (`h[0]` pairs with
            /// the newest tap), the output at sample `n` approximates
            /// `x[n − δ] = x[n − (M−1)/2]`, i.e. centred between the two
            /// middle taps of the buffer.
            pub fn half_sample_delay() -> Self {
                // Denominators are powers of two; ratios are exact in IEEE 754.
                let coefficients = [$($num as f32 / $den as f32),*];
                Self::with_config(Config { coefficients })
            }
        }
        impl Convolve<f64, $width> {
            /// Returns a convolution filter pre-configured with Lagrange
            /// half-sample-delay coefficients for `δ = (M−1)/2`.
            ///
            /// With the project's newest-first ordering (`h[0]` pairs with
            /// the newest tap), the output at sample `n` approximates
            /// `x[n − δ] = x[n − (M−1)/2]`, i.e. centred between the two
            /// middle taps of the buffer.
            pub fn half_sample_delay() -> Self {
                Self::with_config(Config {
                    coefficients: [$(f64::from($num) / f64::from($den)),*]
                })
            }
        }
    };
}

// Rational coefficients are the closed-form Lagrange evaluations at δ = (M−1)/2,
// derivable by hand from the formula in the module documentation.

half_sample_delay_impl!(4 => [
    -1 / 16, 9 / 16, 9 / 16, -1 / 16
]);
half_sample_delay_impl!(6 => [
    3 / 256, -25 / 256, 75 / 128, 75 / 128, -25 / 256, 3 / 256
]);
half_sample_delay_impl!(8 => [
    -5 / 2048, 49 / 2048, -245 / 2048, 1225 / 2048,
    1225 / 2048, -245 / 2048, 49 / 2048, -5 / 2048
]);
half_sample_delay_impl!(10 => [
    35 / 65536, -405 / 65536, 567 / 16384, -2205 / 16384,
    19845 / 32768, 19845 / 32768, -2205 / 16384, 567 / 16384,
    -405 / 65536, 35 / 65536
]);

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use crate::traits::{ConfigRef, Filter};

    use super::*;

    fn collatz() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    // MARK: Integer-delta identity tests

    #[test]
    fn integer_delta_identity_m3() {
        for delta in 0..=2 {
            let filter = Convolve::<f32, 3>::lagrange(delta as f32);
            let c = filter.config_ref().coefficients;

            for (k, &val) in c.iter().enumerate() {
                if k == delta {
                    assert_abs_diff_eq!(val, 1.0, epsilon = 1e-5);
                } else {
                    assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn integer_delta_identity_m5() {
        for delta in 0..=4 {
            let filter = Convolve::<f32, 5>::lagrange(delta as f32);
            let c = filter.config_ref().coefficients;

            for (k, &val) in c.iter().enumerate() {
                if k == delta {
                    assert_abs_diff_eq!(val, 1.0, epsilon = 1e-5);
                } else {
                    assert_abs_diff_eq!(val, 0.0, epsilon = 1e-5);
                }
            }
        }
    }

    // MARK: Sum-to-one tests

    #[test]
    fn sum_to_one_m4() {
        for delta_int in 0..=10 {
            let delta = delta_int as f32 * 0.3;
            let filter = Convolve::<f32, 4>::lagrange(delta);
            let coeffs = filter.config_ref().coefficients;
            let sum: f32 = coeffs.iter().sum();

            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn sum_to_one_m6() {
        for delta_int in 0..=16 {
            let delta = delta_int as f32 * 0.3;
            let filter = Convolve::<f32, 6>::lagrange(delta);
            let coeffs = filter.config_ref().coefficients;
            let sum: f32 = coeffs.iter().sum();

            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn sum_to_one_f64_edge() {
        // Test near edges where ill-conditioning is worst.
        let filter1 = Convolve::<f64, 8>::lagrange(0.001);
        let sum1: f64 = filter1.config_ref().coefficients.iter().sum();
        assert_abs_diff_eq!(sum1, 1.0, epsilon = 1e-10);

        let filter2 = Convolve::<f64, 8>::lagrange(6.999);
        let sum2: f64 = filter2.config_ref().coefficients.iter().sum();
        assert_abs_diff_eq!(sum2, 1.0, epsilon = 1e-10);
    }

    // MARK: Linear-signal preservation

    #[test]
    fn linear_signal_preservation() {
        // x[n] = a + b*n, output should be a + b*(n - delta) after warm-up.
        let a = 3.0f32;
        let b = 2.5f32;
        let delta = 1.5f32;
        let m: usize = 4;
        let mut filter = Convolve::<f32, 4>::lagrange(delta);

        for n in 0..=20 {
            let x = a + b * (n as f32);
            let out = filter.filter(x);

            if n >= m {
                let expected = a + b * ((n as f32) - delta);
                assert_abs_diff_eq!(out, expected, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn lagrange_delta_0p3_matches_x_minus_0p3() {
        let mut f = Convolve::<f64, 4>::lagrange(0.3);
        for n in 0..40 {
            let x = n as f64;
            let y = f.filter(x);
            if n >= 16 {
                let expected = (n as f64) - 0.3;
                assert!(
                    (y - expected).abs() < 1e-12,
                    "n={n}: y={y} expected={expected}"
                );
            }
        }
    }

    // MARK: Phase-lag test (slow sinusoid)

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn phase_lag_slow_sinusoid() {
        let f = 0.05f64;
        let delta = 2.5f64;
        let m: usize = 6;
        let n_samples = 10_000;

        let mut filter = Convolve::<f64, 6>::lagrange(delta);

        // Warm up
        for n in 0..(m * 4) {
            let _ = filter.filter(f64::sin(2.0 * core::f64::consts::PI * f * (n as f64)));
        }

        // Measure phase in steady state via IQ demodulation.
        let mut i_acc = 0.0f64;
        let mut q_acc = 0.0f64;

        for n in 0..n_samples {
            let idx = (m * 4 + n) as f64;
            let phase = 2.0 * core::f64::consts::PI * f * idx;
            let x = f64::sin(phase);
            let y = filter.filter(x);

            i_acc += y * f64::sin(phase);
            q_acc += y * f64::cos(phase);
        }

        let measured_phase = f64::atan2(-q_acc, i_acc);

        // Expected phase: 2π · f · δ
        let expected_phase = 2.0 * core::f64::consts::PI * f * delta;

        // 1 mrad tolerance
        assert_abs_diff_eq!(measured_phase, expected_phase, epsilon = 1e-3);
    }

    // MARK: Smoke test

    #[test]
    fn smoke() {
        let filter = Convolve::<f32, 4>::lagrange(1.5);
        let input = collatz();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();

        #[rustfmt::skip]
        let expected = vec![
            0.0, -0.0625, 0.125, 4.375, 4.6875, 3.0, 6.1875, 12.375, 14.625, 16.625, 12.375, 9.5,
            12.0, 8.1875, 13.0, 18.3125, 10.0, 6.6875, 16.5, 21.3125, 13.5, 5.6875, 11.0, 15.8125,
            11.6875, 17.0, 11.0, 55.375, 156.3125, 153.9375, 53.0, 62.6875, 59.6875, 10.0, 20.8125,
            11.6875, 17.0, 21.5, 20.1875, 29.125, 15.5, 63.1875, 63.5, 13.0, 23.8125, 15.1875,
            10.5, 65.8125, 62.1875, 11.6875,
        ];

        assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
    }

    // MARK: f64 precision

    #[test]
    fn f64_half_sample_matches_f32() {
        let filter_f32 = Convolve::<f32, 4>::lagrange(1.5);
        let filter_f64 = Convolve::<f64, 4>::lagrange(1.5);

        let c32 = filter_f32.config_ref().coefficients;
        let c64 = filter_f64.config_ref().coefficients;

        for (a, b) in c32.iter().zip(c64.iter()) {
            assert_abs_diff_eq!(f64::from(*a), *b, epsilon = 1e-7);
        }
    }

    // MARK: Half-sample-delay table tests

    #[test]
    fn half_sample_m4_golden() {
        let filter = Convolve::<f32, 4>::half_sample_delay();
        let c = filter.config_ref().coefficients;

        assert_abs_diff_eq!(c[0], -0.0625, epsilon = f32::EPSILON);
        assert_abs_diff_eq!(c[1], 0.5625, epsilon = f32::EPSILON);
        assert_abs_diff_eq!(c[2], 0.5625, epsilon = f32::EPSILON);
        assert_abs_diff_eq!(c[3], -0.0625, epsilon = f32::EPSILON);
    }

    #[test]
    fn half_sample_m4_matches_runtime() {
        let table = Convolve::<f32, 4>::half_sample_delay();
        let runtime = Convolve::<f32, 4>::lagrange(1.5);

        let ct = table.config_ref().coefficients;
        let cr = runtime.config_ref().coefficients;

        for (a, b) in ct.iter().zip(cr.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
    }

    #[test]
    fn half_sample_m6_constructed() {
        let filter = Convolve::<f32, 6>::half_sample_delay();
        let coeffs = filter.config_ref().coefficients;
        let sum: f32 = coeffs.iter().sum();

        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn half_sample_m6_matches_runtime() {
        let table = Convolve::<f64, 6>::half_sample_delay();
        let runtime = Convolve::<f64, 6>::lagrange(2.5);

        let ct = table.config_ref().coefficients;
        let cr = runtime.config_ref().coefficients;

        for (a, b) in ct.iter().zip(cr.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn half_sample_m8_constructed() {
        let filter = Convolve::<f32, 8>::half_sample_delay();
        let coeffs = filter.config_ref().coefficients;
        let sum: f32 = coeffs.iter().sum();

        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn half_sample_m8_matches_runtime() {
        let table = Convolve::<f64, 8>::half_sample_delay();
        let runtime = Convolve::<f64, 8>::lagrange(3.5);

        let ct = table.config_ref().coefficients;
        let cr = runtime.config_ref().coefficients;

        for (a, b) in ct.iter().zip(cr.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn half_sample_m10_f32_bit_exact_with_runtime() {
        let table = Convolve::<f32, 10>::half_sample_delay();
        let runtime = Convolve::<f32, 10>::lagrange(4.5);
        for (a, b) in table
            .config_ref()
            .coefficients
            .iter()
            .zip(runtime.config_ref().coefficients.iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch: table={a:e} runtime={b:e}"
            );
        }
    }

    #[test]
    fn half_sample_m10_constructed() {
        let filter = Convolve::<f32, 10>::half_sample_delay();
        let coeffs = filter.config_ref().coefficients;
        let sum: f32 = coeffs.iter().sum();

        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn half_sample_m10_matches_runtime() {
        let table = Convolve::<f64, 10>::half_sample_delay();
        let runtime = Convolve::<f64, 10>::lagrange(4.5);

        let ct = table.config_ref().coefficients;
        let cr = runtime.config_ref().coefficients;

        for (a, b) in ct.iter().zip(cr.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn half_sample_smoke() {
        // Half-sample delay via table should match lagrange(1.5) output.
        let filter = Convolve::<f32, 4>::half_sample_delay();
        let input = collatz();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();

        let runtime_filter = Convolve::<f32, 4>::lagrange(1.5);
        let runtime_output: Vec<_> = collatz()
            .iter()
            .scan(runtime_filter, |f, &x| Some(f.filter(x)))
            .collect();

        assert_eq!(output.len(), runtime_output.len());
        for (a, b) in output.iter().zip(runtime_output.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
    }

    #[test]
    fn half_sample_f64_matches_f32_m4() {
        let filter_f32 = Convolve::<f32, 4>::half_sample_delay();
        let filter_f64 = Convolve::<f64, 4>::half_sample_delay();

        let c32 = filter_f32.config_ref().coefficients;
        let c64 = filter_f64.config_ref().coefficients;

        for (a, b) in c32.iter().zip(c64.iter()) {
            assert_abs_diff_eq!(f64::from(*a), *b, epsilon = 1e-7);
        }
    }

    // MARK: Out-of-range delta panic tests

    #[test]
    #[should_panic(expected = "delta must be non-negative")]
    fn lagrange_delta_negative_panics() {
        let _ = Convolve::<f64, 3>::lagrange(-0.1);
    }

    #[test]
    #[should_panic(expected = "delta must be <= M-1")]
    fn lagrange_delta_too_large_panics() {
        let _ = Convolve::<f64, 3>::lagrange(3.0);
    }
}
