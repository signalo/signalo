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
//! [`ConvolveArray::with_config`](crate::filters::fir::convolve::ConvolveArray::with_config)
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

use super::{Config, ConvolveArray};

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
    ///
    /// # Panics
    ///
    /// Panics if `M < 2`, `delta < 0`, or `delta > M−1`.
    fn lagrange(delta: T) -> Self;
}

#[cfg(any(feature = "libm", feature = "std"))]
impl<T: Float, const M: usize> FractionalDelay<T> for ConvolveArray<T, M> {
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
        impl ConvolveArray<f32, $width> {
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
                #[allow(clippy::cast_precision_loss)]
                let coefficients = [$($num as f32 / $den as f32),*];
                Self::with_config(Config { coefficients })
            }
        }
        impl ConvolveArray<f64, $width> {
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
mod tests;
