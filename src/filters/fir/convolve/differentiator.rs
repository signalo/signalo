// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! FIR differentiator filters.
//!
//! Central-difference FIR filters approximating the first derivative using
//! the Fornberg (1988) stencils. These stencils produce the mathematically
//! exact first derivative at the centre point for polynomial signals up to
//! degree `N − 1`.
//!
//! # Group delay
//!
//! The Fornberg central-difference stencil is centred on the middle tap. The
//! convolution is causal (`y[n] = Σ h[k]·x[n−k]`), so the output approximates
//! `f'(n − M)` where `M = (N−1)/2` is the half-width of the filter. For
//! example, with `N = 5` (`M = 2`), `filter(x[n])` returns an estimate of
//! `f'(n − 2)`, not `f'(n)`.
//!
//! Each filter is symmetric and DC-free (h\[k\] = −h\[N−1−k\]), with exactly
//! zero DC gain. The output is per-sample (dy/dn, not dy/dt). To obtain a
//! true `dy/dt`, multiply the output by the sample rate.
//!
//! # Constructors
//!
//! Use [`FirDifferentiator::central_difference()`] as the preferred constructor
//! for `N ∈ {3, 5, 7, 9}` — it returns exact pre-computed Fornberg coefficients
//! from closed-form tables.
//!
//! For the long tail (`3 ≤ N ≤ 19`, odd), use the inherent runtime
//! constructor `Convolve::<f, N>::central_difference_runtime()`.
//! For `f32`, the runtime constructor is restricted to `N ≤ 9` due to
//! mantissa precision limits; for `N ≥ 11`, use `f64`. At
//! `N ∈ {3, 5, 7, 9}`, the runtime and table constructors produce
//! bit-identical coefficients for both `f32` and `f64`.
//!
//! # Boundary effects
//!
//! See [`ConvolveArray`] for cold-start behaviour: the tap buffer is pre-filled
//! with `N` zeros, so the first `N − 1` outputs are biased by implicit
//! zero-padding `x[n] = 0` for `n < 0`.
//!
//! # Related
//!
//! - [`Differentiate`](crate::filters::differentiate::Differentiate) for
//!   the O(1)-state two-tap backward difference (`h = [-1, 1]`).
//! - `laplacian()` and `second_central_difference()` for the second
//!   derivative.
//! - [`savitzky_golay`](super::savitzky_golay) for Savitzky-Golay polynomial
//!   smoothing.
//!
//! # Coefficient ordering
//!
//! `h[0]` pairs with the newest sample (see
//! [`ConvolveArray::filter`](super::ConvolveArray::filter)).

use crate::traits::WithConfig;

use super::{Config, ConvolveArray};

/// Trait for first-order central-difference FIR filters.
///
/// Implements the Fornberg (1988) central-difference stencil for the first
/// derivative. These filters have exactly zero DC gain and produce per-sample
/// differences (dy/dn). Multiply by the sample rate for a true dy/dt.
///
/// This trait is implemented for `ConvolveArray<T, N>` with `N ∈ {3, 5, 7, 9}`,
/// using pre-computed closed-form Fornberg tables. For higher odd `N` up to
/// 19, use [`ConvolveArray::central_difference_runtime()`] instead.
pub trait FirDifferentiator: Sized {
    /// Returns a convolution filter with the Fornberg central-difference
    /// coefficients for the first derivative.
    ///
    /// Output corresponds to the derivative at the centre tap, i.e.
    /// `n − (N−1)/2`.
    fn central_difference() -> Self;
}

macro_rules! central_difference_impl {
    ($width:expr => [$($num:literal / $den:literal),* $(,)?]) => {
        #[allow(clippy::cast_precision_loss)]
        impl FirDifferentiator for ConvolveArray<f32, $width> {
            fn central_difference() -> Self {
                Self::with_config(Config {
                    coefficients: [$($num as f32 / $den as f32),*]
                })
            }
        }
        impl FirDifferentiator for ConvolveArray<f64, $width> {
            fn central_difference() -> Self {
                Self::with_config(Config {
                    coefficients: [$(f64::from($num) / f64::from($den)),*]
                })
            }
        }
    };
}

// Fornberg (1988) central-difference stencils for the first derivative.
// Fornberg, B. (1988). "Generation of finite difference formulas on
// arbitrarily spaced grids." Mathematics of Computation 51(184), 699–706.

central_difference_impl!(3 => [
    1 / 2,   0 / 1,   -1 / 2
]);
central_difference_impl!(5 => [
    -1 / 12,  8 / 12,   0 / 1,  -8 / 12,   1 / 12
]);
central_difference_impl!(7 => [
     1 / 60, -9 / 60,  45 / 60,   0 / 1,  -45 / 60,   9 / 60,  -1 / 60
]);
central_difference_impl!(9 => [
    -3 / 840, 32 / 840, -168 / 840, 672 / 840, 0 / 1, -672 / 840, 168 / 840, -32 / 840, 3 / 840
]);

// MARK: Runtime constructor

macro_rules! central_difference_runtime_impl {
    ($float:ty) => {
        impl<const N: usize> ConvolveArray<$float, N> {
            /// Returns a convolution filter with Fornberg central-difference
            /// coefficients for the first derivative, computed at runtime for
            /// arbitrary odd `N` (3 ≤ N ≤ 19).
            ///
            /// For `N ∈ {3, 5, 7, 9}`, prefer
            /// [`FirDifferentiator::central_difference()`] — it uses exact
            /// pre-computed coefficients from closed-form tables and avoids
            /// runtime factorial computation.
            ///
            /// # Scale
            ///
            /// Output is per-sample. Multiply by the sample rate for a true
            /// `dy/dt`. Output corresponds to the derivative at the centre
            /// tap, i.e. `n − (N−1)/2`.
            ///
            /// # Precision
            ///
            /// For `f32` at `N ≥ 11`, the `u64 → f32` cast incurs rounding error
            /// because factorials exceed the `f32` mantissa. For `N = 19` the
            /// largest factorial is `18! ≈ 6.4·10^15`, which exceeds the f32
            /// mantissa precision (2^24); f32 coefficients at high `N` will be
            /// rounded. Prefer `f64` for exact coefficients at higher orders.
            ///
            /// The `f32` runtime constructor is gated to `N ≤ 9`.
            /// For `N ≤ 9`, outputs are identical to the table-lookup constructors.
            /// If you need `N ≥ 11` and better than 6-digit accuracy, prefer the `f64`
            /// instantiation.
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            pub fn central_difference_runtime() -> Self {
                assert!(N % 2 == 1, "central_difference_runtime requires odd N");
                assert!(N >= 3, "central_difference_runtime requires N >= 3");
                assert!(
                    N <= 19,
                    "central_difference_runtime limited to N <= 19 (u64 overflow guard)"
                );
                assert!(
                    N <= 9 || core::mem::size_of::<$float>() > 4,
                    "central_difference_runtime: for f32, N must be <= 9 due to precision limits; use f64 for N >= 11"
                );

                let m = (N - 1) / 2;

                // Precompute factorials up to 2*m. 20! fits in u64.
                let mut factorial = [1u64; 21];
                for i in 1..=2 * m {
                    factorial[i] = factorial[i - 1] * (i as u64);
                }

                let m_fact = factorial[m];
                let m_fact_sq = m_fact * m_fact;

                // signed_coeff[i] = coefficient at signed position (i as isize) - (m as isize)
                let mut signed_coeff = [0.0; N];

                for j in 1..=m {
                    // Fornberg (1988) closed form:
                    //   c_j = (-1)^{j+1} · (m!)² / (j · (m+j)! · (m−j)!)
                    let sign = if j % 2 == 0 { -1.0 } else { 1.0 };
                    let numerator = m_fact_sq as $float;
                    let denominator = ((j as u64) * factorial[m + j] * factorial[m - j]) as $float;
                    let val = numerator / denominator;

                    let cj = (sign as $float) * val;
                    signed_coeff[m + j] = cj;
                    signed_coeff[m - j] = -cj;
                }

                // Convert signed-position array to convolution order:
                // h[k] = c_{m-k} = signed_coeff[N - 1 - k]
                let mut coefficients = [0.0; N];
                for k in 0..N {
                    coefficients[k] = signed_coeff[N - 1 - k];
                }

                Self::with_config(Config { coefficients })
            }
        }
    };
}

central_difference_runtime_impl!(f32);
central_difference_runtime_impl!(f64);

// MARK: Laplacian (second-order differentiator)

macro_rules! laplacian_impl {
    ($float:ty) => {
        impl ConvolveArray<$float, 3> {
            /// Returns a 3-tap Laplacian filter with coefficients `[1, -2, 1]`.
            ///
            /// The Laplacian approximates the second derivative. For a signal
            /// `f(n)`, the output approximates `f″(n)` with per-sample scaling.
            /// Multiply by `sample_rate²` for a true `d²y/dt²`.
            /// Output corresponds to the derivative at the centre tap, i.e.
            /// `n − (N−1)/2`.
            pub fn laplacian() -> Self {
                Self::with_config(Config {
                    coefficients: [1.0 as $float, -2.0 as $float, 1.0 as $float],
                })
            }
        }
    };
}

laplacian_impl!(f32);
laplacian_impl!(f64);

macro_rules! second_central_difference_impl {
    ($float:ty) => {
        impl ConvolveArray<$float, 5> {
            /// Returns a 5-tap second-order central-difference filter
            /// approximating the second derivative.
            ///
            /// Coefficients: `[-1/12, 4/3, -5/2, 4/3, -1/12]`.
            /// Fornberg (1988) stencil for the second derivative.
            /// Output corresponds to the derivative at the centre tap, i.e.
            /// `n − (N−1)/2`.
            pub fn second_central_difference() -> Self {
                Self::with_config(Config {
                    coefficients: [
                        -1.0 as $float / 12.0 as $float,
                        4.0 as $float / 3.0 as $float,
                        -5.0 as $float / 2.0 as $float,
                        4.0 as $float / 3.0 as $float,
                        -1.0 as $float / 12.0 as $float,
                    ],
                })
            }
        }
    };
}

second_central_difference_impl!(f32);
second_central_difference_impl!(f64);

#[cfg(test)]
mod tests;
