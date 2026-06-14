// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Mathematical utility functions for DSP operations.

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

/// Modified Bessel function of the first kind, order 0.
///
/// Computed via the ascending power series:
///
/// ```text
/// I₀(x) = Σ_{m=0}^{∞} (x/2)^(2m) / (m!)²
/// ```
///
/// The series converges quickly for all reasonable `x`. Iteration
/// stops when the relative contribution of the current term drops
/// below `T::epsilon()`, with a safety cap of 64 iterations
/// (sufficient for β up to ≈ 20, giving I₀(20) ≈ 4.4 × 10⁸).
///
/// # Panics
///
/// Panics if `T::from` conversions from standard f64 literals fail
/// (impossible for any `Float`-implementing type).
#[cfg(any(feature = "libm", feature = "std"))]
#[allow(clippy::unwrap_used)]
pub fn bessel_i0<T: Float>(x: T) -> T {
    let eps = T::epsilon();
    let mut sum = T::one();
    let mut term = T::one();
    let x_half = x / T::from(2.0).unwrap();
    let x_half_sq = x_half * x_half;
    let mut m = 0;
    while m < 64 {
        term = term * x_half_sq / T::from((m + 1) * (m + 1)).unwrap();
        sum = sum + term;
        m += 1;
        if term / sum < eps {
            break;
        }
    }
    sum
}
