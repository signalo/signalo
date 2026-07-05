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
/// below `T::epsilon()`, with a safety cap of 64 iterations.
///
/// # Limits
///
/// The series iteration is capped at 64 terms, which is sufficient for
/// `|x| ≤ 20`. For larger arguments, the function will `debug_assert!`
/// (panic in debug builds) with a message indicating the limit.
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
            return sum;
        }
    }
    debug_assert!(
        term / sum < eps,
        "bessel_i0: series did not converge for x; |x| <= 20 is supported",
    );
    sum
}

/// Asserts `den` is finite and `|den| ≥ T::min_positive_value().sqrt()`,
/// then returns it unchanged.
///
/// `min_positive_value().sqrt()` gives an aggressive but safe lower bound:
/// dividing any value whose magnitude is `≤ T::max_value()` by `den` will
/// stay finite. For `f32` the threshold is ≈ 1.08e−19, for `f64` ≈ 1.49e−154.
///
/// # Panics
///
/// Panics if `den` is not finite or `|den|` is below the safe floor.
#[cfg(any(feature = "libm", feature = "std"))]
#[must_use]
pub fn safe_normalise_divisor<T: Float + core::fmt::Debug>(den: T, msg: &'static str) -> T {
    assert!(
        den.is_finite(),
        "{msg}: denominator must be finite (got {den:?})"
    );
    let floor = T::min_positive_value().sqrt();
    let den_abs = den.abs();
    assert!(
        den_abs >= floor,
        "{msg}: denominator magnitude {den_abs:?} below safe floor {floor:?}",
    );
    den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    #[should_panic(expected = "floor")]
    fn safe_normalise_divisor_rejects_subnormal() {
        let _ = safe_normalise_divisor(f32::from_bits(1), "test");
    }

    #[cfg(any(feature = "libm", feature = "std"))]
    #[test]
    fn safe_normalise_divisor_accepts_one() {
        assert_eq!(safe_normalise_divisor(1.0_f32, "test"), 1.0);
    }
}
