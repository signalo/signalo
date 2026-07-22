// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Mathematical utility functions for DSP operations.

pub mod phase;

use core::ops::{Add, AddAssign, Sub, SubAssign};

use num_traits::Zero;

#[cfg(any(feature = "libm", feature = "std"))]
use num_traits::Float;

/// Transposes a row-major matrix into a separate output slice.
///
/// `input` has `height` rows and `width` columns. `output` receives the
/// transposed matrix, which has `width` rows and `height` columns. Both slices
/// must have length `width * height`. Supports rectangular matrices, not just
/// square ones.
///
/// # Panics
///
/// Panics when either slice length is not `width * height`, including
/// multiplication overflow.
pub(crate) fn matrix_transpose<T: Clone>(
    output: &mut [T],
    input: &[T],
    width: usize,
    height: usize,
) {
    let Some(count) = width.checked_mul(height) else {
        panic!("matrix_transpose: width * height overflowed");
    };
    assert_eq!(
        input.len(),
        count,
        "matrix_transpose: input length must equal width * height"
    );
    assert_eq!(
        output.len(),
        count,
        "matrix_transpose: output length must equal width * height"
    );

    for y in 0..height {
        for x in 0..width {
            output[x * height + y] = input[y * width + x].clone();
        }
    }
}

/// Transposes a row-major matrix prefix into a separate output slice.
///
/// `input` is treated as the prefix of a row-major matrix with `height` rows and
/// `width` columns. `output` receives the transposed matrix, which has `width`
/// rows and `height` columns. Missing input elements are filled with `padding`.
/// The output slice must have length `width * height`, and `input.len()` must
/// not exceed that length.
///
/// # Panics
///
/// Panics when `output.len() != width * height`, `input.len() > width * height`,
/// or if `width * height` overflows.
pub(crate) fn matrix_transpose_padded<T: Clone>(
    output: &mut [T],
    input: &[T],
    width: usize,
    height: usize,
    padding: T,
) {
    let Some(count) = width.checked_mul(height) else {
        panic!("matrix_transpose_padded: width * height overflowed");
    };
    assert!(
        input.len() <= count,
        "matrix_transpose_padded: input length must be <= width * height"
    );
    assert_eq!(
        output.len(),
        count,
        "matrix_transpose_padded: output length must equal width * height"
    );
    if input.len() == count {
        matrix_transpose(output, input, width, height);
        return;
    }

    output.fill(padding);
    for (src, value) in input.iter().enumerate() {
        let x = src % width;
        let y = src / width;
        output[x * height + y] = value.clone();
    }
}

/// Transposes a row-major matrix in place.
///
/// The matrix has `height` rows and `width` columns, and `matrix.len()` must be
/// `width * height`. Supports rectangular matrices, not just square ones.
///
/// This uses a rotation-based in-place algorithm that avoids a full auxiliary
/// matrix. It is not linear-time: for large matrices, prefer
/// [`matrix_transpose`] when a separate output buffer is available.
///
/// # Complexity
///
/// Performs `O(width^2 * height^2 / 4)` element moves.
///
/// # Panics
///
/// Panics when `width * height != matrix.len()`, including multiplication
/// overflow.
pub(crate) fn matrix_transpose_in_place<T>(matrix: &mut [T], width: usize, height: usize) {
    let Some(count) = width.checked_mul(height) else {
        panic!("matrix_transpose_in_place: width * height overflowed");
    };
    assert_eq!(
        matrix.len(),
        count,
        "matrix_transpose_in_place: matrix length must equal width * height"
    );

    // Each inner iteration rotates a window of `step` elements ending at `last`
    // (exclusive). `step` starts at 1 and grows by `width - x - 1` per row,
    // advancing the rotation boundary through the transposition permutation for
    // column group `x`.
    for x in 0..width {
        let count_adjustment = width - x - 1;
        let mut step = 1;
        for y in 0..height {
            let last = count - (y + x * height);
            let first = last - step;
            matrix[first..last].rotate_left(1);
            step += count_adjustment;
        }
    }
}

/// Kahan compensated summation accumulator.
///
/// Tracks a running sum plus a compensation term that estimates floating-point
/// rounding error. This is useful for long-running accumulators that repeatedly
/// add small deltas, such as rolling sums and numerically integrated control
/// terms.
///
/// For integer types this behaves like ordinary summation until arithmetic
/// overflow semantics apply, with additional overhead and no numerical benefit.
/// For arbitrary numeric types, the usual Kahan error model only applies when
/// their arithmetic behaves like rounded floating-point arithmetic.
#[derive(Clone, Debug)]
pub struct KahanSum<T> {
    sum: T,
    compensation: T,
}

impl<T> Default for KahanSum<T>
where
    T: Zero,
{
    fn default() -> Self {
        Self {
            sum: T::zero(),
            compensation: T::zero(),
        }
    }
}

impl<T> KahanSum<T>
where
    T: Clone + Add<T, Output = T> + Sub<T, Output = T>,
{
    /// Add one value to the running sum.
    pub fn add(&mut self, value: T) {
        let y = value - self.compensation.clone();
        let t = self.sum.clone() + y.clone();
        self.compensation = (t.clone() - self.sum.clone()) - y;
        self.sum = t;
    }

    /// Subtract one value from the running sum.
    pub fn subtract(&mut self, value: T)
    where
        T: Zero,
    {
        self.add(T::zero() - value);
    }

    /// Return the current compensated running sum.
    #[must_use]
    pub fn value(&self) -> T {
        self.sum.clone()
    }
}

impl<T> KahanSum<T>
where
    T: Zero,
{
    /// Reset the accumulator to zero.
    pub fn reset(&mut self) {
        self.sum = T::zero();
        self.compensation = T::zero();
    }
}

impl<T> AddAssign<T> for KahanSum<T>
where
    T: Clone + Add<T, Output = T> + Sub<T, Output = T>,
{
    fn add_assign(&mut self, rhs: T) {
        self.add(rhs);
    }
}

impl<T> SubAssign<T> for KahanSum<T>
where
    T: Clone + Add<T, Output = T> + Sub<T, Output = T> + Zero,
{
    fn sub_assign(&mut self, rhs: T) {
        self.subtract(rhs);
    }
}

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

/// Error function support for floating-point types.
#[cfg(feature = "libm")]
pub trait Erf: Float {
    /// Computes the error function.
    #[must_use]
    fn erf(self) -> Self;
}

#[cfg(feature = "libm")]
impl Erf for f32 {
    #[inline]
    fn erf(self) -> Self {
        libm::erff(self)
    }
}

#[cfg(feature = "libm")]
impl Erf for f64 {
    #[inline]
    fn erf(self) -> Self {
        libm::erf(self)
    }
}

/// Computes the error function.
#[cfg(feature = "libm")]
#[must_use]
#[inline]
pub fn erf<T: Erf>(x: T) -> T {
    x.erf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn kahan_sum_accumulates_values() {
        let mut sum = KahanSum::default();

        sum += 1.0_f32;
        sum += 2.0;
        sum += 3.0;

        assert_eq!(sum.value(), 6.0);
    }

    #[test]
    fn kahan_sum_subtracts_values() {
        let mut sum = KahanSum::default();

        sum += 10.0_f32;
        sum.subtract(3.0);
        sum -= 2.0;

        assert_eq!(sum.value(), 5.0);
    }

    #[test]
    fn kahan_sum_reduces_repeated_small_increment_error() {
        let mut kahan = KahanSum::default();
        let mut plain = 0.0_f32;
        for _ in 0..10_000 {
            kahan += 0.1;
            plain += 0.1;
        }

        let expected = 1_000.0_f32;
        assert!((kahan.value() - expected).abs() < (plain - expected).abs());
    }

    #[test]
    fn kahan_sum_reset_clears_state() {
        let mut sum = KahanSum::default();
        sum.add(1.0_f32);

        sum.reset();

        assert_eq!(sum.value(), 0.0);
    }

    #[test]
    fn matrix_transpose_padded_handles_short_input() {
        let input = [1, 2, 3, 4, 5];
        let mut output = [99; 6];

        matrix_transpose_padded(&mut output, &input, 3, 2, 0);

        assert_eq!(output, [1, 4, 2, 5, 3, 0]);
    }

    #[test]
    fn matrix_transpose_padded_handles_larger_output() {
        let input = [1, 2, 3, 4, 5];
        let mut output = [99; 9];

        matrix_transpose_padded(&mut output, &input, 3, 3, 0);

        assert_eq!(output, [1, 4, 0, 2, 5, 0, 3, 0, 0]);
    }

    #[test]
    fn matrix_transpose_handles_rectangular_matrix() {
        let input = [1, 2, 3, 4, 5, 6];
        let mut output = [0; 6];

        matrix_transpose(&mut output, &input, 3, 2);

        assert_eq!(output, [1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn matrix_transpose_handles_square_matrix() {
        let input = [1, 2, 3, 4];
        let mut output = [0; 4];

        matrix_transpose(&mut output, &input, 2, 2);

        assert_eq!(output, [1, 3, 2, 4]);
    }

    #[test]
    fn matrix_transpose_in_place_handles_rectangular_matrix() {
        let mut matrix = [1, 2, 3, 4, 5, 6];

        matrix_transpose_in_place(&mut matrix, 3, 2);

        assert_eq!(matrix, [1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn matrix_transpose_in_place_handles_square_matrix() {
        let mut matrix = [1, 2, 3, 4];

        matrix_transpose_in_place(&mut matrix, 2, 2);

        assert_eq!(matrix, [1, 3, 2, 4]);
    }

    #[test]
    fn matrix_transpose_in_place_matches_out_of_place() {
        for (width, height) in [(4, 3), (3, 4), (1, 8), (8, 1), (4, 4), (3, 5)] {
            let input: Vec<_> = (1..=width * height).collect();
            let mut expected = vec![0; width * height];
            matrix_transpose(&mut expected, &input, width, height);

            let mut actual = input.clone();
            matrix_transpose_in_place(&mut actual, width, height);

            assert_eq!(actual, expected, "width={width} height={height}");
        }
    }

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
