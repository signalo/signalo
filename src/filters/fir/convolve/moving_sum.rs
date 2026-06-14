// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving-sum FIR filters.
//!
//! A `MovingSum` is a bounded sliding sum over `N` samples:
//! `y[n] = Σ_{k=0}^{N-1} x[n-k]`.
//!
//! This is distinct from [`Integrate`](crate::filters::integrate::Integrate),
//! which is an IIR running sum (`y[n] = y[n-1] + x[n]`) with unbounded growth.
//! For a bounded sliding average, use [`Mean`](crate::filters::mean::mean::Mean),
//! which is essentially `MovingSum / N`.
//!
//! | Operator    | Formula                    | State | Boundedness             |
//! |-------------|----------------------------|-------|-------------------------|
//! | `Integrate` | `y[n] = y[n-1] + x[n]`     | O(1)  | Unbounded               |
//! | `MovingSum` | `y[n] = Σ x[n-k]`          | O(N)  | Bounded by `N · max│x│` |
//! | `Mean`      | `y[n] = (1/N) · MovingSum` | O(N)  | Bounded                 |
//!
//! # Coefficient ordering
//!
//! Coefficients are all-ones, so ordering is irrelevant. The filter sums
//! the `N` most recent input samples.

use num_traits::Num;

use crate::traits::WithConfig;

use super::{Config, Convolve};

/// Trait for moving-sum FIR convolution filters.
pub trait MovingSum: Sized {
    /// Returns a convolution filter pre-configured with all-ones coefficients,
    /// computing the bounded sliding sum over `N` samples.
    fn moving_sum() -> Self;
}

impl<T, const N: usize> MovingSum for Convolve<T, N>
where
    T: Num + Clone,
{
    fn moving_sum() -> Self {
        let coefficients = core::array::from_fn(|_| T::one());

        Self::with_config(Config { coefficients })
    }
}

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

    #[test]
    fn coefficients_are_all_ones() {
        let filter = Convolve::<f32, 5>::moving_sum();
        let coeffs = filter.config_ref().coefficients;

        for c in coeffs {
            assert_abs_diff_eq!(c, 1.0, epsilon = f32::EPSILON);
        }
    }

    #[test]
    fn coefficient_sum_equals_n() {
        let filter = Convolve::<f32, 5>::moving_sum();
        let coeffs = filter.config_ref().coefficients;
        let sum: f32 = coeffs.iter().sum();

        assert_abs_diff_eq!(sum, 5.0, epsilon = f32::EPSILON);
    }

    #[test]
    fn constant_signal_response() {
        // Feed constant 1.0 repeatedly; after warm-up, output → N.
        let mut filter = Convolve::<f32, 4>::moving_sum();

        for _ in 0..4 {
            let _ = filter.filter(1.0);
        }

        // After N=4 samples, the buffer is fully populated; output should be N.
        for _ in 0..4 {
            assert_abs_diff_eq!(filter.filter(1.0), 4.0, epsilon = f32::EPSILON);
        }
    }

    #[test]
    fn matches_mean_times_n() {
        // MovingSum<N> / N == arithmetic average of the last N samples.
        // Verify on a known sequence in steady state.
        let mut filter = Convolve::<f32, 3>::moving_sum();

        // Warm up with three 4.0 samples
        for _ in 0..3 {
            let _ = filter.filter(4.0);
        }

        // Steady state: MovingSum should be 12.0; divided by N (=3) the mean is 4.0
        let out = filter.filter(4.0);
        assert_abs_diff_eq!(out, 12.0, epsilon = f32::EPSILON);
        assert_abs_diff_eq!(out / 3.0, 4.0, epsilon = f32::EPSILON);

        // With values 2.0, 4.0, 6.0 the sum should be 12.0, avg 4.0
        let mut filter2 = Convolve::<f32, 3>::moving_sum();

        for _ in 0..3 {
            let _ = filter2.filter(0.0);
        }
        let _ = filter2.filter(2.0);
        let _ = filter2.filter(4.0);
        let out2 = filter2.filter(6.0);
        assert_abs_diff_eq!(out2, 12.0, epsilon = f32::EPSILON);
        assert_abs_diff_eq!(out2 / 3.0, 4.0, epsilon = f32::EPSILON);
    }

    #[test]
    fn smoke() {
        // Collatz sequence smoke test — compare against a golden vector.
        let filter = Convolve::<f32, 3>::moving_sum();
        let input = collatz();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();

        #[rustfmt::skip]
        let expected = vec![
            0.0, 1.0, 8.0, 10.0, 14.0, 15.0, 29.0, 37.0, 48.0, 38.0, 39.0, 29.0, 32.0, 35.0, 43.0,
            38.0, 33.0, 36.0, 52.0, 47.0, 34.0, 29.0, 37.0, 40.0, 48.0, 43.0, 144.0, 301.0, 399.0,
            306.0, 232.0, 129.0, 137.0, 44.0, 52.0, 47.0, 55.0, 63.0, 76.0, 63.0, 151.0, 125.0,
            146.0, 53.0, 61.0, 48.0, 136.0, 131.0, 139.0, 59.0,
        ];

        assert_abs_diff_eq!(output.as_slice(), expected.as_slice(), epsilon = 1e-5);
    }

    #[test]
    fn integer_moving_sum() {
        let mut filter = Convolve::<i32, 3>::moving_sum();

        assert_eq!(filter.filter(1), 1);
        assert_eq!(filter.filter(2), 3);
        assert_eq!(filter.filter(3), 6);
        assert_eq!(filter.filter(4), 9);
        assert_eq!(filter.filter(5), 12);
    }
}
