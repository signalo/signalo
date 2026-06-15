// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Savitzky-Golay filters.
//!
//! For N=1, the Savitzky-Golay filter reduces to the identity. For N=2,
//! the rightmost-sample prediction of a linear fit through 2 points is also
//! the identity (x\[n\] is recovered exactly).

#![allow(clippy::wildcard_imports)]

use crate::traits::WithConfig;

use super::convolve::{Config, Convolve};

/// Trait for Savitzky-Golay convolution filters.
pub trait SavitzkyGolay: Sized {
    /// Creates a convolution filter pre-configured with the Savitzky-Golay coefficients.
    fn savitzky_golay() -> Self;
}

macro_rules! savitzky_golay_impl_float {
    ($width:expr => [$($num:literal / $den:literal),* $(,)?]) => {
        #[allow(clippy::cast_precision_loss)]
        impl SavitzkyGolay for Convolve<f32, $width> {
            fn savitzky_golay() -> Self {
                Self::with_config(Config {
                    coefficients: [$($num as f32 / $den as f32),*]
                })
            }
        }
        impl SavitzkyGolay for Convolve<f64, $width> {
            fn savitzky_golay() -> Self {
                Self::with_config(Config {
                    coefficients: [$(f64::from($num) / f64::from($den)),*]
                })
            }
        }
    };
}

// Coefficients: polynomial order = 2 for N ≥ 3, rightmost-sample prediction.
// Numerators follow the pattern: start at (3N² − 9N + 10)/2, decrease by N−1
// each step, wrapping through zero to negatives.
// Denominator = sum of numerators for each N (guarantees DC gain = 1).
// Verified against scipy.signal.savgol_coeffs(N, 2, pos=N-1, use='dot').
//
// Source for original coefficients:
// https://gregstanleyandassociates.com/whitepapers/FaultDiagnosis/Filtering/LeastSquares-Filter/leastsquares-filter.htm

savitzky_golay_impl_float!(1 => [
    1 / 1
]);
savitzky_golay_impl_float!(2 => [
    // N=2: linear fit through 2 points is exact; prediction at rightmost = x[n].
    1 / 1, 0 / 1
]);
savitzky_golay_impl_float!(3 => [
    5 / 6, 2 / 6, -1 / 6
]);
savitzky_golay_impl_float!(4 => [
    7 / 10, 4 / 10, 1 / 10, -2 / 10
]);
savitzky_golay_impl_float!(5 => [
    3 / 5, 2 / 5, 1 / 5, 0 / 1, -1 / 5
]);
savitzky_golay_impl_float!(6 => [
    11 / 21, 8 / 21, 5 / 21, 2 / 21, -1 / 21, -4 / 21
]);
savitzky_golay_impl_float!(7 => [
    13 / 28, 10 / 28, 7 / 28, 4 / 28, 1 / 28, -2 / 28, -5 / 28
]);
savitzky_golay_impl_float!(8 => [
    5 / 12, 4 / 12, 3 / 12, 2 / 12, 1 / 12, 0 / 1, -1 / 12, -2 / 12
]);
savitzky_golay_impl_float!(9 => [
    17 / 45, 14 / 45, 11 / 45, 8 / 45, 5 / 45, 2 / 45, -1 / 45, -4 / 45, -7 / 45
]);
savitzky_golay_impl_float!(10 => [
    19 / 55, 16 / 55, 13 / 55, 10 / 55, 7 / 55, 4 / 55, 1 / 55, -2 / 55, -5 / 55, -8 / 55
]);
savitzky_golay_impl_float!(11 => [
    21 / 66, 18 / 66, 15 / 66, 12 / 66, 9 / 66, 6 / 66, 3 / 66, 0 / 1, -3 / 66, -6 / 66,
    -9 / 66
]);
savitzky_golay_impl_float!(12 => [
    23 / 78, 20 / 78, 17 / 78, 14 / 78, 11 / 78, 8 / 78, 5 / 78, 2 / 78, -1 / 78, -4 / 78,
    -7 / 78, -10 / 78
]);
savitzky_golay_impl_float!(13 => [
    25 / 91, 22 / 91, 19 / 91, 16 / 91, 13 / 91, 10 / 91, 7 / 91, 4 / 91, 1 / 91, -2 / 91,
    -5 / 91, -8 / 91, -11 / 91
]);

#[cfg(test)]
mod tests;
