// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Savitzky-Golay filters.

#![allow(clippy::wildcard_imports)]

use crate::traits::WithConfig;

use super::{Config, Convolve};

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
mod tests {
    #![allow(clippy::unreadable_literal)]

    use alloc::vec;
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use crate::traits::Filter;

    use super::*;

    fn get_input() -> Vec<f32> {
        // Numeric test fixture (not a true Collatz subsequence).
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output_1() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output_2() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output_3() -> Vec<f32> {
        vec![
            0.0, 0.8333333, 6.166667, 3.833333, 3.666667, 8.0, 15.16667, 14.83333, 17.5, 9.166667,
            10.5, 11.16667, 8.166667, 15.66667, 18.33333, 6.166667, 8.5, 20.0, 21.33333, 9.166667,
            4.833333, 13.66667, 16.33333, 10.83333, 20.0, 14.33333, 92.0, 185.3333, 131.5, 21.0,
            76.33333, 36.5, 5.666667, 18.66667, 10.83333, 19.66667, 22.33333, 21.0, 31.83333, 14.5,
            87.83333, 41.66667, 8.666667, 21.66667, 13.83333, 16.0, 89.33333, 41.16667, 6.333333,
            26.16667,
        ]
    }

    fn get_output_4() -> Vec<f32> {
        vec![
            0.0, 0.7, 5.3, 4.3, 4.8, 6.4, 14.5, 15.3, 18.5, 9.9, 11.5, 8.7, 10.1, 13.6, 17.8, 9.5,
            8.3, 15.8, 22.4, 12.5, 5.7, 10.0, 15.8, 13.1, 18.6, 14.2, 82.0, 166.8, 156.7, 51.6,
            56.2, 26.1, 27.2, -1.2, 15.9, 16.0, 21.8, 22.6, 30.1, 17.1, 78.7, 43.2, 32.8, 1.8,
            18.9, 13.4, 77.6, 47.7, 28.4, 6.7,
        ]
    }

    fn get_output_5() -> Vec<f32> {
        vec![
            0.0, 0.6, 4.6, 4.2, 5.2, 7.0, 12.4, 15.4, 18.8, 12.2, 11.4, 9.6, 8.0, 14.4, 16.0, 10.8,
            10.4, 14.2, 19.0, 15.4, 8.6, 9.2, 12.4, 13.6, 19.4, 14.2, 72.2, 152.4, 154.4, 88.0,
            70.2, 13.0, 17.2, 15.6, -3.0, 19.4, 18.4, 22.6, 30.4, 18.4, 71.2, 45.8, 35.6, 21.2,
            0.0, 17.6, 66.2, 48.2, 36.4, 23.0,
        ]
    }

    fn get_output_6() -> Vec<f32> {
        vec![
            0.0, 0.5238095, 4.047619, 3.952381, 5.142857, 7.190476, 12.28571, 13.85714, 18.85714,
            13.66667, 13.09524, 9.619048, 8.666667, 11.90476, 16.57143, 10.38095, 11.33333,
            15.09524, 17.28571, 13.7619, 11.66667, 10.85714, 11.14286, 11.04762, 19.19048, 15.7619,
            64.80952, 137.8095, 149.5238, 99.14286, 98.04762, 27.85714, 3.047619, 6.571429,
            9.952381, 1.095238, 21.14286, 19.66667, 29.71429, 20.66667, 65.2381, 45.85714,
            39.33333, 24.85714, 15.42857, -0.0952381, 63.0, 44.42857, 39.2381, 30.42857,
        ]
    }

    fn get_output_7() -> Vec<f32> {
        vec![
            0.0, 0.4642857, 3.607143, 3.678571, 4.928571, 7.035714, 12.0, 13.85714, 17.39286,
            14.67857, 14.35714, 11.28571, 8.642857, 11.89286, 14.17857, 11.71429, 10.85714,
            15.35714, 17.78571, 13.0, 10.71429, 13.21429, 12.21429, 10.0, 16.32143, 16.28571, 60.5,
            125.6071, 141.0357, 104.7143, 107.5, 56.82143, 14.10714, -7.5, 1.107143, 10.75, 4.0,
            22.07143, 26.42857, 21.60714, 61.89286, 45.0, 40.67857, 29.46429, 18.78571, 12.35714,
            41.28571, 45.60714, 37.60714, 33.89286,
        ]
    }

    fn get_output_8() -> Vec<f32> {
        vec![
            0.0, 0.4166667, 3.25, 3.416667, 4.666667, 6.75, 11.5, 13.66667, 17.25, 14.08333,
            15.33333, 12.66667, 10.16667, 11.41667, 13.91667, 10.16667, 12.0, 14.5, 17.83333, 14.0,
            10.33333, 12.16667, 14.16667, 11.0, 14.75, 14.16667, 56.5, 116.75, 132.5833, 104.5,
            112.8333, 70.25, 40.08333, 1.0, -13.08333, 1.583333, 11.5, 6.166667, 28.0, 19.75,
            58.58333, 45.16667, 40.91667, 31.91667, 23.33333, 15.16667, 47.33333, 28.41667,
            39.91667, 33.25,
        ]
    }

    fn get_output_9() -> Vec<f32> {
        vec![
            0.0, 0.3777778, 2.955556, 3.177778, 4.4, 6.422222, 10.93333, 13.24444, 16.95556,
            14.48889, 14.86667, 13.82222, 11.53333, 12.46667, 13.26667, 10.33333, 10.55556, 15.2,
            16.88889, 14.51111, 11.48889, 11.62222, 13.13333, 12.88889, 15.13333, 12.97778,
            50.88889, 108.8, 125.9556, 102.3778, 113.0889, 79.57778, 53.55556, 24.44444, -6.888889,
            -12.95556, 2.311111, 12.17778, 12.75556, 22.0, 53.48889, 44.55556, 41.86667, 33.2,
            26.11111, 19.35556, 46.17778, 35.22222, 24.93333, 36.02222,
        ]
    }

    fn get_output_10() -> Vec<f32> {
        vec![
            0.0, 0.3454545, 2.709091, 2.963636, 4.145455, 6.090909, 10.36364, 12.72727, 16.45455,
            14.65455, 15.29091, 13.63636, 12.74545, 13.50909, 14.05455, 10.05455, 10.63636, 13.6,
            17.38182, 14.01818, 12.2, 12.52727, 12.52727, 12.07273, 16.49091, 13.49091, 46.65455,
            99.85455, 119.4545, 100.5636, 111.4727, 83.69091, 63.70909, 37.63636, 14.0, -8.927273,
            -12.25455, 3.127273, 17.27273, 8.472727, 52.70909, 41.69091, 41.96364, 34.98182,
            27.83636, 22.12727, 47.09091, 35.4, 31.34545, 22.63636,
        ]
    }

    fn get_output_11() -> Vec<f32> {
        vec![
            0.0, 0.3181818, 2.5, 2.772727, 3.909091, 5.772727, 9.818182, 12.18182, 15.86364,
            14.54545, 15.5, 14.22727, 12.72727, 14.5, 14.90909, 11.04545, 10.31818, 13.40909,
            15.77273, 14.77273, 11.95455, 13.09091, 13.27273, 11.59091, 15.45455, 14.90909,
            44.45455, 92.54545, 111.5455, 97.90909, 110.0, 85.45455, 69.13636, 48.18182, 26.40909,
            9.409091, -9.863636, -11.27273, 8.090909, 12.72727, 37.68182, 42.36364, 39.81818,
            35.86364, 30.0, 24.0, 47.31818, 37.27273, 31.81818, 28.59091,
        ]
    }

    fn get_output_12() -> Vec<f32> {
        vec![
            0.0, 0.2948718, 2.320513, 2.602564, 3.692308, 5.474359, 9.307692, 11.64103, 15.24359,
            14.28205, 15.44872, 14.60256, 13.41026, 14.39744, 15.76923, 12.08974, 11.19231,
            12.88462, 15.44872, 13.52564, 12.83333, 12.79487, 13.74359, 12.34615, 14.74359,
            14.10256, 43.51282, 87.48718, 104.7949, 93.20513, 107.641, 86.82051, 72.26923,
            54.48718, 36.78205, 20.67949, 6.320513, -10.12821, -6.333333, 4.25641, 39.24359,
            29.41026, 40.82051, 34.52564, 31.30769, 26.30769, 47.14103, 38.38462, 33.84615,
            29.19231,
        ]
    }

    fn get_output_13() -> Vec<f32> {
        vec![
            0.0, 0.2747253, 2.164835, 2.450549, 3.494505, 5.197802, 8.835165, 11.12088, 14.62637,
            13.93407, 15.24176, 14.71429, 13.89011, 14.96703, 15.63736, 13.13187, 12.15385,
            13.50549, 14.82418, 13.40659, 11.82418, 13.56044, 13.42857, 12.85714, 15.21978,
            13.54945, 40.82418, 83.95604, 99.93407, 88.98901, 103.2527, 86.93407, 74.87912,
            58.62637, 43.40659, 30.45055, 16.52747, 4.241758, -6.406593, -9.450549, 29.26374,
            31.49451, 28.94505, 36.0, 30.48352, 27.83516, 47.69231, 39.0, 35.17582, 31.25275,
        ]
    }

    #[test]
    fn savitzky_golay_1() {
        let filter: Convolve<f32, 1> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_1().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_2() {
        let filter: Convolve<f32, 2> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_2().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_3() {
        let filter: Convolve<f32, 3> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_3().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_4() {
        let filter: Convolve<f32, 4> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_4().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_5() {
        let filter: Convolve<f32, 5> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_5().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_6() {
        let filter: Convolve<f32, 6> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_6().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_7() {
        let filter: Convolve<f32, 7> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_7().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_8() {
        let filter: Convolve<f32, 8> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_8().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_9() {
        let filter: Convolve<f32, 9> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_9().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_10() {
        let filter: Convolve<f32, 10> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_10().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_11() {
        let filter: Convolve<f32, 11> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_11().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_12() {
        let filter: Convolve<f32, 12> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_12().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn savitzky_golay_13() {
        let filter: Convolve<f32, 13> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_abs_diff_eq!(
            output.as_slice(),
            get_output_13().as_slice(),
            epsilon = 0.001
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn coefficients_sum_to_one() {
        // Savitzky-Golay smoothers must have unity DC gain: Σ h[k] = 1.
        // This is a defining property independent of the coefficient source.
        let c3 = Convolve::<f32, 3>::savitzky_golay();
        let c5 = Convolve::<f32, 5>::savitzky_golay();
        let c7 = Convolve::<f32, 7>::savitzky_golay();
        let c9 = Convolve::<f32, 9>::savitzky_golay();
        let c13 = Convolve::<f32, 13>::savitzky_golay();

        {
            let sum: f32 = c3.config_ref().coefficients.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 5e-5);
        }
        {
            let sum: f32 = c5.config_ref().coefficients.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 5e-5);
        }
        {
            let sum: f32 = c7.config_ref().coefficients.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 5e-5);
        }
        {
            let sum: f32 = c9.config_ref().coefficients.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 5e-5);
        }
        {
            let sum: f32 = c13.config_ref().coefficients.iter().sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 5e-5);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn coefficients_match_scipy_reference_n5() {
        // scipy.signal.savgol_coeffs(5, 2, pos=0, use='dot')
        // Quadratic fit to 5 points, prediction at rightmost position.
        // Reference values to 6 decimal places:
        let expected: [f32; 5] = [0.6, 0.4, 0.2, 0.0, -0.2];
        let filter = Convolve::<f32, 5>::savitzky_golay();
        let actual = filter.config_ref().coefficients;
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-5);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn coefficients_match_scipy_reference_n7() {
        // scipy.signal.savgol_coeffs(7, 2, pos=0, use='dot')
        // Quadratic fit to 7 points, prediction at rightmost position.
        let expected: [f32; 7] = [
            0.46428571,
            0.35714286,
            0.25,
            0.14285714,
            0.03571429,
            -0.07142857,
            -0.17857143,
        ];
        let filter = Convolve::<f32, 7>::savitzky_golay();
        let actual = filter.config_ref().coefficients;
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-5);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn coefficients_match_scipy_reference_n9() {
        // scipy.signal.savgol_coeffs(9, 4, pos=0, use='dot')
        // Quartic fit to 9 points, prediction at rightmost position.
        let expected: [f32; 9] = [
            0.37777778,
            0.31111111,
            0.24444444,
            0.17777778,
            0.11111111,
            0.04444444,
            -0.02222222,
            -0.08888889,
            -0.15555556,
        ];
        let filter = Convolve::<f32, 9>::savitzky_golay();
        let actual = filter.config_ref().coefficients;
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, e, epsilon = 1e-5);
        }
    }
}
