// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Savitzky-Golay filters.

#![allow(clippy::wildcard_imports)]

use signalo_traits::WithConfig;

use super::{Config, Convolve};

/// Trait for Savitzky-Golay convolution filters.
pub trait SavitzkyGolay: Sized {
    /// Creates a convolution filter pre-configured with the Savitzky-Golay coefficients.
    fn savitzky_golay() -> Self;
}

macro_rules! savitzky_golay_impl_float {
    ($width:expr => [$($coeffs:expr),*]) => {
        impl SavitzkyGolay for Convolve<f32, $width> {
            fn savitzky_golay() -> Self {
                Self::with_config(Config {
                    coefficients: [$($coeffs),*]
                })
            }
        }
        impl SavitzkyGolay for Convolve<f64, $width> {
            fn savitzky_golay() -> Self {
                Self::with_config(Config {
                    coefficients: [$($coeffs),*]
                })
            }
        }
    };
}

// Source: https://gregstanleyandassociates.com/whitepapers/FaultDiagnosis/Filtering/LeastSquares-Filter/leastsquares-filter.htm

savitzky_golay_impl_float!(1 => [
    1.0
]);
savitzky_golay_impl_float!(2 => [
    1.0, 0.0
]);
savitzky_golay_impl_float!(3 => [
    0.83333, 0.33333, -0.16667
]);
savitzky_golay_impl_float!(4 => [
    0.7, 0.4, 0.1, -0.2
]);
savitzky_golay_impl_float!(5 => [
    0.6, 0.4, 0.2, 0.0, -0.2
]);
savitzky_golay_impl_float!(6 => [
    0.52381, 0.38095, 0.2381, 0.09524, -0.04762, -0.19048
]);
savitzky_golay_impl_float!(7 => [
    0.46429, 0.35714, 0.25, 0.14286, 0.03571, -0.07143, -0.17857
]);
savitzky_golay_impl_float!(8 => [
    0.41667, 0.33333, 0.25, 0.16667, 0.08333, 0.0, -0.08333, -0.16667
]);
savitzky_golay_impl_float!(9 => [
    0.37778, 0.31111, 0.24444, 0.17778, 0.11111, 0.04444, -0.02222, -0.08889, -0.15556
]);
savitzky_golay_impl_float!(10 => [
    0.34545, 0.29091, 0.23636, 0.18182, 0.12727, 0.07273, 0.01818, -0.03636, -0.09091, -0.14545
]);
savitzky_golay_impl_float!(11 => [
    0.31818, 0.27273, 0.22727, 0.18182, 0.13636, 0.09091, 0.04545, 0.0, -0.04545, -0.09091,
    -0.13636
]);
savitzky_golay_impl_float!(12 => [
    0.29487, 0.25641, 0.21795, 0.17949, 0.14103, 0.10256, 0.06410, 0.02564, -0.01282, -0.05128,
    -0.08974, -0.12821
]);
savitzky_golay_impl_float!(13 => [
    0.27473, 0.24176, 0.20879, 0.17582, 0.14286, 0.10989, 0.07692, 0.04396, 0.01099, -0.02198,
    -0.05495, -0.08791, -0.12088
]);

#[cfg(test)]
mod tests {
    use super::*;

    use nearly_eq::assert_nearly_eq;
    use signalo_traits::Filter;

    fn get_input() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
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
            0.0, 0.83333, 6.16664, 3.8333, 3.6666198, 7.99995, 15.16657, 14.83321, 17.49984,
            9.16654, 10.499869, 11.166571, 8.16656, 15.66655, 18.33319, 6.16654, 8.49989, 19.99988,
            21.33316, 9.16651, 4.83322, 13.66657, 16.33321, 10.8331995, 19.99984, 14.33319,
            91.99952, 185.33232, 131.49866, 20.998981, 76.33256, 36.49957, 5.666212, 18.66652,
            10.83316, 19.66651, 22.333149, 20.99979, 31.83308, 14.49979, 87.83283, 41.666252,
            8.666179, 21.666489, 13.83313, 15.99984, 89.33288, 41.16623, 6.3328705, 26.16647,
        ]
    }

    fn get_output_4() -> Vec<f32> {
        vec![
            0.0, 0.7, 5.3, 4.2999997, 4.8, 6.4, 14.5, 15.299999, 18.5, 9.9, 11.5, 8.7, 10.1, 13.6,
            17.8, 9.5, 8.299999, 15.8, 22.4, 12.5, 5.7, 10.0, 15.8, 13.1, 18.6, 14.2, 82.0, 166.8,
            156.7, 51.6, 56.199997, 26.1, 27.199999, -1.2000008, 15.9, 16.0, 21.8, 22.6, 30.099998,
            17.1, 78.7, 43.2, 32.8, 1.7999983, 18.9, 13.4, 77.6, 47.700005, 28.4, 6.699999,
        ]
    }

    fn get_output_5() -> Vec<f32> {
        vec![
            0.0, 0.6, 4.6000004, 4.2, 5.2, 7.0, 12.400001, 15.400001, 18.800001, 12.200001,
            11.400001, 9.6, 8.0, 14.400002, 16.0, 10.800001, 10.400001, 14.2, 19.0, 15.4, 8.6, 9.2,
            12.4, 13.6, 19.4, 14.2, 72.200005, 152.40001, 154.4, 88.0, 70.200005, 13.0, 17.2, 15.6,
            -2.999999, 19.400002, 18.400002, 22.600002, 30.400002, 18.400002, 71.200005, 45.8,
            35.600002, 21.2, 0.0, 17.6, 66.200005, 48.200005, 36.4, 23.0,
        ]
    }

    fn get_output_6() -> Vec<f32> {
        vec![
            0.0,
            0.52381,
            4.0476203,
            3.9523702,
            5.14289,
            7.1904902,
            12.285721,
            13.85713,
            18.8572,
            13.666691,
            13.095301,
            9.619011,
            8.66666,
            11.904741,
            16.57142,
            10.380949,
            11.333401,
            15.095221,
            17.28566,
            13.7619095,
            11.6667595,
            10.857141,
            11.14278,
            11.04759,
            19.19053,
            15.76189,
            64.809616,
            137.80937,
            149.52393,
            99.143585,
            98.04835,
            27.856586,
            3.0473738,
            6.5711718,
            9.95232,
            1.0949221,
            21.14286,
            19.666641,
            29.714333,
            20.66666,
            65.238235,
            45.85689,
            39.333748,
            24.857191,
            15.42856,
            -0.095560074,
            63.00006,
            44.42831,
            39.238533,
            30.428703,
        ]
    }

    fn get_output_7() -> Vec<f32> {
        vec![
            0.0, 0.46429, 3.60717, 3.67856, 4.92859, 7.0357504, 12.00002, 13.85715, 17.39291,
            14.678551, 14.357149, 11.2857, 8.64281, 11.89291, 14.178579, 11.71423, 10.8572,
            15.35717, 17.785671, 12.99999, 10.714319, 13.21429, 12.21423, 9.99999, 16.32153,
            16.28567, 60.5004, 125.60762, 141.0356, 104.71431, 107.50046, 56.82054, 14.10673,
            -7.4996905, 1.1068487, 10.749979, 4.000098, 22.07141, 26.42868, 21.60705, 61.893257,
            44.999733, 40.67855, 29.464529, 18.78533, 12.35707, 41.28618, 45.60684, 37.60721,
            33.89312,
        ]
    }

    fn get_output_8() -> Vec<f32> {
        vec![
            0.0, 0.41667, 3.2500198, 3.41665, 4.66668, 6.75003, 11.50001, 13.66667, 17.25005,
            14.083301, 15.33336, 12.666681, 10.166651, 11.41671, 13.91667, 10.166581, 12.00008,
            14.50001, 17.83329, 14.0000105, 10.33336, 12.166651, 14.166651, 11.00001, 14.750071,
            14.16658, 56.50032, 116.750305, 132.58304, 104.50002, 112.83389, 70.24938, 40.08344,
            1.0004768, -13.083906, 1.5831327, 11.50025, 6.16633, 28.00014, 19.749872, 58.58367,
            45.166397, 40.91665, 31.91696, 23.333038, 15.166649, 47.33392, 28.416021, 39.91678,
            33.25025,
        ]
    }

    fn get_output_9() -> Vec<f32> {
        vec![
            0.0, 0.37778, 2.95557, 3.17777, 4.39998, 6.4222403, 10.933331, 13.2444, 16.95553,
            14.48882, 14.86656, 13.82216, 11.533219, 12.46657, 13.26658, 10.33313, 10.555429,
            15.19997, 16.88872, 14.510949, 11.4888, 11.622099, 13.133169, 12.88876, 15.13327,
            12.97762, 50.888878, 108.8002, 125.95504, 102.376976, 113.08883, 79.57709, 53.55439,
            24.44439, -6.8896317, -12.956882, 2.3107915, 12.17742, 12.75507, 21.999828, 53.4888,
            44.555347, 41.86609, 33.19998, 26.11082, 19.35496, 46.17797, 35.22181, 24.93243,
            36.02231,
        ]
    }

    fn get_output_10() -> Vec<f32> {
        vec![
            0.0, 0.34545, 2.7090602, 2.9636302, 4.14541, 6.0908804, 10.363541, 12.727221, 16.45441,
            14.6545315, 15.290781, 13.63639, 12.74537, 13.5091, 14.05448, 10.054629, 10.63632,
            13.600031, 17.38174, 14.018209, 12.199981, 12.527281, 12.527261, 12.07277, 16.490831,
            13.490951, 46.65407, 99.853935, 119.45385, 100.56329, 111.47197, 83.69105, 63.708683,
            37.63692, 14.0000305, -8.926382, -12.254108, 3.1280322, 17.27251, 8.47321, 52.70854,
            41.69112, 41.96318, 34.98209, 27.836031, 22.127672, 47.09026, 35.40058, 31.34492,
            22.636992,
        ]
    }

    fn get_output_11() -> Vec<f32> {
        vec![
            0.0, 0.31818, 2.49999, 2.77274, 3.90907, 5.77273, 9.81814, 12.18182, 15.86356,
            14.54546, 15.49992, 14.22725, 12.72717, 14.4999695, 14.90905, 11.04549, 10.3182,
            13.40907, 15.772759, 14.77271, 11.9545, 13.09094, 13.27272, 11.59087, 15.45452,
            14.90915, 44.45437, 92.54544, 111.54541, 97.909096, 109.99947, 85.45445, 69.13549,
            48.18132, 26.408764, 9.409851, -9.86347, -11.272154, 8.091591, 12.7272, 37.68197,
            42.36386, 39.81781, 35.8639, 29.99953, 24.000208, 47.317642, 37.27299, 31.818354,
            28.59094,
        ]
    }

    fn get_output_12() -> Vec<f32> {
        vec![
            0.0, 0.29487, 2.3205, 2.60256, 3.6923099, 5.47437, 9.307699, 11.641, 15.243589,
            14.282101, 15.44878, 14.60259, 13.41028, 14.39734, 15.76924, 12.089769, 11.192369,
            12.88464, 15.4487, 13.52554, 12.83342, 12.79491, 13.743589, 12.346109, 14.743528,
            14.10256, 43.512726, 87.4869, 104.7948, 93.205475, 107.64188, 86.821144, 72.268814,
            54.486526, 36.78211, 20.67921, 6.3210998, -10.128019, -6.3337803, 4.2562094, 39.243767,
            29.40982, 40.82071, 34.5257, 31.30809, 26.30736, 47.14074, 38.3845, 33.8463, 29.192587,
        ]
    }

    fn get_output_13() -> Vec<f32> {
        vec![
            0.0, 0.27473, 2.16487, 2.4505699, 3.4945197, 5.1978197, 8.83526, 11.12094, 14.6264305,
            13.93409, 15.241799, 14.714279, 13.890131, 14.9671, 15.63743, 13.13189, 12.153749,
            13.505539, 14.824249, 13.40656, 11.82408, 13.560519, 13.42861, 12.85707, 15.2198105,
            13.549561, 40.824608, 83.95688, 99.934715, 88.98865, 103.252625, 86.93428, 74.87899,
            58.62589, 43.40729, 30.45087, 16.526562, 4.241388, -6.406721, -9.450667, 29.26364,
            31.494772, 28.944927, 35.99968, 30.483759, 27.83514, 47.69244, 39.00046, 35.1757,
            31.252369,
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

        assert_nearly_eq!(output, get_output_1(), 0.001);
    }

    #[test]
    fn savitzky_golay_2() {
        let filter: Convolve<f32, 2> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_2(), 0.001);
    }

    #[test]
    fn savitzky_golay_3() {
        let filter: Convolve<f32, 3> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_3(), 0.001);
    }

    #[test]
    fn savitzky_golay_4() {
        let filter: Convolve<f32, 4> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_4(), 0.001);
    }

    #[test]
    fn savitzky_golay_5() {
        let filter: Convolve<f32, 5> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_5(), 0.001);
    }

    #[test]
    fn savitzky_golay_6() {
        let filter: Convolve<f32, 6> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_6(), 0.001);
    }

    #[test]
    fn savitzky_golay_7() {
        let filter: Convolve<f32, 7> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_7(), 0.001);
    }

    #[test]
    fn savitzky_golay_8() {
        let filter: Convolve<f32, 8> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_8(), 0.001);
    }

    #[test]
    fn savitzky_golay_9() {
        let filter: Convolve<f32, 9> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_9(), 0.001);
    }

    #[test]
    fn savitzky_golay_10() {
        let filter: Convolve<f32, 10> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_10(), 0.001);
    }

    #[test]
    fn savitzky_golay_11() {
        let filter: Convolve<f32, 11> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_11(), 0.001);
    }

    #[test]
    fn savitzky_golay_12() {
        let filter: Convolve<f32, 12> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_12(), 0.001);
    }

    #[test]
    fn savitzky_golay_13() {
        let filter: Convolve<f32, 13> = Convolve::savitzky_golay();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        assert_nearly_eq!(output, get_output_13(), 0.001);
    }
}
