// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Daubechies filters.

use generic_array::typenum::*;
use num_traits::Zero;

use signalo_traits::WithConfig;

use filter::convolve::Config as ConvolveConfig;
use filter::wavelet::{
    analyze::{Analyze, Config as AnalyzeConfig},
    synthesize::{Config as SynthesizeConfig, Synthesize},
};

/// Trait for Daubechies wavelet filters.
pub trait Daubechies: Sized {
    /// Creates a wavelet filter pre-configured with the Daubechies coefficients.
    fn daubechies() -> Self;
}

macro_rules! daubechies_impl_float {
    ($head:ty, $($tail:ty),* $(,)*: $n:ty => [$($low_pass:expr),* $(,)*]) => {
        daubechies_impl_float!($head: $n => [$($low_pass),*]);
        daubechies_impl_float!($($tail),*: $n => [$($low_pass),*]);
    };
    ($t:ty: $n:ty => [$($low_pass:expr),* $(,)*]) => {
        impl Daubechies for Analyze<$t, $n> {
            fn daubechies() -> Self {
                let mut low_pass = arr![$t; $($low_pass),*];
                // Normalize:
                let sum: $t = low_pass.iter().sum();
                if !sum.is_zero() {
                    for coeff in low_pass.iter_mut() {
                        *coeff = *coeff / sum;
                    }
                }
                // The high-pass coefficients are derived by taking the low-pass coefficients:
                let mut high_pass = low_pass.clone();
                // reversing their order:
                high_pass.reverse();
                // and then inverting the sign of every odd-indexed coefficient:
                for (index, coeff) in high_pass.iter_mut().enumerate() {
                    if index % 2 != 0 {
                        *coeff = - *coeff;
                    }
                }
                let config = AnalyzeConfig {
                    low_pass: ConvolveConfig {
                        coefficients: low_pass,
                    },
                    high_pass: ConvolveConfig {
                        coefficients: high_pass,
                    },
                };
                Self::with_config(config)
            }
        }

        impl Daubechies for Synthesize<$t, $n> {
            fn daubechies() -> Self {
                let mut low_pass = arr![$t; $($low_pass),*];
                // Normalize:
                let sum: $t = low_pass.iter().sum();
                if !sum.is_zero() {
                    for coeff in low_pass.iter_mut() {
                        *coeff = *coeff / sum;
                    }
                }
                // The high-pass coefficients are derived by taking the low-pass coefficients:
                let mut high_pass = low_pass.clone();
                // reversing their order:
                high_pass.reverse();
                // and then inverting the sign of every odd-indexed coefficient:
                for (index, coeff) in high_pass.iter_mut().enumerate() {
                    if index % 2 != 0 {
                        *coeff = - *coeff;
                    }
                }

                // FIXME:
                // This additional reverse is the only difference to `analysis`.
                // We should merge the logic somehow.
                low_pass.reverse();
                high_pass.reverse();

                let config = SynthesizeConfig {
                    low_pass: ConvolveConfig {
                        coefficients: low_pass,
                    },
                    high_pass: ConvolveConfig {
                        coefficients: high_pass,
                    },
                };

                Self::with_config(config)
            }
        }
    };
}

// Source: http://wavelets.pybytes.com/wavelet/db1/

daubechies_impl_float!(f32, f64: U2 => [
    0.7071067812,
    0.7071067812,
]);
daubechies_impl_float!(f32, f64: U4 => [
    0.4829629131,
    0.8365163037,
    0.2241438680,
    -0.1294095226,
]);
daubechies_impl_float!(f32, f64: U6 => [
    0.3326705530,
    0.8068915093,
    0.4598775021,
    -0.1350110200,
    -0.0854412739,
    0.0352262919,
]);
daubechies_impl_float!(f32, f64: U8 => [
    0.2303778133,
    0.7148465706,
    0.6308807679,
    -0.0279837694,
    -0.1870348117,
    0.0308413818,
    0.0328830117,
    -0.0105974018,
]);
daubechies_impl_float!(f32, f64: U10 => [
    0.1601023980,
    0.6038292698,
    0.7243085284,
    0.1384281459,
    -0.2422948871,
    -0.0322448696,
    0.0775714938,
    -0.0062414902,
    -0.0125807520,
    0.0033357253,
]);
daubechies_impl_float!(f32, f64: U12 => [
    0.1115407434,
    0.4946238904,
    0.7511339080,
    0.3152503517,
    -0.2262646940,
    -0.1297668676,
    0.0975016056,
    0.0275228655,
    -0.0315820393,
    0.0005538422,
    0.0047772575,
    -0.0010773011,
]);
daubechies_impl_float!(f32, f64: U14 => [
    0.0778520541,
    0.3965393195,
    0.7291320908,
    0.4697822874,
    -0.1439060039,
    -0.2240361850,
    0.0713092193,
    0.0806126092,
    -0.0380299369,
    -0.0165745416,
    0.0125509986,
    0.0004295780,
    -0.0018016407,
    0.0003537138,
]);
daubechies_impl_float!(f32, f64: U16 => [
    0.0544158422,
    0.3128715909,
    0.6756307363,
    0.5853546837,
    -0.0158291053,
    -0.2840155430,
    0.0004724846,
    0.1287474266,
    -0.0173693010,
    -0.0440882539,
    0.0139810279,
    0.0087460940,
    -0.0048703530,
    -0.0003917404,
    0.0006754494,
    -0.0001174768,
]);
daubechies_impl_float!(f32, f64: U18 => [
    0.0380779474,
    0.2438346746,
    0.6048231237,
    0.6572880780,
    0.1331973858,
    -0.2932737833,
    -0.0968407832,
    0.1485407493,
    0.0307256815,
    -0.0676328291,
    0.0002509471,
    0.0223616621,
    -0.0047232048,
    -0.0042815037,
    0.0018476469,
    0.0002303858,
    -0.0002519632,
    0.0000393473,
]);
daubechies_impl_float!(f32, f64: U20 => [
    0.0266700579,
    0.1881768001,
    0.5272011889,
    0.6884590395,
    0.2811723437,
    -0.2498464243,
    -0.1959462744,
    0.1273693403,
    0.0930573646,
    -0.0713941472,
    -0.0294575368,
    0.0332126741,
    0.0036065536,
    -0.0107331755,
    0.0013953517,
    0.0019924053,
    -0.0006858567,
    -0.0001164669,
    0.0000935887,
    -0.0000132642,
]);

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::VecDeque;
    use std::iter::FromIterator;

    use signalo_traits::filter::Filter;

    use filter::wavelet::Decomposition;

    fn get_input() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_analysis_2() -> Vec<Decomposition<f32>> {
        get_low_2()
            .into_iter()
            .zip(get_high_2())
            .map(|(low, high)| Decomposition { low, high })
            .collect()
    }

    fn get_low_2() -> Vec<f32> {
        vec![
            0.0, 0.5, 4.0, 4.5, 3.5, 6.5, 12.0, 14.5, 16.0, 12.5, 10.0, 11.5, 9.0, 13.0, 17.0,
            10.5, 8.0, 16.0, 20.0, 13.5, 7.0, 11.0, 15.0, 12.5, 16.5, 16.5, 60.5, 145.5, 144.0,
            63.0, 62.0, 55.5, 15.5, 19.5, 13.0, 17.0, 21.0, 21.0, 27.5, 21.0, 58.5, 58.5, 18.5,
            22.5, 16.0, 16.0, 60.0, 57.5, 17.5, 24.0,
        ]
    }

    fn get_high_2() -> Vec<f32> {
        vec![
            0.0, 0.5, 3.0, -2.5, 1.5, 1.5, 4.0, -1.5, 3.0, -6.5, 4.0, -2.5, 0.0, 4.0, 0.0, -6.5,
            4.0, 4.0, 0.0, -6.5, 0.0, 4.0, 0.0, -2.5, 6.5, -6.5, 50.5, 34.5, -36.0, -45.0, 44.0,
            -50.5, 10.5, -6.5, 0.0, 4.0, 0.0, 0.0, 6.5, -13.0, 50.5, -50.5, 10.5, -6.5, 0.0, 0.0,
            44.0, -46.5, 6.5, 0.0,
        ]
    }

    fn get_synthesis_2() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![
            0.0, 0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0,
            4.0, 12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0,
            18.0, 106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0,
            16.0, 16.0, 16.0, 104.0, 11.0, 24.0,
        ]
    }

    fn get_analysis_4() -> Vec<Decomposition<f32>> {
        get_low_4()
            .into_iter()
            .zip(get_high_4())
            .map(|(low, high)| Decomposition { low, high })
            .collect()
    }

    fn get_low_4() -> Vec<f32> {
        vec![
            0.000, 0.342, 2.982, 4.982, 3.908, 5.366, 10.806, 14.714, 15.982, 13.884, 10.152,
            10.567, 10.067, 11.275, 16.464, 13.292, 7.603, 13.007, 20.196, 16.292, 7.871, 8.542,
            14.464, 14.025, 14.775, 17.232, 46.553, 126.609, 160.032, 88.401, 47.493, 57.377,
            26.990, 10.912, 15.792, 14.542, 20.464, 21.732, 25.440, 24.250, 45.423, 65.363, 31.179,
            13.912, 18.792, 14.810, 46.053, 66.345, 29.722, 14.619,
        ]
    }

    fn get_high_4() -> Vec<f32> {
        vec![
            0.000, -0.092, -0.799, -0.701, 3.025, -2.732, -0.458, -0.701, 2.933, -1.335, 4.567,
            -5.982, 3.982, -2.440, -2.000, 3.922, 2.518, -7.172, 0.732, 3.922, 3.250, -5.172,
            -2.000, 3.190, 0.060, -3.768, -1.553, -36.004, 23.831, 49.800, -10.141, -43.493,
            53.381, -38.553, 10.422, -5.172, -2.000, 2.732, -1.190, -0.871, 1.697, -24.887, 57.821,
            -38.553, 10.422, -4.440, -8.053, -13.490, 52.113, -35.010,
        ]
    }

    fn get_synthesis_4() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![
            0.000, 0.000, -0.000, 0.000, 1.000, 7.000, 2.000, 5.000, 8.000, 16.000, 13.000, 19.000,
            6.000, 14.000, 9.000, 9.000, 17.000, 17.000, 4.000, 12.000, 20.000, 20.000, 7.000,
            7.000, 15.000, 15.000, 10.001, 23.000, 10.000, 111.001, 180.001, 108.000, 18.000,
            106.000, 5.000, 26.001, 13.000, 13.000, 21.000, 21.000, 21.000, 34.000, 8.000, 109.000,
            7.999, 29.000, 16.000, 15.999, 16.000, 104.000,
        ]
    }

    fn split_analysis(output: &[Decomposition<f32>]) -> (Vec<f32>, Vec<f32>) {
        let low: Vec<_> = output
            .into_iter()
            .map(|Decomposition { low, .. }| *low)
            .collect();
        let high: Vec<_> = output
            .into_iter()
            .map(|Decomposition { high, .. }| *high)
            .collect();
        (low, high)
    }

    fn with_padding<T: Clone>(vec: Vec<T>, prefix: usize, suffix: usize) -> Vec<T> {
        debug_assert!(vec.len() > 0);

        let len = vec.len();
        let prefix_item = vec[0].clone();
        let suffix_item = vec[len - 1].clone();

        let mut deque = VecDeque::from_iter(vec);

        for _ in 0..prefix {
            deque.push_front(prefix_item.clone());
        }

        for _ in 0..suffix {
            deque.push_back(suffix_item.clone());
        }

        Vec::from_iter(deque)
    }

    fn without_padding<T: Clone>(vec: Vec<T>, prefix: usize, suffix: usize) -> Vec<T> {
        debug_assert!(vec.len() >= prefix + suffix);

        let mut deque = VecDeque::from_iter(vec);

        for _ in 0..prefix {
            deque.pop_front();
        }

        for _ in 0..suffix {
            deque.pop_back();
        }

        Vec::from_iter(deque)
    }

    #[test]
    fn daubechies_analysis_2() {
        const PADDING: usize = 2;

        let input = with_padding(get_input(), PADDING, PADDING);

        let analyze: Analyze<f32, U2> = Analyze::daubechies();
        let padded_analysis: Vec<_> = input
            .into_iter()
            .scan(analyze, |filter, input| Some(filter.filter(input)))
            .collect();
        let analysis = without_padding(padded_analysis, PADDING, PADDING);
        let (low, high) = split_analysis(&analysis);

        assert_nearly_eq!(low, get_low_2(), 0.001);
        assert_nearly_eq!(high, get_high_2(), 0.001);
    }

    #[test]
    fn daubechies_synthesis_2() {
        const PADDING: usize = 2;

        let input = with_padding(get_analysis_2(), PADDING, PADDING);

        let synthesize: Synthesize<f32, U2> = Synthesize::daubechies();
        let padded_synthesis: Vec<_> = input
            .into_iter()
            .scan(synthesize, |filter, input| Some(filter.filter(input)))
            .collect();
        let synthesis = without_padding(padded_synthesis, PADDING, PADDING);

        assert_nearly_eq!(synthesis, get_synthesis_2(), 0.001);
    }

    #[test]
    fn daubechies_analysis_4() {
        const PADDING: usize = 4;

        let input = with_padding(get_input(), PADDING, PADDING);

        let analyze: Analyze<f32, U4> = Analyze::daubechies();

        let padded_analysis: Vec<_> = input
            .into_iter()
            .scan(analyze, |filter, input| Some(filter.filter(input)))
            .collect();
        let analysis = without_padding(padded_analysis, PADDING, PADDING);
        let (low, high) = split_analysis(&analysis);

        assert_nearly_eq!(low, get_low_4(), 0.001);
        assert_nearly_eq!(high, get_high_4(), 0.001);
    }

    #[test]
    fn daubechies_synthesis_4() {
        const PADDING: usize = 4;

        let input = with_padding(get_analysis_4(), PADDING, PADDING);

        let synthesize: Synthesize<f32, U4> = Synthesize::daubechies();

        let padded_synthesis: Vec<_> = input
            .into_iter()
            .scan(synthesize, |filter, input| Some(filter.filter(input)))
            .collect();
        let synthesis = without_padding(padded_synthesis, PADDING, PADDING);

        assert_nearly_eq!(synthesis, get_synthesis_4(), 0.001);
    }
}
