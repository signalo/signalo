// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Daubechies filters.

#![allow(
    clippy::wildcard_imports,
    clippy::excessive_precision,
    clippy::approx_constant
)]

use num_traits::Zero;

use signalo_traits::WithConfig;

use convolve::Config as ConvolveConfig;
use wavelet::{
    analyze::{Analyze, Config as AnalyzeConfig},
    synthesize::{Config as SynthesizeConfig, Synthesize},
};

/// Trait for Daubechies wavelet filters.
pub trait Daubechies: Sized {
    /// Creates a wavelet filter pre-configured with the Daubechies coefficients.
    fn daubechies() -> Self;
}

macro_rules! daubechies_impl_float {
    ($head:ty, $($tail:ty),* $(,)*: $n:expr => [$($low_pass:expr),* $(,)*]) => {
        daubechies_impl_float!($head: $n => [$($low_pass),*]);
        daubechies_impl_float!($($tail),*: $n => [$($low_pass),*]);
    };
    ($t:ty: $n:expr => [$($low_pass:expr),* $(,)*]) => {
        impl Daubechies for Analyze<$t, $n> {
            fn daubechies() -> Self {
                let mut low_pass = [$($low_pass),*];
                // Normalize:
                let sum: $t = low_pass.iter().sum();
                if !sum.is_zero() {
                    for coeff in low_pass.iter_mut() {
                        *coeff /= sum;
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
                let mut low_pass = [$($low_pass),*];
                // Normalize:
                let sum: $t = low_pass.iter().sum();
                if !sum.is_zero() {
                    for coeff in low_pass.iter_mut() {
                        *coeff /= sum;
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

daubechies_impl_float!(f32, f64: 2 => [
    0.707_106_781_2,
    0.707_106_781_2,
]);

daubechies_impl_float!(f32, f64: 4 => [
    0.482_962_913_1,
    0.836_516_303_7,
    0.224_143_868_0,
    -0.129_409_522_6,
]);
daubechies_impl_float!(f32, f64: 6 => [
    0.332_670_553_0,
    0.806_891_509_3,
    0.459_877_502_1,
    -0.135_011_020_0,
    -0.085_441_273_9,
    0.035_226_291_9,
]);
daubechies_impl_float!(f32, f64: 8 => [
    0.230_377_813_3,
    0.714_846_570_6,
    0.630_880_767_9,
    -0.027_983_769_4,
    -0.187_034_811_7,
    0.030_841_381_8,
    0.032_883_011_7,
    -0.010_597_401_8,
]);
daubechies_impl_float!(f32, f64: 10 => [
    0.160_102_398_0,
    0.603_829_269_8,
    0.724_308_528_4,
    0.138_428_145_9,
    -0.242_294_887_1,
    -0.032_244_869_6,
    0.077_571_493_8,
    -0.006_241_490_2,
    -0.012_580_752_0,
    0.003_335_725_3,
]);
daubechies_impl_float!(f32, f64: 12 => [
    0.111_540_743_4,
    0.494_623_890_4,
    0.751_133_908_0,
    0.315_250_351_7,
    -0.226_264_694_0,
    -0.129_766_867_6,
    0.097_501_605_6,
    0.027_522_865_5,
    -0.031_582_039_3,
    0.000_553_842_2,
    0.004_777_257_5,
    -0.001_077_301_1,
]);
daubechies_impl_float!(f32, f64: 14 => [
    0.077_852_054_1,
    0.396_539_319_5,
    0.729_132_090_8,
    0.469_782_287_4,
    -0.143_906_003_9,
    -0.224_036_185_0,
    0.071_309_219_3,
    0.080_612_609_2,
    -0.038_029_936_9,
    -0.016_574_541_6,
    0.012_550_998_6,
    0.000_429_578_0,
    -0.001_801_640_7,
    0.000_353_713_8,
]);
daubechies_impl_float!(f32, f64: 16 => [
    0.054_415_842_2,
    0.312_871_590_9,
    0.675_630_736_3,
    0.585_354_683_7,
    -0.015_829_105_3,
    -0.284_015_543_0,
    0.000_472_484_6,
    0.128_747_426_6,
    -0.017_369_301_0,
    -0.044_088_253_9,
    0.013_981_027_9,
    0.008_746_094_0,
    -0.004_870_353_0,
    -0.000_391_740_4,
    0.000_675_449_4,
    -0.000_117_476_8,
]);
daubechies_impl_float!(f32, f64: 18 => [
    0.038_077_947_4,
    0.243_834_674_6,
    0.604_823_123_7,
    0.657_288_078_0,
    0.133_197_385_8,
    -0.293_273_783_3,
    -0.096_840_783_2,
    0.148_540_749_3,
    0.030_725_681_5,
    -0.067_632_829_1,
    0.000_250_947_1,
    0.022_361_662_1,
    -0.004_723_204_8,
    -0.004_281_503_7,
    0.001_847_646_9,
    0.000_230_385_8,
    -0.000_251_963_2,
    0.000_039_347_3,
]);
daubechies_impl_float!(f32, f64: 20 => [
    0.026_670_057_9,
    0.188_176_800_1,
    0.527_201_188_9,
    0.688_459_039_5,
    0.281_172_343_7,
    -0.249_846_424_3,
    -0.195_946_274_4,
    0.127_369_340_3,
    0.093_057_364_6,
    -0.071_394_147_2,
    -0.029_457_536_8,
    0.033_212_674_1,
    0.003_606_553_6,
    -0.010_733_175_5,
    0.001_395_351_7,
    0.001_992_405_3,
    -0.000_685_856_7,
    -0.000_116_466_9,
    0.000_093_588_7,
    -0.000_013_264_2,
]);

#[cfg(test)]
mod tests {
    use super::*;

    use signalo_traits::Filter;

    use wavelet::Decomposition;

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

        let prefix_iter = ::std::iter::repeat(prefix_item).take(prefix);
        let suffix_iter = ::std::iter::repeat(suffix_item).take(suffix);

        prefix_iter.chain(vec).chain(suffix_iter).collect()
    }

    fn without_padding<T: Clone>(vec: Vec<T>, prefix: usize, suffix: usize) -> Vec<T> {
        debug_assert!(vec.len() >= prefix + suffix);

        let take_len = vec.len() - prefix - suffix;

        vec.into_iter().skip(prefix).take(take_len).collect()
    }

    #[test]
    fn daubechies_analysis_2() {
        const PADDING: usize = 2;

        let input = with_padding(get_input(), PADDING, PADDING);

        let analyze: Analyze<f32, 2> = Analyze::daubechies();
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

        let synthesize: Synthesize<f32, 2> = Synthesize::daubechies();
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

        let analyze: Analyze<f32, 4> = Analyze::daubechies();

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

        let synthesize: Synthesize<f32, 4> = Synthesize::daubechies();

        let padded_synthesis: Vec<_> = input
            .into_iter()
            .scan(synthesize, |filter, input| Some(filter.filter(input)))
            .collect();
        let synthesis = without_padding(padded_synthesis, PADDING, PADDING);

        assert_nearly_eq!(synthesis, get_synthesis_4(), 0.001);
    }
}
