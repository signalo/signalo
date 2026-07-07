// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Daubechies wavelet implementation.
//!
//! Provides Daubechies wavelets with configurable order, commonly used for signal analysis
//! and compression due to their compact support and smoothness properties.

#![allow(
    clippy::wildcard_imports,
    clippy::excessive_precision,
    clippy::approx_constant
)]

use core::marker::PhantomData;

use num_traits::Zero;

use crate::traits::WithConfig;

use super::{
    analyze::{AnalyzeArray, Config as AnalyzeConfig},
    synthesize::{Config as SynthesizeConfig, SynthesizeArray},
};
use crate::filters::fir::convolve::Config as ConvolveConfig;

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
        impl Daubechies for AnalyzeArray<$t, $n> {
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
                    _phantom: PhantomData,
                };
                Self::with_config(config)
            }
        }

        impl Daubechies for SynthesizeArray<$t, $n> {
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
                    _phantom: PhantomData,
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
mod tests;
