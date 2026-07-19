// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Percentile approximation sinks based on histogram binning.
//!
//! Approximates percentiles (quartiles, medians, etc.) by computing cumulative bin counts
//! from the underlying histogram and returning interpolated quantile values.

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Finalize, Reset, Sink, State as StateTrait,
    StateMut, WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

use num_traits::{float::FloatCore, NumCast};

use super::histogram::{Config as HistogramConfig, HistogramArray};

/// Configuration for a percentile sink.
///
/// Specifies the histogram range `[min, max]` and the target percentile `(0.0..1.0)`.
/// The percentile value represents the target quantile: `0.5` = median, `0.25` = quartile, etc.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Minimum value of the histogram range.
    pub min: T,
    /// Maximum value of the histogram range.
    pub max: T,
    /// Target percentile as a fraction in `[0.0, 1.0]`.
    pub percentile: T,
}

impl<T: Clone> Config<T> {
    /// Creates a new percentile configuration.
    pub fn new(min: T, max: T, percentile: T) -> Self {
        Self {
            min,
            max,
            percentile,
        }
    }
}

impl<T: Clone + Default> Default for Config<T> {
    fn default() -> Self {
        Self {
            min: T::default(),
            max: T::default(),
            percentile: T::default(),
        }
    }
}

/// A percentile sink that approximates percentiles using histogram binning.
///
/// Internally wraps a [`Histogram`](super::histogram::Histogram) sink and divides the `[min, max]`
/// range into equal-width bins. The percentile is approximated by finding the bin where the
/// cumulative count exceeds the target percentage of total samples, then linearly
/// interpolating within that bin.
///
/// # Complexity
///
/// - **Time per sample (sink):** O(1); delegates to the underlying histogram bin lookup.
/// - **Time (finalize):** O(B) where B is the number of bins; scans bins to find the target quantile.
/// - **Space:** O(B) for the bin counters.
#[derive(Clone, Debug)]
pub struct Percentile<T: Clone, B> {
    config: Config<T>,
    histogram: super::histogram::Histogram<T, B>,
}

/// A [`Percentile`] backed by a fixed-size array of `N` bins.
pub type PercentileArray<T, const N: usize> = Percentile<T, [u32; N]>;

/// A [`Percentile`] backed by a heap-allocated, runtime-sized `Vec` of bins.
#[cfg(feature = "alloc")]
pub type PercentileVec<T> = Percentile<T, alloc::vec::Vec<u32>>;

/// A [`Percentile`] that borrows a `[u32]` slice for its bin storage.
///
/// This alias allows sharing a caller-owned bin-counter slice without taking
/// ownership of it. Construct via [`Percentile::from_parts`], passing a
/// `&mut [u32]` slice and a corresponding [`Histogram`](super::histogram::Histogram).
pub type PercentileRefMut<'a, T> = Percentile<T, &'a mut [u32]>;

impl<T: Clone, B: crate::storage::AsSlice<u32>> Percentile<T, B> {
    /// Creates a new percentile sink from a config and caller-supplied histogram.
    ///
    /// The histogram's bin storage is taken as-is. For a clean start, pass a
    /// zero-initialized histogram.
    pub fn from_parts(config: Config<T>, histogram: super::histogram::Histogram<T, B>) -> Self {
        Self { config, histogram }
    }
}

impl<T: Clone + Default, const N: usize> Default for PercentileArray<T, N> {
    fn default() -> Self {
        Self {
            config: Config::default(),
            histogram: HistogramArray::default(),
        }
    }
}

impl<T: Clone, B> ConfigTrait for Percentile<T, B> {
    type Config = Config<T>;
}

impl<T: Clone, B> StateTrait for Percentile<T, B> {
    type State = super::histogram::State<B>;
}

impl<T: Clone + Default, const N: usize> WithConfig for PercentileArray<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let histogram_config = HistogramConfig::new(config.min.clone(), config.max.clone());

        Self {
            config,
            histogram: HistogramArray::with_config(histogram_config),
        }
    }
}

impl<T: Clone, B> ConfigRef for Percentile<T, B> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: Clone, B> ConfigClone for Percentile<T, B> {
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T: Clone, B> StateMut for Percentile<T, B> {
    fn state_mut(&mut self) -> &mut Self::State {
        self.histogram.state_mut()
    }
}

impl<T: Clone, B> HasGuts for Percentile<T, B> {
    type Guts = (Config<T>, super::histogram::Histogram<T, B>);
}

impl<T: Clone, B> FromGuts for Percentile<T, B> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, histogram) = guts;
        Self { config, histogram }
    }
}

impl<T: Clone, B> IntoGuts for Percentile<T, B> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.histogram)
    }
}

impl<T: Clone + Default, const N: usize> Reset for PercentileArray<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T: Clone + Default, const N: usize> ResetMut for PercentileArray<T, N> where Self: Reset {}

macro_rules! impl_sink_percentile {
    ($ty:ty) => {
        impl<B: crate::storage::AsSlice<u32>> Sink<$ty> for Percentile<$ty, B> {
            #[inline]
            fn sink(&mut self, input: $ty) {
                self.histogram.sink(input);
            }
        }
    };
}

impl_sink_percentile!(f32);
impl_sink_percentile!(f64);

impl<T: Clone + Default + FloatCore, B: crate::storage::AsSlice<u32>> Finalize
    for Percentile<T, B>
{
    type Output = Option<T>;

    #[inline]
    fn finalize(self) -> Self::Output {
        let bins = self.histogram.finalize();
        let bins_slice = bins.as_slice();

        let total: u32 = bins_slice.iter().sum();
        if total == 0 {
            return None;
        }

        let target_count = NumCast::from(total).unwrap_or(T::zero()) * self.config.percentile;

        let mut cumulative = 0u32;
        let mut bin_index = 0;
        let mut prev_cumulative = 0u32;
        for (i, &count) in bins_slice.iter().enumerate() {
            prev_cumulative = cumulative;
            cumulative += count;
            if NumCast::from(cumulative).unwrap_or(T::zero()) >= target_count {
                bin_index = i;
                break;
            }
        }

        let range = self.config.max - self.config.min;
        if range == T::zero() {
            return Some(self.config.min);
        }

        let n_t: T = NumCast::from(bins_slice.len()).unwrap_or(T::zero());
        let bin_width = range / n_t;
        let i_t: T = NumCast::from(bin_index).unwrap_or(T::zero());
        let bin_min = self.config.min + i_t * bin_width;

        let remainder = target_count - NumCast::from(prev_cumulative).unwrap_or(T::zero());
        let count_t: T = NumCast::from(bins_slice[bin_index]).unwrap_or(T::one());
        let fraction_in_bin = remainder / count_t;
        let interpolated = bin_min + fraction_in_bin * bin_width;

        Some(interpolated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_uniform_distribution_median() {
        use alloc::vec;
        // Test: uniform distribution over [0, 4], N=4, percentile=0.5 → median ≈ 2.0
        const N: usize = 4;
        let config = Config::new(0.0f32, 4.0f32, 0.5f32);
        let mut percentile: PercentileArray<f32, N> = PercentileArray::with_config(config);

        // Feed uniform distribution: one value in each bin
        let inputs = vec![0.5f32, 1.5f32, 2.5f32, 3.5f32];
        for input in inputs {
            percentile.sink(input);
        }

        let result = percentile.finalize();
        assert!(result.is_some(), "Expected Some for non-empty input");
        if let Some(median) = result {
            // With 4 samples distributed as [1,1,1,1], median at 50th percentile
            // Cumulative: [1, 2, 3, 4]. Target is 4*0.5=2.0
            // At i=1: cumulative=2, which meets target → bin_index=1
            // Bin 1 range: [1.0, 2.0), fraction in bin = (2.0-1)/1 = 1.0
            // Interpolated = 1.0 + 1.0 * 1.0 = 2.0
            assert!(
                (median - 2.0).abs() < 0.1,
                "Median should be ≈ 2.0, got {median}"
            );
        }
    }

    #[test]
    fn test_empty_input() {
        // Test: empty input → None
        const N: usize = 4;
        let config = Config::new(0.0f32, 4.0f32, 0.5f32);
        let percentile: PercentileArray<f32, N> = PercentileArray::with_config(config);

        let result = percentile.finalize();
        assert_eq!(result, None, "Empty input should return None");
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_percentile_f64() {
        use alloc::vec;

        // Test f64 support
        const N: usize = 4;
        let config = Config::new(0.0f64, 4.0f64, 0.5f64);
        let mut percentile: PercentileArray<f64, N> = PercentileArray::with_config(config);

        let inputs = vec![0.5f64, 1.5f64, 2.5f64, 3.5f64];
        for input in inputs {
            percentile.sink(input);
        }

        let result = percentile.finalize();
        assert!(result.is_some());
        if let Some(median) = result {
            assert!((median - 2.0).abs() < 0.1, "f64 median should be ≈ 2.0");
        }
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn test_quartile_25() {
        use alloc::vec;

        // Test: 25th percentile (first quartile)
        const N: usize = 4;
        let config = Config::new(0.0f32, 4.0f32, 0.25f32);
        let mut percentile: PercentileArray<f32, N> = PercentileArray::with_config(config);

        let inputs = vec![0.5f32, 1.5f32, 2.5f32, 3.5f32];
        for input in inputs {
            percentile.sink(input);
        }

        let result = percentile.finalize();
        assert!(result.is_some());
        if let Some(q25) = result {
            // Cumulative: [1, 2, 3, 4]. Target is 4*0.25=1.0
            // At i=0: cumulative=1, meets target → bin_index=0
            // Bin 0 range: [0.0, 1.0), fraction = (1.0-0)/1 = 1.0
            // Interpolated = 0.0 + 1.0 * 1.0 = 1.0
            assert!((q25 - 1.0).abs() < 0.1, "Q25 should be ≈ 1.0, got {q25}");
        }
    }
}
