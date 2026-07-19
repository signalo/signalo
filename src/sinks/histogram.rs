// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Histogram sinks for signal distribution analysis.
//!
//! Divides a signal's range into equally-sized bins and counts how many samples fall into each bin,
//! providing a distribution histogram. Out-of-bounds values are clamped to edge bins.

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Finalize, Reset, Sink, State as StateTrait,
    StateMut, WithConfig,
};
use num_traits::{float::FloatCore, NumCast};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Configuration for a histogram sink.
///
/// Specifies the bin range [min, max] for the histogram. Input values are mapped to bins
/// and out-of-bounds values are clamped to the edge bins (no panic).
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Minimum value of the histogram range
    pub min: T,
    /// Maximum value of the histogram range
    pub max: T,
}

impl<T: Clone> Config<T> {
    /// Creates a new histogram configuration.
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

impl<T: Clone + Default> Default for Config<T> {
    fn default() -> Self {
        Self {
            min: T::default(),
            max: T::default(),
        }
    }
}

/// State for a histogram sink.
///
/// Maintains a storage of bin counters. Each bin is a u32 counter that increments
/// when input values fall within that bin's range.
#[derive(Clone, Debug)]
pub struct State<B> {
    /// Storage of bin counters.
    pub bins: B,
}

impl<const N: usize> Default for State<[u32; N]> {
    fn default() -> Self {
        Self { bins: [0; N] }
    }
}

/// A histogram sink that tracks the distribution of input values.
///
/// Divides the range [min, max] into equal-width bins (with the bin count taken from
/// the length of the `bins` storage) and counts how many input values fall into each
/// bin. Out-of-bounds values are clamped to the edge bins.
///
/// # Complexity
///
/// - **Time per sample:** O(1); one float multiply and a bin index clamp.
/// - **Space:** O(B) where B is the number of bins.
#[derive(Clone, Debug)]
pub struct Histogram<T: Clone, B> {
    config: Config<T>,
    state: State<B>,
}

/// A [`Histogram`] backed by a fixed-size array of `N` bins.
pub type HistogramArray<T, const N: usize> = Histogram<T, [u32; N]>;

/// A [`Histogram`] backed by a heap-allocated, runtime-sized `Vec` of bins.
#[cfg(feature = "alloc")]
pub type HistogramVec<T> = Histogram<T, alloc::vec::Vec<u32>>;

/// A [`Histogram`] that borrows a `[u32]` slice for its bin storage.
///
/// This alias allows sharing a caller-owned bin-counter slice without taking
/// ownership of it. Construct via [`Histogram::from_parts`], passing a
/// zero-initialized `&mut [u32]` slice.
pub type HistogramRefMut<'a, T> = Histogram<T, &'a mut [u32]>;

impl<T: Clone, B: AsSlice<u32>> Histogram<T, B> {
    /// Creates a new histogram from a config and caller-supplied bin storage.
    ///
    /// The `bins` array is taken as-is with their current contents. If the bins contain
    /// non-zero counts, those values contribute to the histogram from the first sample.
    ///
    /// # Expected storage state
    ///
    /// For a clean start, pass a zero-initialized bin array.
    ///
    /// # Panics
    ///
    /// Panics if `bins` is empty.
    pub fn from_parts(config: Config<T>, bins: B) -> Self {
        assert!(
            !bins.as_slice().is_empty(),
            "Histogram requires at least one bin"
        );
        Self {
            config,
            state: State { bins },
        }
    }
}

impl<T: Clone + Default, const N: usize> Default for HistogramArray<T, N> {
    fn default() -> Self {
        Self {
            config: Config::default(),
            state: State::default(),
        }
    }
}

impl<T: Clone, B> ConfigTrait for Histogram<T, B> {
    type Config = Config<T>;
}

impl<T: Clone, B> StateTrait for Histogram<T, B> {
    type State = State<B>;
}

impl<T: Clone + Default, const N: usize> WithConfig for HistogramArray<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self::from_parts(config, [0; N])
    }
}

impl<T: Clone, B> ConfigRef for Histogram<T, B> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: Clone, B> ConfigClone for Histogram<T, B> {
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T: Clone, B> StateMut for Histogram<T, B> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T: Clone, B> HasGuts for Histogram<T, B> {
    type Guts = (Config<T>, State<B>);
}

impl<T: Clone, B> FromGuts for Histogram<T, B> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T: Clone, B> IntoGuts for Histogram<T, B> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T: Clone, B: AsSlice<u32>> Reset for Histogram<T, B> {
    fn reset(mut self) -> Self {
        for bin in self.state.bins.as_mut_slice() {
            *bin = 0;
        }
        self
    }
}

#[cfg(feature = "derive")]
impl<T: Clone, B: AsSlice<u32>> ResetMut for Histogram<T, B> where Self: Reset {}

impl<T: Clone + FloatCore, B: AsSlice<u32>> Sink<T> for Histogram<T, B> {
    #[inline]
    fn sink(&mut self, input: T) {
        let bins = self.state.bins.as_mut_slice();
        let n = bins.len();

        let range = self.config.max - self.config.min;

        if range == T::zero() {
            if n > 0 {
                bins[0] += 1;
            }
            return;
        }

        let normalized = (input - self.config.min) / range;
        let normalized = if normalized < T::zero() {
            T::zero()
        } else if normalized >= T::one() {
            T::one()
        } else {
            normalized
        };

        let n_t: T = NumCast::from(n).unwrap_or(T::zero());
        let bin_float = normalized * n_t;

        let bin_index = bin_float.to_usize().unwrap_or(0);
        let bin_index = if bin_index >= n { n - 1 } else { bin_index };

        bins[bin_index] += 1;
    }
}

impl<T: Clone, B> Finalize for Histogram<T, B> {
    type Output = B;
    fn finalize(self) -> Self::Output {
        self.state.bins
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_uniform_distribution() {
        // Test case from plan: N=4, min=0.0, max=4.0, inputs=[0.5, 1.5, 2.5, 3.5]
        // Expected: bins=[1,1,1,1] (each input falls into a different bin)
        const N: usize = 4;
        let config = Config::new(0.0f32, 4.0f32);
        let mut histogram: HistogramArray<f32, N> = Histogram::with_config(config);

        let inputs = vec![0.5f32, 1.5f32, 2.5f32, 3.5f32];
        for input in inputs {
            histogram.sink(input);
        }

        let bins = histogram.finalize();
        assert_eq!(
            bins,
            [1, 1, 1, 1],
            "Uniform distribution should fill each bin once"
        );
    }

    #[test]
    fn test_oob_clamping() {
        // Test case from plan: OOB input -1.0 should clamp to bin 0, 100.0 to last bin
        const N: usize = 4;
        let config = Config::new(0.0f32, 4.0f32);
        let mut histogram: HistogramArray<f32, N> = Histogram::with_config(config);

        histogram.sink(-1.0f32); // Below min, should clamp to bin 0
        histogram.sink(100.0f32); // Above max, should clamp to bin N-1

        let bins = histogram.finalize();
        assert_eq!(bins[0], 1, "Negative OOB should clamp to bin 0");
        assert_eq!(bins[3], 1, "Positive OOB should clamp to last bin");
    }

    #[test]
    fn test_bin_boundaries() {
        // Test that values exactly on bin boundaries are handled correctly
        const N: usize = 4;
        let config = Config::new(0.0f32, 4.0f32);
        let mut histogram: HistogramArray<f32, N> = Histogram::with_config(config);

        // With N=4, bin boundaries are at [0, 1, 2, 3, 4]
        // Bin 0: [0, 1), Bin 1: [1, 2), Bin 2: [2, 3), Bin 3: [3, 4]
        histogram.sink(0.0f32); // Min boundary, should be in bin 0
        histogram.sink(4.0f32); // Max boundary, should be in last bin
        histogram.sink(1.0f32); // Bin boundary
        histogram.sink(2.0f32); // Bin boundary

        let bins = histogram.finalize();
        // After normalization, these should distribute properly
        assert!(bins[0] > 0, "Bin 0 should have at least one entry");
        assert!(bins[3] > 0, "Bin 3 should have at least one entry");
    }

    #[test]
    fn test_concentration() {
        // Test that many values in the same range concentrate in one bin
        const N: usize = 4;
        let config = Config::new(0.0f32, 10.0f32);
        let mut histogram: HistogramArray<f32, N> = Histogram::with_config(config);

        // Feed values that should all fall into bin 0 (range [0, 2.5))
        for i in 0..10 {
            let val = (i as f32) * 0.2; // [0.0, 0.2, 0.4, ..., 1.8]
            histogram.sink(val);
        }

        let bins = histogram.finalize();
        assert_eq!(bins[0], 10, "All values should concentrate in bin 0");
        assert_eq!(bins[1] + bins[2] + bins[3], 0, "Other bins should be empty");
    }

    #[test]
    fn test_f64_support() {
        // Verify that Histogram works with f64
        const N: usize = 2;
        let config = Config::new(0.0f64, 2.0f64);
        let mut histogram: HistogramArray<f64, N> = Histogram::with_config(config);

        histogram.sink(0.5f64);
        histogram.sink(1.5f64);

        let bins = histogram.finalize();
        assert_eq!(bins[0], 1);
        assert_eq!(bins[1], 1);
    }

    #[test]
    fn test() {
        const N: usize = 20;
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        let config = Config::new(0.0f32, 20.0f32);
        let mut histogram: HistogramArray<f32, N> = Histogram::with_config(config);
        for value in input {
            histogram.sink(value);
        }
        let bins = histogram.finalize();
        assert_eq!(bins[0], 1);
        assert_eq!(bins[7], 2);
        assert_eq!(bins[19], 3);
    }

    #[test]
    #[should_panic(expected = "Histogram requires at least one bin")]
    fn from_parts_empty_bins_panics() {
        let config = Config::new(0.0, 10.0);
        let empty: &mut [u32] = &mut [];
        let _ = HistogramRefMut::from_parts(config, empty);
    }

    #[test]
    fn histogram_ref_mut_records_bins() {
        let config = Config::new(0.0, 3.0);
        let mut bins: [u32; 3] = [0; 3];
        let mut hist = HistogramRefMut::from_parts(config, &mut bins);
        hist.sink(0.5);
        hist.sink(1.5);
        hist.sink(2.5);
        hist.sink(0.5);
        let result = hist.finalize();
        assert_eq!(result.as_slice(), &[2, 1, 1]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_vec_backend_from_parts() {
        // Verify the `HistogramVec` alias works via `from_parts`.
        let config = Config::new(0.0f32, 4.0f32);
        let bins = vec![0u32; 4];
        let mut histogram: HistogramVec<f32> = Histogram::from_parts(config, bins);

        let inputs = vec![0.5f32, 1.5f32, 2.5f32, 3.5f32];
        for input in inputs {
            histogram.sink(input);
        }

        let bins = histogram.finalize();
        assert_eq!(bins, vec![1, 1, 1, 1]);
    }
}
