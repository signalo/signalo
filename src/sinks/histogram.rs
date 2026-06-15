// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Histogram sinks for signal distribution analysis.
//!
//! Divides a signal's range into equally-sized bins and counts how many samples fall into each bin,
//! providing a distribution histogram. Out-of-bounds values are clamped to edge bins.

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
pub struct Config<T, const N: usize> {
    /// Minimum value of the histogram range
    pub min: T,
    /// Maximum value of the histogram range
    pub max: T,
}

impl<T: Clone, const N: usize> Config<T, N> {
    /// Creates a new histogram configuration.
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

impl<T: Clone + Default, const N: usize> Default for Config<T, N> {
    fn default() -> Self {
        Self {
            min: T::default(),
            max: T::default(),
        }
    }
}

/// State for a histogram sink.
///
/// Maintains an array of bin counters. Each bin is a u32 counter that increments
/// when input values fall within that bin's range.
#[derive(Clone, Debug)]
pub struct State<const N: usize> {
    /// Array of bin counters
    pub bins: [u32; N],
}

impl<const N: usize> Default for State<N> {
    fn default() -> Self {
        Self { bins: [0; N] }
    }
}

/// A histogram sink that tracks the distribution of input values.
///
/// Divides the range [min, max] into N equal-width bins and counts how many input
/// values fall into each bin. Out-of-bounds values are clamped to the edge bins.
#[derive(Clone, Debug)]
pub struct Histogram<T: Clone, const N: usize> {
    config: Config<T, N>,
    state: State<N>,
}

impl<T: Clone + Default, const N: usize> Default for Histogram<T, N> {
    fn default() -> Self {
        Self {
            config: Config::default(),
            state: State::default(),
        }
    }
}

impl<T: Clone, const N: usize> ConfigTrait for Histogram<T, N> {
    type Config = Config<T, N>;
}

impl<T: Clone, const N: usize> StateTrait for Histogram<T, N> {
    type State = State<N>;
}

impl<T: Clone, const N: usize> WithConfig for Histogram<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T: Clone, const N: usize> ConfigRef for Histogram<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: Clone, const N: usize> ConfigClone for Histogram<T, N> {
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T: Clone, const N: usize> StateMut for Histogram<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T: Clone, const N: usize> HasGuts for Histogram<T, N> {
    type Guts = (Config<T, N>, State<N>);
}

impl<T: Clone, const N: usize> FromGuts for Histogram<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T: Clone, const N: usize> IntoGuts for Histogram<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T: Clone, const N: usize> Reset for Histogram<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T: Clone, const N: usize> ResetMut for Histogram<T, N> where Self: Reset {}

impl<T: Clone + FloatCore, const N: usize> Sink<T> for Histogram<T, N> {
    #[inline]
    fn sink(&mut self, input: T) {
        let range = self.config.max - self.config.min;

        if range == T::zero() {
            if N > 0 {
                self.state.bins[0] += 1;
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

        let n_t: T = NumCast::from(N).unwrap_or(T::zero());
        let bin_float = normalized * n_t;

        let bin_index = bin_float.to_usize().unwrap_or(0);
        let bin_index = if bin_index >= N { N - 1 } else { bin_index };

        self.state.bins[bin_index] += 1;
    }
}

impl<T: Clone, const N: usize> Finalize for Histogram<T, N> {
    type Output = [u32; N];
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
        let mut histogram: Histogram<f32, N> = Histogram::with_config(config);

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
        let mut histogram: Histogram<f32, N> = Histogram::with_config(config);

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
        let mut histogram: Histogram<f32, N> = Histogram::with_config(config);

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
        let mut histogram: Histogram<f32, N> = Histogram::with_config(config);

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
        let mut histogram: Histogram<f64, N> = Histogram::with_config(config);

        histogram.sink(0.5f64);
        histogram.sink(1.5f64);

        let bins = histogram.finalize();
        assert_eq!(bins[0], 1);
        assert_eq!(bins[1], 1);
    }

    #[test]
    fn test() {
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        const N: usize = 20;
        let config = Config::new(0.0f32, 20.0f32);
        let mut histogram: Histogram<f32, N> = Histogram::with_config(config);
        for value in input {
            histogram.sink(value);
        }
        let bins = histogram.finalize();
        assert_eq!(bins[0], 1);
        assert_eq!(bins[7], 2);
        assert_eq!(bins[19], 3);
    }
}
