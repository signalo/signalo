// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving average filters.

use core::fmt;

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::{Num, Zero};

use crate::storage::RingBuffer;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Configuration for a [`Mean`] filter.
///
/// This type is a unit struct because all mean-filter parameters are
/// encoded in the type system (window size `N` for the `*Array` aliases,
/// or the capacity of the supplied ring-buffer). It exists so that
/// [`Mean`] can satisfy the [`WithConfig`] and related config traits
/// uniformly with other filters.
#[derive(Clone, Debug, Default)]
pub struct Config;

/// The mean filter's state.
///
/// Generic over the ring-buffer backend `R` that stores the tap window.
/// Use [`MeanArray`] for stack-allocated tap storage or [`MeanVec`] for
/// heap-allocated tap storage.
#[derive(Clone)]
pub struct State<T, R> {
    /// The current mean value.
    pub mean: Option<T>,
    /// The current taps buffer.
    pub taps: R,
    /// The current weight (number of samples accumulated so far, as `T`).
    pub weight: T,
}

impl<T, R> fmt::Debug for State<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("mean", &self.mean)
            .field("taps", &self.taps)
            .field("weight", &self.weight)
            .finish()
    }
}

/// A mean filter producing the moving average over a given signal.
///
/// # Storage
///
/// The tap ring-buffer backend is selected by the `R` type parameter.
/// Prefer the concrete aliases for common use:
///
/// - [`MeanArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`MeanVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Complexity
///
/// - **Time per sample:** O(N) when the window is full (sum recomputed from scratch to
///   prevent floating-point drift); O(1) during the initial N-sample warm-up.
/// - **Space:** O(N); circular buffer of N samples plus scalar accumulators.
#[derive(Clone)]
pub struct Mean<T, R> {
    config: Config,
    state: State<T, R>,
}

/// A mean filter backed by a const-generic [`FixedCircularBuffer`] tap buffer.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The tap
/// ring-buffer lives entirely on the stack.
pub type MeanArray<T, const N: usize> = Mean<T, FixedCircularBuffer<T, N>>;

/// A mean filter backed by a heap-allocated [`HeapCircularBuffer`] tap buffer.
///
/// Requires the `alloc` feature. Use [`Mean::from_parts`] to construct
/// this variant, since the tap buffer cannot be `Default`-constructed without
/// knowing the desired capacity at compile time.
#[cfg(feature = "alloc")]
pub type MeanVec<T> = Mean<T, HeapCircularBuffer<T>>;

/// A mean filter that borrows a [`CircularBuffer`] tap buffer.
///
/// This alias allows sharing a caller-owned ring buffer without taking
/// ownership of it. Construct via [`Mean::from_parts`], passing
/// a `&mut CircularBuffer<T>` for the tap buffer.
pub type MeanRefMut<'a, T> = Mean<T, &'a mut CircularBuffer<T>>;

impl<T, const N: usize> Default for MeanArray<T, N>
where
    T: Zero,
{
    /// Creates a [`MeanArray`] with an empty tap buffer.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    fn default() -> Self {
        assert!(N > 0, "Mean: window size N must be > 0");
        let state = State {
            mean: None,
            taps: FixedCircularBuffer::new(),
            weight: T::zero(),
        };
        Self {
            config: Config,
            state,
        }
    }
}

impl<T, R> fmt::Debug for Mean<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Mean").field("state", &self.state).finish()
    }
}

impl<T, R> Mean<T, R>
where
    R: RingBuffer<T>,
{
    /// Creates a [`Mean`] filter from an already-constructed `taps` ring-buffer.
    ///
    /// Use this constructor when the tap storage is not `Default`-constructible,
    /// e.g. for [`MeanVec`] whose capacity must be known at runtime.
    ///
    /// The `taps` buffer is taken as-is with its current contents. The
    /// accumulator (`mean` and `weight`) starts fresh, so the mean will
    /// converge to the correct value over `N` samples as pre-existing entries
    /// are evicted.
    ///
    /// # Expected storage state
    ///
    /// For predictable output from the first sample, pass an empty buffer.
    ///
    /// # Panics
    ///
    /// Panics if `taps.capacity()` is zero.
    pub fn from_parts(config: Config, taps: R) -> Self
    where
        T: Zero,
    {
        assert!(
            taps.capacity() > 0,
            "Mean: window size (taps capacity) must be > 0"
        );
        let state = State {
            mean: None,
            taps,
            weight: T::zero(),
        };
        Self { config, state }
    }
}

impl<T, R> ConfigTrait for Mean<T, R> {
    type Config = Config;
}

impl<T, R> StateTrait for Mean<T, R> {
    type State = State<T, R>;
}

impl<T, const N: usize> WithConfig for MeanArray<T, N>
where
    T: Zero,
{
    type Output = Self;

    /// Creates a [`MeanArray`] from a configuration.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Mean: window size N must be > 0");
        let state = State {
            mean: None,
            taps: FixedCircularBuffer::new(),
            weight: T::zero(),
        };
        Self { config, state }
    }
}

impl<T, R> ConfigRef for Mean<T, R> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, R> ConfigClone for Mean<T, R> {
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, R> StateMut for Mean<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for Mean<T, R> {
    type Guts = (Config, State<T, R>);
}

impl<T, R> FromGuts for Mean<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, R> IntoGuts for Mean<T, R> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for MeanArray<T, N>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for MeanArray<T, N> where Self: Reset {}

impl<T, R> Filter<T> for Mean<T, R>
where
    T: Clone + Num,
    R: RingBuffer<T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let old_weight = self.state.weight.clone();

        #[allow(clippy::option_if_let_else)]
        let (mean, weight) = if let Some(_old_input) = self.state.taps.push_back(input.clone()) {
            let sum = self
                .state
                .taps
                .iter()
                .fold(T::zero(), |acc, x| acc + x.clone());
            (sum, old_weight)
        } else {
            let old_mean = self.state.mean.clone().unwrap_or_else(T::zero);
            let mean = old_mean + input;
            let weight = old_weight + T::one();
            (mean, weight)
        };
        self.state.mean = Some(mean.clone());
        self.state.weight = weight.clone();
        mean / weight
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    #[should_panic(expected = "window size N must be > 0")]
    fn zero_window_panics() {
        let _: MeanArray<f32, 0> = MeanArray::default();
    }

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.000, 0.500, 2.667, 3.333, 4.667, 5.000, 9.667, 9.000, 12.667, 9.333, 13.000, 9.667,
            10.667, 11.667, 14.333, 12.667, 11.000, 12.000, 17.333, 15.667, 11.333, 9.667, 12.333,
            13.333, 16.000, 14.333, 48.000, 46.333, 49.000, 18.000, 47.333, 43.000, 45.667, 14.667,
            17.333, 15.667, 18.333, 21.000, 25.333, 21.000, 50.333, 41.667, 48.667, 17.667, 20.333,
            16.000, 45.333, 43.667, 46.333, 19.667,
        ]
    }

    #[test]
    fn test() {
        let filter: MeanArray<f32, 3> = MeanArray::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_abs_diff_eq!(output.as_slice(), get_output().as_slice(), epsilon = 0.001);
    }

    #[test]
    fn test_non_zero_start() {
        let filter: MeanArray<f32, 3> = MeanArray::default();
        let inputs = [10.0, 20.0, 30.0];
        let expected_outputs = vec![10.0, 15.0, 20.0];
        let outputs: Vec<_> = inputs
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_abs_diff_eq!(
            outputs.as_slice(),
            expected_outputs.as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn no_float_drift_over_long_run() {
        let mut filter: MeanArray<f64, 4> = MeanArray::default();
        for _ in 0..1_000_000 {
            filter.filter(1.0);
            filter.filter(3.0);
            filter.filter(1.0);
            filter.filter(3.0);
        }
        // Buffer is [1,3,1,3]
        // Adding 2.0 replaces 1.0. Buffer becomes [3,1,3,2], sum 9.0, mean 2.25.
        let result = filter.filter(2.0);
        assert!((result - 2.25).abs() < 1e-10, "mean drifted to {result}");
    }
}
