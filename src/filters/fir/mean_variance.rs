// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving average filters.

use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use core::fmt;

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

/// Output of `MeanVariance` filter.
#[derive(Clone, Debug)]
pub struct Output<T> {
    /// Mean of values.
    pub mean: T,
    /// Variance of values.
    pub variance: T,
}

/// Configuration for a [`MeanVariance`] filter.
///
/// This type is a unit struct because all mean-variance-filter parameters are
/// encoded in the type system (window size `N` for the `*Array` aliases,
/// or the capacity of the supplied ring-buffer). It exists so that
/// [`MeanVariance`] can satisfy the [`WithConfig`] and related config traits
/// uniformly with other filters.
#[derive(Clone, Debug, Default)]
pub struct Config;

/// The mean/variance filter's state.
///
/// Generic over the ring-buffer backend `R` that stores the tap window.
/// Use [`MeanVarianceArray`] for stack-allocated tap storage or
/// [`MeanVarianceVec`] for heap-allocated tap storage.
#[derive(Clone)]
pub struct State<T, R> {
    /// Buffer of recent input values.
    pub taps: R,
    /// The running sum of the window.
    pub sum: T,
    /// The running sum of squares of the window.
    pub sum_sq: T,
    /// Number of filled slots in the window.
    pub weight: usize,
}

impl<T, R> fmt::Debug for State<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("taps", &self.taps)
            .field("sum", &self.sum)
            .field("sum_sq", &self.sum_sq)
            .field("weight", &self.weight)
            .finish()
    }
}

/// A mean/variance filter producing the moving average and variance over a given signal.
///
/// # Storage
///
/// The tap ring-buffer backend is selected by the `R` type parameter.
/// Prefer the concrete aliases for common use:
///
/// - [`MeanVarianceArray<T, N>`] — stack-allocated, `no_std`-friendly.
/// - [`MeanVarianceVec<T>`] — heap-allocated, requires the `alloc` feature.
///
/// # Complexity
///
/// - **Time per sample:** O(N); weight-to-T conversion iterates up to N times;
///   all other operations are O(1). The weight loop will be eliminated by the compiler for
///   primitive types once `FromPrimitive` is available.
/// - **Space:** O(N); circular tap buffer of N samples plus three scalar accumulators
///   (`sum`, `sum_sq`, `weight`).
#[derive(Clone)]
pub struct MeanVariance<T, R> {
    config: Config,
    state: State<T, R>,
}

/// A mean/variance filter backed by a const-generic [`FixedCircularBuffer`] tap buffer.
///
/// This alias is the `no_std`-friendly, zero-allocation form. The tap
/// ring-buffer lives entirely on the stack.
pub type MeanVarianceArray<T, const N: usize> = MeanVariance<T, FixedCircularBuffer<T, N>>;

/// A mean/variance filter backed by a heap-allocated [`HeapCircularBuffer`] tap buffer.
///
/// Requires the `alloc` feature. Use [`MeanVariance::from_parts`] to construct
/// this variant, since the tap buffer cannot be `Default`-constructed without
/// knowing the desired capacity at compile time.
#[cfg(feature = "alloc")]
pub type MeanVarianceVec<T> = MeanVariance<T, HeapCircularBuffer<T>>;

/// A mean/variance filter that borrows a [`CircularBuffer`] tap buffer.
///
/// This alias allows sharing a caller-owned ring buffer without taking
/// ownership of it. Construct via [`MeanVariance::from_parts`], passing
/// a `&mut CircularBuffer<T>` for the tap buffer.
pub type MeanVarianceRefMut<'a, T> = MeanVariance<T, &'a mut CircularBuffer<T>>;

impl<T, const N: usize> Default for MeanVarianceArray<T, N>
where
    T: Zero,
{
    /// Creates a [`MeanVarianceArray`] with an empty tap buffer.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    fn default() -> Self {
        assert!(N > 0, "MeanVariance: window size N must be > 0");
        let state = State {
            taps: FixedCircularBuffer::new(),
            sum: T::zero(),
            sum_sq: T::zero(),
            weight: 0,
        };
        Self {
            config: Config,
            state,
        }
    }
}

impl<T, R> fmt::Debug for MeanVariance<T, R>
where
    T: fmt::Debug,
    R: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MeanVariance")
            .field("state", &self.state)
            .finish()
    }
}

impl<T, R> MeanVariance<T, R>
where
    R: RingBuffer<T>,
{
    /// Creates a [`MeanVariance`] filter from an already-constructed `taps` ring-buffer.
    ///
    /// Use this constructor when the tap storage is not `Default`-constructible,
    /// e.g. for [`MeanVarianceVec`] whose capacity must be known at runtime.
    ///
    /// The `taps` buffer is taken as-is with its current contents. The
    /// accumulators (`sum` and `sum_sq`) start fresh, so the mean and variance
    /// will converge to their correct values over `N` samples as pre-existing
    /// entries are evicted.
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
            "MeanVariance: window size (taps capacity) must be > 0"
        );
        let state = State {
            taps,
            sum: T::zero(),
            sum_sq: T::zero(),
            weight: 0,
        };
        Self { config, state }
    }
}

impl<T, R> ConfigTrait for MeanVariance<T, R> {
    type Config = Config;
}

impl<T, R> StateTrait for MeanVariance<T, R> {
    type State = State<T, R>;
}

impl<T, const N: usize> WithConfig for MeanVarianceArray<T, N>
where
    T: Zero,
{
    type Output = Self;

    /// Creates a [`MeanVarianceArray`] from a configuration.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "MeanVariance: window size N must be > 0");
        let state = State {
            taps: FixedCircularBuffer::new(),
            sum: T::zero(),
            sum_sq: T::zero(),
            weight: 0,
        };
        Self { config, state }
    }
}

impl<T, R> ConfigRef for MeanVariance<T, R> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, R> ConfigClone for MeanVariance<T, R> {
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, R> StateMut for MeanVariance<T, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, R> HasGuts for MeanVariance<T, R> {
    type Guts = (Config, State<T, R>);
}

impl<T, R> FromGuts for MeanVariance<T, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, R> IntoGuts for MeanVariance<T, R> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for MeanVarianceArray<T, N>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for MeanVarianceArray<T, N> where Self: Reset {}

impl<T, R> Filter<T> for MeanVariance<T, R>
where
    T: Clone + Num + PartialOrd,
    R: RingBuffer<T>,
{
    type Output = Output<T>;

    fn filter(&mut self, input: T) -> Self::Output {
        let input_sq = input.clone() * input.clone();
        if let Some(old) = self.state.taps.push_back(input.clone()) {
            let old_sq = old.clone() * old.clone();
            self.state.sum = self.state.sum.clone() - old + input;
            self.state.sum_sq = self.state.sum_sq.clone() - old_sq + input_sq;
        } else {
            self.state.sum = self.state.sum.clone() + input;
            self.state.sum_sq = self.state.sum_sq.clone() + input_sq;
            self.state.weight += 1;
        }

        let weight = {
            let mut w = T::zero();
            for _ in 0..self.state.weight {
                w = w + T::one();
            }
            w
        };

        let mean = self.state.sum.clone() / weight.clone();
        let sum_sq_n = self.state.sum.clone() * self.state.sum.clone() / weight.clone();
        let variance = (self.state.sum_sq.clone() - sum_sq_n) / weight;
        Output { mean, variance }
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
        let _: MeanVarianceArray<f32, 0> = MeanVarianceArray::default();
    }

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_mean() -> Vec<f32> {
        vec![
            0.000, 0.500, 2.667, 3.333, 4.667, 5.000, 9.667, 9.000, 12.667, 9.333, 13.000, 9.667,
            10.667, 11.667, 14.333, 12.667, 11.000, 12.000, 17.333, 15.667, 11.333, 9.667, 12.333,
            13.333, 16.000, 14.333, 48.000, 46.333, 49.000, 18.000, 47.333, 43.000, 45.667, 14.667,
            17.333, 15.667, 18.333, 21.000, 25.333, 21.000, 50.333, 41.667, 48.667, 17.667, 20.333,
            16.000, 45.333, 43.667, 46.333, 19.667,
        ]
    }

    fn get_variance() -> Vec<f32> {
        vec![
            0.000, 0.250, 9.556, 6.889, 4.222, 6.000, 21.556, 28.667, 48.222, 48.222, 28.667,
            10.889, 5.556, 14.222, 14.222, 37.556, 28.667, 42.667, 14.222, 37.556, 37.556, 14.222,
            14.222, 5.556, 28.667, 37.556, 2012.667, 2101.556, 1922.000, 0.000, 1720.889, 2012.667,
            1893.556, 74.889, 37.556, 14.222, 14.222, 0.000, 37.556, 112.667, 1833.556, 2266.889,
            1893.556, 74.889, 37.556, 0.000, 1720.889, 1824.222, 1690.889, 37.556,
        ]
    }

    #[test]
    fn mean() {
        let filter: MeanVarianceArray<f32, 3> = MeanVarianceArray::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).mean))
            .collect();
        assert_abs_diff_eq!(output.as_slice(), get_mean().as_slice(), epsilon = 0.001);
    }

    #[test]
    fn variance() {
        let filter: MeanVarianceArray<f32, 3> = MeanVarianceArray::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).variance))
            .collect();
        assert_abs_diff_eq!(
            output.as_slice(),
            get_variance().as_slice(),
            epsilon = 0.001
        );
    }

    #[test]
    fn variance_always_positive() {
        let filter: MeanVarianceArray<f32, 5> = MeanVarianceArray::default();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).variance))
            .collect();
        for var in output {
            assert!(var >= -1e-6, "variance should be >= 0, was {var}");
        }
    }

    #[test]
    fn variance_constant_signal_zero() {
        let mut filter: MeanVarianceArray<f32, 5> = MeanVarianceArray::default();
        for _ in 0..5 {
            filter.filter(42.0);
        }
        for _ in 0..5 {
            let out = filter.filter(42.0);
            assert_abs_diff_eq!(out.variance, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "MeanVariance: window size (taps capacity) must be > 0")]
    fn from_parts_zero_capacity_panics() {
        let taps = FixedCircularBuffer::<f32, 0>::new();
        let _ = MeanVariance::<f32, _>::from_parts(Config::default(), taps);
    }
}
