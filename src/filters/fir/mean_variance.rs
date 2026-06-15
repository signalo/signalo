// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving average filters.

use circular_buffer::CircularBuffer;
use core::fmt;

use num_traits::{Num, Zero};

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Filter, Reset, State as StateTrait, StateMut,
};

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

/// The mean/variance filter's state.
#[derive(Clone)]
pub struct State<T, const N: usize> {
    /// Buffer of recent input values.
    pub taps: CircularBuffer<N, T>,
    /// The running sum of the window.
    pub sum: T,
    /// The running sum of squares of the window.
    pub sum_sq: T,
    /// Number of filled slots in the window.
    pub weight: usize,
}

impl<T, const N: usize> fmt::Debug for State<T, N>
where
    T: fmt::Debug,
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
/// # Complexity
///
/// - **Time per sample:** O(N); weight-to-T conversion iterates up to N times;
///   all other operations are O(1). The weight loop will be eliminated by the compiler for
///   primitive types once `FromPrimitive` is available.
/// - **Space:** O(N); circular tap buffer of N samples plus three scalar accumulators
///   (`sum`, `sum_sq`, `weight`).
#[derive(Clone)]
pub struct MeanVariance<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> Default for MeanVariance<T, N>
where
    T: Zero,
{
    fn default() -> Self {
        assert!(N > 0, "MeanVariance: window size N must be > 0");
        let state = State {
            taps: CircularBuffer::default(),
            sum: T::zero(),
            sum_sq: T::zero(),
            weight: 0,
        };
        Self { state }
    }
}

impl<T, const N: usize> fmt::Debug for MeanVariance<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MeanVariance")
            .field("state", &self.state)
            .finish()
    }
}

impl<T, const N: usize> StateTrait for MeanVariance<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> StateMut for MeanVariance<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for MeanVariance<T, N> {
    type Guts = State<T, N>;
}

impl<T, const N: usize> FromGuts for MeanVariance<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, const N: usize> IntoGuts for MeanVariance<T, N> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for MeanVariance<T, N>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for MeanVariance<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for MeanVariance<T, N>
where
    T: Clone + Num + PartialOrd,
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
        let _: MeanVariance<f32, 0> = MeanVariance::default();
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
        let filter: MeanVariance<f32, 3> = MeanVariance::default();
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
        let filter: MeanVariance<f32, 3> = MeanVariance::default();
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
        let filter: MeanVariance<f32, 5> = MeanVariance::default();
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
        let mut filter: MeanVariance<f32, 5> = MeanVariance::default();
        for _ in 0..5 {
            filter.filter(42.0);
        }
        for _ in 0..5 {
            let out = filter.filter(42.0);
            assert_abs_diff_eq!(out.variance, 0.0, epsilon = 1e-6);
        }
    }
}
