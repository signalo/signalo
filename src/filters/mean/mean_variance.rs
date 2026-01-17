// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving average filters.

use core::fmt;

use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Filter, Reset, State as StateTrait, StateMut,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

use super::mean::Mean;

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
    /// The current mean value.
    pub mean: Mean<T, N>,
    /// The current variance value.
    pub variance: Mean<T, N>,
}

impl<T, const N: usize> fmt::Debug for State<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("mean", &self.mean)
            .field("variance", &self.variance)
            .finish()
    }
}

/// A mean/variance filter producing the moving average and variance over a given signal.
#[derive(Clone)]
pub struct MeanVariance<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> Default for MeanVariance<T, N>
where
    Mean<T, N>: Default,
{
    fn default() -> Self {
        let state = {
            let mean = Mean::default();
            let variance = Mean::default();
            State { mean, variance }
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
    unsafe fn state_mut(&mut self) -> &mut Self::State {
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
    Mean<T, N>: Default,
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
        // Filters the input value, returning the current mean and variance.
        // Calculates variance using Welford's method.
        let mean_old = unsafe {
            self.state
                .mean
                .state_mut()
                .mean
                .clone()
                .unwrap_or_else(|| input.clone())
        };
        let mean = self.state.mean.filter(input.clone());
        let deviation_old = input.clone() - mean_old;
        let deviation_new = input - mean.clone();
        let squared = deviation_old * deviation_new;
        let variance = self.state.variance.filter(squared);
        Output { mean, variance }
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use nearly_eq::assert_nearly_eq;

    use super::*;

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
            0.000, 0.250, 8.833, 11.500, 10.778, -3.889, -4.444, 48.111, 37.222, 70.667, 14.000,
            37.556, 13.111, -8.889, -31.556, 70.000, 88.000, 69.333, -57.556, 81.111, 173.556,
            154.000, 11.556, -16.222, -22.111, 45.222, 1443.222, 2672.889, 3868.333, 2440.333,
            2267.222, 2752.222, 3427.445, 2479.445, 788.889, 58.556, -33.444, -78.222, -106.889,
            210.889, 1110.444, 2799.000, 3133.667, 2306.333, 755.000, 125.666, 1148.555, 2456.222,
            3252.777, 1991.556,
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
        assert_nearly_eq!(output, get_mean(), 0.001);
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
        assert_nearly_eq!(output, get_variance(), 0.001);
    }
}
