// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving average filters.

use std::fmt;

use arraydeque::Array;

use num_traits::{Num, Signed, Zero};

use signalo_traits::filter::Filter;

use traits::{InitialState, Resettable, Stateful, StatefulUnsafe};

use super::mean::Mean;

/// A mean filter's internal state.
#[derive(Clone)]
pub struct State<A>
where
    A: Array,
    A::Item: Clone,
{
    pub mean: Mean<A>,
    pub variance: Mean<A>,
}

impl<T, A> fmt::Debug for State<A>
where
    T: Clone + fmt::Debug,
    A: Array<Item = T> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("mean", &self.mean)
            .field("variance", &self.variance)
            .finish()
    }
}

/// A filter producing the moving average and variance over a given signal.
// #[derive(Clone, Default)]
pub struct MeanVariance<A>
where
    A: Array,
    A::Item: Clone,
{
    state: State<A>,
}

// impl<T, A> Clone for MeanVariance<A>
// where
//     T: Clone,
//     A: Clone + Array<Item = T>,
// {
//     fn clone(&self) -> Self {
//         Self {
//             state: self.state.clone(),
//             mean: self.mean.clone(),
//             variance: self.variance.clone(),
//         }
//     }
// }

impl<T, A> Default for MeanVariance<A>
where
    T: Default + Clone + Zero,
    A: Default + Array<Item = T>,
{
    fn default() -> Self {
        let state = Self::initial_state(());
        Self { state }
    }
}

impl<T, A> fmt::Debug for MeanVariance<A>
where
    T: Clone + fmt::Debug,
    A: Array<Item = T> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MeanVariance")
            .field("state", &self.state)
            .finish()
    }
}

impl<T, A> Stateful for MeanVariance<A>
where
    T: Clone,
    A: Array<Item = T>,
{
    type State = State<A>;
}

unsafe impl<T, A> StatefulUnsafe for MeanVariance<A>
where
    T: Clone,
    A: Array<Item = T>,
{
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, A> InitialState<()> for MeanVariance<A>
where
    T: Clone + Default + Zero,
    A: Array<Item = T> + Default,
{
    fn initial_state(_: ()) -> Self::State {
        let mean = Mean::default();
        let variance = Mean::default();
        State { mean, variance }
    }
}

impl<T, A> Resettable for MeanVariance<A>
where
    T: Clone + Default + Zero,
    A: Array<Item = T> + Default,
{
    fn reset(&mut self) {
        self.state = Self::initial_state(());
    }
}

impl<T, A> Filter<T> for MeanVariance<A>
where
    T: Copy + Num + Signed + PartialOrd + ::std::fmt::Debug, // FIXME
    A: Array<Item = T> + fmt::Debug,
{
    /// (mean, variance)
    type Output = (T, T);

    fn filter(&mut self, input: T) -> Self::Output {
        let mean_old = unsafe { self.state.mean.state().value.unwrap_or(input) };
        let mean_new = self.state.mean.filter(input);
        let deviation_old = (input - mean_old).abs();
        let deviation_new = (input - mean_new).abs();
        let squared = deviation_old * deviation_new;
        let variance = self.state.variance.filter(squared);
        (mean_new, variance)
    }
}

#[cfg(test)]
mod tests {
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
            0.000, 0.250, 8.833, 11.500, 11.889, 9.222, 8.667, 60.111, 71.000, 104.444, 57.111,
            46.889, 22.444, 44.444, 53.778, 155.333, 137.333, 156.000, 57.556, 178.889, 202.000,
            221.556, 104.000, 76.222, 82.111, 124.556, 1522.556, 2672.889, 3868.333, 2440.333,
            2267.222, 2752.222, 3427.445, 2479.445, 788.889, 168.778, 123.000, 78.222, 106.889,
            378.444, 1278.000, 2799.000, 3133.667, 2306.333, 755.000, 125.666, 1148.555, 2456.222,
            3252.777, 2323.777,
        ]
    }

    #[test]
    fn mean() {
        let filter: MeanVariance<[f32; 3]> = MeanVariance::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).0))
            .collect();
        assert_nearly_eq!(output, get_mean(), 0.001);
    }

    #[test]
    fn variance() {
        let filter: MeanVariance<[f32; 3]> = MeanVariance::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).1))
            .collect();
        assert_nearly_eq!(output, get_variance(), 0.001);
    }
}
