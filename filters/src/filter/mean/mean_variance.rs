// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving average filters.

use std::fmt;

use arraydeque::Array;

use num_traits::Num;

use signalo_traits::filter::Filter;
use traits::Stateful;

use super::mean::Mean;

/// A filter producing the moving average and variance over a given signal.
// #[derive(Clone, Default)]
pub struct MeanVariance<A>
where
    A: Array,
{
    state: Option<A::Item>,
    mean: Mean<A>,
    variance: Mean<A>,
}

impl<T, A> Clone for MeanVariance<A>
where
    T: Clone,
    A: Clone + Array<Item = T>,
{
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            mean: self.mean.clone(),
            variance: self.variance.clone(),
        }
    }
}

impl<T, A> Default for MeanVariance<A>
where
    T: Default,
    A: Default + Array<Item = T>,
{
    fn default() -> Self {
        Self {
            state: None,
            mean: Mean::default(),
            variance: Mean::default(),
        }
    }
}

impl<T, A> fmt::Debug for MeanVariance<A>
where
    T: fmt::Debug,
    A: Array<Item = T> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MeanVariance")
            .field("state", &self.state)
            .field("mean", &self.mean)
            .field("variance", &self.variance)
            .finish()
    }
}

impl<T, A> Filter<T> for MeanVariance<A>
where
    T: Copy + Num,
    A: Array<Item = T> + fmt::Debug,
{
    /// (mean, variance)
    type Output = (T, T);

    fn filter(&mut self, input: T) -> Self::Output {
        let mean_old = self.state.unwrap_or(input);
        let mean = self.mean.filter(input);
        let squared = (input - mean_old) * (input - mean);
        let variance = self.variance.filter(squared);
        self.state = Some(mean);
        (mean, variance)
    }
}

impl<T, A> Stateful for MeanVariance<A>
where
    Mean<A>: Default,
    A: Array<Item = T> + fmt::Debug,
{
    #[inline]
    fn reset(&mut self) {
        self.mean.reset();
        self.variance.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0,
            17.0, 4.0, 12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0,
            18.0, 18.0, 18.0, 106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0,
            109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_mean() -> Vec<f32> {
        vec![
            0.000, 0.500, 2.667, 3.333, 4.667, 5.000, 9.667, 9.000, 12.667, 9.333, 13.000,
            9.667, 10.667, 11.667, 14.333, 12.667, 11.000, 12.000, 17.333, 15.667, 11.333,
            9.667, 12.333, 13.333, 16.000, 14.333, 48.000, 46.333, 49.000, 18.000, 47.333,
            43.000, 45.667, 14.667, 17.333, 15.667, 18.333, 21.000, 25.333, 21.000, 50.333,
            41.667, 48.667, 17.667, 20.333, 16.000, 45.333, 43.667, 46.333, 19.667
        ]
    }

    fn get_variance() -> Vec<f32> {
        vec![
            0.000, 0.250, 9.556, 9.852, 9.870, 3.815, 26.741, 39.889, 57.667, 41.852, 30.074,
            9.852, 2.815, 12.519, 16.370, 45.852, 34.370, 53.630, 30.889, 60.963, 49.481,
            48.889, 23.778, 13.852, 29.889, 33.815, 2061.222, 2322.000, 2606.111, 576.111,
            2013.667, 2257.111, 2368.556, 665.815, 132.000, 27.074, 13.667, 11.259, 42.296,
            112.667, 1833.556, 2271.074, 2279.000, 576.259, 103.593, 20.556, 1723.296,
            2094.741, 2241.148, 488.000
        ]
    }

    #[test]
    fn mean() {
        let filter: MeanVariance<[f32; 3]> = MeanVariance::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        println!("{:?}", input);
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input).0)
        }).collect();
        assert_nearly_eq!(output, get_mean(), 0.001);
    }

    #[test]
    fn variance() {
        let filter: MeanVariance<[f32; 3]> = MeanVariance::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input).1)
        }).collect();
        assert_nearly_eq!(output, get_variance(), 0.001);
    }
}
