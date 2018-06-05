// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Exponential moving average & variance filters.

use num_traits::{One, Zero, Num};

use signalo_traits::filter::Filter;

use super::mean::Mean;

/// A filter producing the exponential moving average and variance over a given signal.
#[derive(Clone, Debug)]
pub struct MeanVariance<T> {
    state: Option<T>,
    mean: Mean<T>,
    variance: Mean<T>,
}

impl<T> MeanVariance<T>
where
    T: Copy + PartialOrd + Zero + One
{
    /// Creates a new `MeanVariance` filter with `beta = 1.0 / n` with `n` being the filter width.
    #[inline]
    pub fn new(beta: T) -> Self {
        assert!(beta > T::zero() && beta <= T::one());
        MeanVariance {
            state: None,
            mean: Mean::new(beta),
            variance: Mean::new(beta),
        }
    }

    /// Returns the filter's `beta` coefficient.
    #[inline]
    pub fn beta(&self) -> &T {
        self.mean.beta()
    }
}

impl<T> Filter<T> for MeanVariance<T>
where
    T: Copy + Num,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_mean() -> Vec<f32> {
        vec![
            0.000, 0.250, 1.938, 1.953, 2.715, 4.036, 7.027, 6.020, 9.265, 8.449, 9.837,
            9.628, 9.471, 11.353, 12.765, 10.574, 10.930, 13.198, 14.898, 12.924, 11.443,
            12.332, 12.999, 12.249, 14.937, 13.703, 38.027, 33.020, 29.265, 26.449, 46.337,
            36.003, 33.502, 28.376, 24.532, 23.649, 22.987, 22.490, 25.368, 21.026, 43.019,
            34.264, 32.948, 28.711, 25.533, 23.150, 43.363, 35.272, 32.454, 30.340
        ]
    }

    fn get_variance() -> Vec<f32> {
        vec![
            0.000, 0.188, 8.684, 6.513, 6.626, 10.207, 34.493, 28.910, 53.271, 41.953,
            37.242, 28.063, 21.121, 26.470, 25.832, 33.778, 25.715, 34.710, 34.709,
            37.728, 34.875, 28.529, 22.732, 18.735, 35.722, 31.362, 1798.539, 1424.107,
            1110.382, 856.581, 1829.006, 1692.140, 1287.865, 1044.710, 827.864, 623.237,
            468.744, 352.298, 289.063, 273.354, 1656.166, 1472.066, 1109.246, 885.793,
            694.640, 538.022, 1629.149, 1418.237, 1087.501, 829.026
        ]
    }

    #[test]
    fn mean() {
        let filter = MeanVariance::new(0.25);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input).0)
        }).collect();
        assert_nearly_eq!(output, get_mean(), 0.001);
    }

    #[test]
    fn variance() {
        let filter = MeanVariance::new(0.25);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input).1)
        }).collect();
        assert_nearly_eq!(output, get_variance(), 0.001);
    }
}
