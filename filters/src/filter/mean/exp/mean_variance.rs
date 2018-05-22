// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Exponential moving average & variance filters.

use std::ops::{Sub, Add, Mul, Div};

use num_traits::{Zero, One};

use signalo_traits::filter::Filter;
use traits::Stateful;

/// A filter producing the approximated moving median over a given signal.
#[derive(Clone, Debug)]
pub struct MeanVariance<T> {
    beta: T,
    state: Option<(T, T)>,
}

impl<T> MeanVariance<T>
where
    T: PartialOrd + Zero + One
{
    /// Creates a new `MeanVariance` filter with `beta = 1.0 / n` with `n` being the filter width.
    #[inline]
    pub fn new(beta: T) -> Self {
        assert!(beta > T::zero() && beta <= T::one());
        MeanVariance { beta, state: None }
    }

    /// Returns the filter's `beta` coefficient.
    #[inline]
    pub fn beta(&self) -> &T {
        &self.beta
    }
}

impl<T> Filter<T> for MeanVariance<T>
where
    T: Copy + Zero + One + Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T>
{
    /// (mean, variance)
    type Output = (T, T);

    fn filter(&mut self, input: T) -> Self::Output {
        let state = match self.state {
            None => {
                (input, T::zero())
            },
            Some((old_mean, old_variance)) => {
                let delta = (input - old_mean) * self.beta;
                let mean = old_mean + delta;
                let variance = (T::one() - self.beta) * (old_variance + (delta * (input - mean)));
                (mean, variance)
            },
        };
        self.state = Some(state);
        state
    }
}

impl<T> Stateful for MeanVariance<T> {
    #[inline]
    fn reset(&mut self) {
        self.state = None;
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
            0.0, 0.140625, 6.5126953, 4.885071, 4.969288, 7.6550264, 25.869503, 21.682716,
            39.953407, 31.464375, 27.93155, 21.047108, 15.840708, 19.852734, 19.373913,
            25.333334, 19.286137, 26.032684, 26.03156, 28.296118, 26.156588, 21.396917,
            17.048645, 14.051302, 26.79162, 23.521252, 1348.9042, 1068.0803, 832.7865,
            642.4359, 1371.7548, 1269.105, 965.8983, 783.5323, 620.89777, 467.42792,
            351.5579, 264.2236, 216.79709, 205.01526, 1242.1246, 1104.0492, 831.9343,
            664.34485, 520.9803, 403.51617, 1221.8617, 1063.6779, 815.62573, 621.76965
        ]
    }

    #[test]
    fn mean() {
        let filter = MeanVariance::new(0.25);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        let mean: Vec<_> = output.iter().map(|(mean, _)| mean).collect();
        let variance: Vec<_> = output.iter().map(|(_, variance)| variance).collect();
        assert_nearly_eq!(mean, get_mean(), 0.001);
        assert_nearly_eq!(variance, get_variance(), 0.001);
    }
}
