// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Value mean & variance sinks.

use num_traits::Num;

use signalo_traits::{Filter, Finalize, Sink};

use bounds::{Bounds, Output as BoundsOutput};
use mean_variance::{MeanVariance, Output as MeanVarianceOutput};

/// Output of `Statistics` filter.
#[derive(Clone, Debug)]
pub struct Output<T> {
    /// Smallest value.
    pub min: T,
    /// Largest value.
    pub max: T,
    /// Mean of values.
    pub mean: T,
    /// Variance of values.
    pub variance: T,
}

#[derive(Clone, Default, Debug)]
struct State<T> {
    bounds: Bounds<T>,
    mean_variance: MeanVariance<T>,
}

/// A sink that computes the [descriptive statistics](https://en.wikipedia.org/wiki/Descriptive_statistics) of a signal.
#[derive(Clone, Default, Debug)]
pub struct Statistics<T> {
    state: State<T>,
}

impl<T> Filter<T> for Statistics<T>
where
    T: Clone + Num,
    Bounds<T>: Filter<T, Output = BoundsOutput<T>>,
    MeanVariance<T>: Filter<T, Output = MeanVarianceOutput<T>>,
{
    type Output = Output<T>;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let BoundsOutput { min, max } = self.state.bounds.filter(input.clone());
        let MeanVarianceOutput { mean, variance } = self.state.mean_variance.filter(input.clone());

        Output {
            min,
            max,
            mean,
            variance,
        }
    }
}

impl<T> Sink<T> for Statistics<T>
where
    Self: Filter<T>,
{
    #[inline]
    fn sink(&mut self, input: T) {
        let _ = self.filter(input);
    }
}

impl<T> Finalize for Statistics<T>
where
    T: PartialOrd + Num,
{
    type Output = Option<Output<T>>;

    #[inline]
    fn finalize(self) -> Self::Output {
        let min_max = self.state.bounds.finalize();
        let mean_variance = self.state.mean_variance.finalize();
        match (min_max, mean_variance) {
            (Some(min_max), Some(mean_variance)) => {
                let BoundsOutput { min, max } = min_max;
                let MeanVarianceOutput { mean, variance } = mean_variance;
                Some(Output {
                    min,
                    max,
                    mean,
                    variance,
                })
            }
            (None, None) => None,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ];
        let mut sink = Statistics::default();
        for input in input {
            sink.sink(input);
        }
        if let Some(Output {
            min,
            max,
            mean,
            variance,
        }) = sink.finalize()
        {
            assert_nearly_eq!(min, 0.0);
            assert_nearly_eq!(max, 180.0);
            assert_nearly_eq!(mean, 26.56);
            assert_nearly_eq!(variance, 1347.68);
        } else {
            panic!("Expected Some(…), found None.");
        }
    }
}
