// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Value mean & variance sinks.

use num_traits::Num;

use signalo_traits::{Filter, Finalize, Sink};

/// Output of `MeanVariance` filter.
#[derive(Clone, Debug)]
pub struct Output<T> {
    /// Mean of values.
    pub mean: T,
    /// Variance of values.
    pub variance: T,
}

#[derive(Clone, Debug)]
struct State<T> {
    // number of values seen so far
    count: T,
    // mean of values seen so far
    mean: T,
    // squared distance from the mean
    variance: T,
}

/// A sink that computes the mean and variance of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct MeanVariance<T> {
    state: Option<State<T>>,
}

impl<T> Filter<T> for MeanVariance<T>
where
    T: Clone + Num,
{
    type Output = Output<T>;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let (old_count, old_mean, old_variance) = match &self.state {
            Some(State {
                ref count,
                ref mean,
                ref variance,
            }) => (count.clone(), mean.clone(), variance.clone()),
            None => (T::zero(), T::zero(), T::zero()),
        };

        let count = old_count + T::one();
        let mean = old_mean.clone() + ((input.clone() - old_mean.clone()) / count.clone());

        let old_delta = input.clone() - old_mean;
        let delta = input - mean.clone();

        let variance = old_variance + (old_delta * delta);

        self.state = Some(State {
            count,
            mean: mean.clone(),
            variance: variance.clone(),
        });

        Output { mean, variance }
    }
}

impl<T> Sink<T> for MeanVariance<T>
where
    Self: Filter<T>,
{
    #[inline]
    fn sink(&mut self, input: T) {
        let _ = self.filter(input);
    }
}

impl<T> Finalize for MeanVariance<T>
where
    T: PartialOrd + Num,
{
    type Output = Option<Output<T>>;

    #[inline]
    fn finalize(self) -> Self::Output {
        self.state.map(|state| {
            let State {
                count,
                mean,
                variance,
            } = state;
            let variance = if count > T::one() {
                variance / (count - T::one())
            } else {
                variance
            };
            Output { mean, variance }
        })
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
        let mut sink = MeanVariance::default();
        for input in input {
            sink.sink(input);
        }
        if let Some(Output { mean, variance }) = sink.finalize() {
            assert_nearly_eq!(mean, 26.56);
            assert_nearly_eq!(variance, 1347.68);
        } else {
            panic!("Expected Some(…), found None.");
        }
    }
}
