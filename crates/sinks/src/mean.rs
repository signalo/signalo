// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Value mean sinks.

use num_traits::Num;

use signalo_traits::{Filter, Finalize, Sink};

#[derive(Clone, Debug)]
struct State<T> {
    // number of values seen so far
    count: T,
    // mean of values seen so far
    mean: T,
}

/// A sink that computes the mean and variance of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct Mean<T> {
    // The sink's state
    state: Option<State<T>>,
}

impl<T> Filter<T> for Mean<T>
where
    T: Clone + PartialOrd + Num,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let (old_count, old_mean) = match &self.state {
            Some(State {
                ref count,
                ref mean,
            }) => (count.clone(), mean.clone()),
            None => (T::zero(), T::zero()),
        };

        let count = old_count + T::one();
        let mean = old_mean.clone() + ((input - old_mean) / count.clone());

        let state = State {
            count,
            mean: mean.clone(),
        };

        self.state = Some(state);

        mean
    }
}

impl<T> Sink<T> for Mean<T>
where
    Self: Filter<T>,
{
    #[inline]
    fn sink(&mut self, input: T) {
        let _ = self.filter(input);
    }
}

impl<T> Finalize for Mean<T>
where
    T: Clone + PartialOrd + Num,
{
    type Output = Option<T>;

    #[inline]
    fn finalize(self) -> Self::Output {
        self.state.map(|state| {
            let State { mean, .. } = state;
            mean
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let sink: Mean<f32> = Mean::default();
        let mean = sink.finalize();
        assert_nearly_eq!(mean, None);
    }

    #[test]
    fn non_empty() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ];
        let mut sink = Mean::default();
        for input in input {
            sink.sink(input);
        }
        let mean = sink.finalize();
        assert_nearly_eq!(mean, Some(26.56));
    }
}
