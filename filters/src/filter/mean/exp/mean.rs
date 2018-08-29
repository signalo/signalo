// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Exponential moving average filters.

use num_traits::Num;

use signalo_traits::filter::Filter;

use traits::{InitialState, Resettable, Stateful, StatefulUnsafe};

/// A mean filter's internal state.
#[derive(Clone, Debug)]
pub struct State<T> {
    pub value: Option<T>,
}

/// A filter producing the exponential moving average over a given signal.
#[derive(Clone, Debug)]
pub struct Mean<T> {
    beta: T,
    state: State<T>,
}

impl<T> Mean<T> {
    /// Creates a new `Mean` filter with `beta = 1.0 / n` with `n` being the filter width.
    ///
    /// Note: `beta` is required to be in the range between `0.0` and `1.0`.
    #[inline]
    pub fn new(beta: T) -> Self {
        let state = Self::initial_state(());
        Mean { beta, state }
    }
}

impl<T> Mean<T> {
    /// Returns the filter's `beta` coefficient.
    #[inline]
    pub fn beta(&self) -> &T {
        &self.beta
    }
}

impl<T> Stateful for Mean<T> {
    type State = State<T>;
}

unsafe impl<T> StatefulUnsafe for Mean<T> {
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> InitialState<()> for Mean<T> {
    fn initial_state(_: ()) -> Self::State {
        let value = None;
        State { value }
    }
}

impl<T> Resettable for Mean<T> {
    fn reset(&mut self) {
        self.state = Self::initial_state(());
    }
}

impl<T> Filter<T> for Mean<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let mean = match &self.state.value {
            None => input,
            Some(ref state) => state.clone() + ((input - state.clone()) * self.beta.clone()),
        };
        self.state.value = Some(mean.clone());
        mean
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

    fn get_output() -> Vec<f32> {
        vec![
            0.000, 0.250, 1.938, 1.953, 2.715, 4.036, 7.027, 6.020, 9.265, 8.449, 9.837, 9.628,
            9.471, 11.353, 12.765, 10.574, 10.930, 13.198, 14.898, 12.924, 11.443, 12.332, 12.999,
            12.249, 14.937, 13.703, 38.027, 33.020, 29.265, 26.449, 46.337, 36.003, 33.502, 28.376,
            24.532, 23.649, 22.987, 22.490, 25.368, 21.026, 43.019, 34.264, 32.948, 28.711, 25.533,
            23.150, 43.363, 35.272, 32.454, 30.340,
        ]
    }

    #[test]
    fn test() {
        let filter = Mean::new(0.25);
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
