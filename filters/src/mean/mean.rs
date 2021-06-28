// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving average filters.

use std::fmt;

use num_traits::{Num, Zero};

use signalo_traits::Filter;
use signalo_traits::{FromGuts, Guts, IntoGuts, Reset, State as StateTrait, StateMut};

use crate::circular_buffer::CircularBuffer;

/// The mean filter's state.
#[derive(Clone)]
pub struct State<T, const N: usize> {
    /// The current mean value.
    pub mean: Option<T>,
    /// The current taps buffer.
    pub taps: CircularBuffer<T, N>,
    /// The current weight.
    pub weight: T,
}

impl<T, const N: usize> fmt::Debug for State<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("mean", &self.mean)
            .field("taps", &self.taps)
            .field("weight", &self.weight)
            .finish()
    }
}

/// A mean filter producing the moving median over a given signal.
#[derive(Clone)]
pub struct Mean<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> Default for Mean<T, N>
where
    T: Zero,
{
    fn default() -> Self {
        let state = {
            let mean = None;
            let taps = CircularBuffer::default();
            let weight = T::zero();
            State { mean, taps, weight }
        };
        Self { state }
    }
}

impl<T, const N: usize> fmt::Debug for Mean<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Mean").field("state", &self.state).finish()
    }
}

impl<T, const N: usize> StateTrait for Mean<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> StateMut for Mean<T, N> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> Guts for Mean<T, N> {
    type Guts = State<T, N>;
}

impl<T, const N: usize> FromGuts for Mean<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, const N: usize> IntoGuts for Mean<T, N> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for Mean<T, N>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive_reset_mut")]
impl<T, N> ResetMut for Mean<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Mean<T, N>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let old_mean = self.state.mean.clone().unwrap_or_else(|| input.clone());
        let old_weight = self.state.weight.clone();

        #[allow(clippy::option_if_let_else)]
        let (mean, weight) = if let Some(old_input) = self.state.taps.push_back(input.clone()) {
            let mean = old_mean - old_input + input;
            (mean, old_weight)
        } else {
            let mean = old_mean + input;
            let weight = old_weight + T::one();
            (mean, weight)
        };
        self.state.mean = Some(mean.clone());
        self.state.weight = weight.clone();
        mean / weight
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
            0.000, 0.500, 2.667, 3.333, 4.667, 5.000, 9.667, 9.000, 12.667, 9.333, 13.000, 9.667,
            10.667, 11.667, 14.333, 12.667, 11.000, 12.000, 17.333, 15.667, 11.333, 9.667, 12.333,
            13.333, 16.000, 14.333, 48.000, 46.333, 49.000, 18.000, 47.333, 43.000, 45.667, 14.667,
            17.333, 15.667, 18.333, 21.000, 25.333, 21.000, 50.333, 41.667, 48.667, 17.667, 20.333,
            16.000, 45.333, 43.667, 46.333, 19.667,
        ]
    }

    #[test]
    fn test() {
        let filter: Mean<f32, 3> = Mean::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
