// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Delay filters.

use core::fmt;

use circular_buffer::CircularBuffer;
use num_traits::Num;

use signalo_traits::Filter;
use signalo_traits::{FromGuts, Guts, IntoGuts, Reset, State as StateTrait, StateMut};

/// The delay filter's state.
#[derive(Clone)]
pub struct State<T, const N: usize> {
    /// The current taps buffer.
    pub taps: CircularBuffer<N, T>,
}

impl<T, const N: usize> fmt::Debug for State<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State").field("taps", &self.taps).finish()
    }
}

/// A delay filter producing the moving median over a given signal.
#[derive(Clone)]
pub struct Delay<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> Default for Delay<T, N> {
    fn default() -> Self {
        let state = {
            let taps = CircularBuffer::default();
            State { taps }
        };
        Self { state }
    }
}

impl<T, const N: usize> fmt::Debug for Delay<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Delay").field("state", &self.state).finish()
    }
}

impl<T, const N: usize> StateTrait for Delay<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> StateMut for Delay<T, N> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> Guts for Delay<T, N> {
    type Guts = State<T, N>;
}

impl<T, const N: usize> FromGuts for Delay<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, const N: usize> IntoGuts for Delay<T, N> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for Delay<T, N> {
    fn reset(self) -> Self {
        Self::default()
    }
}

impl<T, const N: usize> Filter<T> for Delay<T, N>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        loop {
            if let Some(delayed) = self.state.taps.push_back(input.clone()) {
                return delayed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nearly_eq::assert_nearly_eq;

    use super::*;

    #[allow(clippy::wildcard_imports)]

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
            0.0, 0.0, 0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0,
            17.0, 4.0, 12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0,
            18.0, 106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0,
            16.0, 16.0, 16.0, 104.0, 11.0,
        ]
    }

    #[test]
    fn test() {
        let filter: Delay<f32, 2> = Delay::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
