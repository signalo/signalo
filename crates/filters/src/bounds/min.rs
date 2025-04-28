// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving minimum filters.

use core::fmt;

use num_traits::Num;

use signalo_traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Filter, Reset, State as StateTrait, StateMut,
};

#[cfg(feature = "derive")]
use signalo_traits::ResetMut;

use circular_buffer::CircularBuffer;

/// The min filter's state.
#[derive(Clone)]
pub struct State<T, const N: usize> {
    /// The discrete timestamp of the latest input.
    pub time: usize,
    /// The current taps buffer.
    pub taps: CircularBuffer<N, (T, usize)>,
}

impl<T, const N: usize> fmt::Debug for State<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("time", &self.time)
            .field("taps", &self.taps)
            .finish()
    }
}

/// A min filter producing the moving minimum over a given signal.
#[derive(Clone)]
pub struct Min<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> Default for Min<T, N> {
    fn default() -> Self {
        Self {
            state: State {
                time: 0,
                taps: CircularBuffer::default(),
            },
        }
    }
}

impl<T, const N: usize> fmt::Debug for Min<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Min").field("state", &self.state).finish()
    }
}

impl<T, const N: usize> StateTrait for Min<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> StateMut for Min<T, N> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Min<T, N> {
    type Guts = State<T, N>;
}

impl<T, const N: usize> FromGuts for Min<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, const N: usize> IntoGuts for Min<T, N> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for Min<T, N> {
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Min<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Min<T, N>
where
    T: Clone + Num + PartialOrd,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let current_time = self.state.time;

        // pop all items that have left the moving window, from the front:
        while self
            .state
            .taps
            .front()
            .map_or(false, |(_, time)| time + N <= current_time)
        {
            let _ = self.state.taps.pop_front();
        }

        // pop all items larger than the input, from the back:
        while self
            .state
            .taps
            .back()
            .map_or(false, |(value, _)| &input < value)
        {
            let _ = self.state.taps.pop_back();
        }

        // push the input, to the back:
        self.state.taps.push_back((input, current_time));

        if self.state.time < usize::MAX {
            self.state.time += 1;
        } else {
            // Time has overflown, so we need to adjust our state accordingly:
            let offset = self.state.time - N;
            for (_, time) in self.state.taps.iter_mut() {
                *time -= offset;
            }
            self.state.time = N;
        }

        #[allow(clippy::unwrap_used)]
        self.state.taps.front().unwrap().0.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

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
            0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 5.0, 3.0, 3.0, 3.0, 6.0, 6.0, 9.0, 9.0, 9.0, 4.0, 4.0,
            4.0, 12.0, 7.0, 7.0, 7.0, 7.0, 10.0, 10.0, 10.0, 10.0, 10.0, 18.0, 18.0, 18.0, 5.0,
            5.0, 5.0, 13.0, 13.0, 13.0, 21.0, 21.0, 8.0, 8.0, 8.0, 8.0, 8.0, 16.0, 16.0, 16.0,
            11.0, 11.0, 11.0,
        ]
    }

    #[test]
    fn test() {
        const N: usize = 3;

        let filter: Min<f32, N> = Min::default();

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(output, get_output());
    }
}
