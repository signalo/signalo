// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving bounds filters.

use core::fmt;

use num_traits::Num;

use signalo_traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Filter, Reset, State as StateTrait, StateMut,
};

#[cfg(feature = "derive")]
use signalo_traits::ResetMut;

/// Maximum value filter tracking the largest value in a sliding window.
///
/// Efficiently computes the moving maximum using an optimized data structure.
pub mod max;

/// Minimum value filter tracking the smallest value in a sliding window.
///
/// Efficiently computes the moving minimum using an optimized data structure.
pub mod min;

/// The bounds filter's state.
#[derive(Clone)]
pub struct State<T, const N: usize> {
    /// The internal `min` filter.
    pub min: self::min::Min<T, N>,
    /// The internal `max` filter.
    pub max: self::max::Max<T, N>,
}

impl<T, const N: usize> Default for State<T, N> {
    fn default() -> Self {
        Self {
            min: self::min::Min::default(),
            max: self::max::Max::default(),
        }
    }
}

impl<T, const N: usize> fmt::Debug for State<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

/// A bounds filter producing the moving bounds over a given signal.
#[derive(Clone)]
pub struct Bounds<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> Default for Bounds<T, N> {
    fn default() -> Self {
        Self {
            state: State::default(),
        }
    }
}

impl<T, const N: usize> fmt::Debug for Bounds<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Bounds")
            .field("state", &self.state)
            .finish()
    }
}

impl<T, const N: usize> StateTrait for Bounds<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> StateMut for Bounds<T, N> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Bounds<T, N> {
    type Guts = State<T, N>;
}

impl<T, const N: usize> FromGuts for Bounds<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, const N: usize> IntoGuts for Bounds<T, N> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for Bounds<T, N> {
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Bounds<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Bounds<T, N>
where
    T: Clone + Num + PartialOrd,
{
    type Output = (T, T);

    fn filter(&mut self, input: T) -> Self::Output {
        let min = self.state.min.filter(input.clone());
        let max = self.state.max.filter(input);
        (min, max)
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

    fn get_output() -> Vec<(f32, f32)> {
        vec![
            (0.0, 0.0),
            (0.0, 1.0),
            (0.0, 7.0),
            (1.0, 7.0),
            (2.0, 7.0),
            (2.0, 8.0),
            (5.0, 16.0),
            (3.0, 16.0),
            (3.0, 19.0),
            (3.0, 19.0),
            (6.0, 19.0),
            (6.0, 14.0),
            (9.0, 14.0),
            (9.0, 17.0),
            (9.0, 17.0),
            (4.0, 17.0),
            (4.0, 17.0),
            (4.0, 20.0),
            (12.0, 20.0),
            (7.0, 20.0),
            (7.0, 20.0),
            (7.0, 15.0),
            (7.0, 15.0),
            (10.0, 15.0),
            (10.0, 23.0),
            (10.0, 23.0),
            (10.0, 111.0),
            (10.0, 111.0),
            (18.0, 111.0),
            (18.0, 18.0),
            (18.0, 106.0),
            (5.0, 106.0),
            (5.0, 106.0),
            (5.0, 26.0),
            (13.0, 26.0),
            (13.0, 21.0),
            (13.0, 21.0),
            (21.0, 21.0),
            (21.0, 34.0),
            (8.0, 34.0),
            (8.0, 109.0),
            (8.0, 109.0),
            (8.0, 109.0),
            (8.0, 29.0),
            (16.0, 29.0),
            (16.0, 16.0),
            (16.0, 104.0),
            (11.0, 104.0),
            (11.0, 104.0),
            (11.0, 24.0),
        ]
    }

    #[test]
    fn test() {
        const N: usize = 3;

        let filter: Bounds<f32, N> = Bounds::default();

        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();

        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(output, get_output());
    }
}
