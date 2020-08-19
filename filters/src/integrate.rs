// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Differentiation filters.

use std::ops::Sub;

use num_traits::Zero;

use signalo_traits::Filter;
use signalo_traits::{FromGuts, Guts, IntoGuts, Reset, State as StateTrait, StateMut};

/// The integration filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Current value.
    pub value: T,
}

/// A integration filter that produces the integral of the signal.
#[derive(Clone, Debug)]
pub struct Integrate<T> {
    state: State<T>,
}

impl<T> Default for Integrate<T>
where
    T: Zero,
{
    fn default() -> Self {
        let state = {
            let value = T::zero();
            State { value }
        };
        Self { state }
    }
}

impl<T> StateTrait for Integrate<T> {
    type State = State<T>;
}

impl<T> StateMut for Integrate<T> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> Guts for Integrate<T> {
    type Guts = State<T>;
}

impl<T> FromGuts for Integrate<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T> IntoGuts for Integrate<T> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T> Reset for Integrate<T>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive_reset_mut")]
impl<T> ResetMut for Integrate<T> where Self: Reset {}

impl<T> Filter<T> for Integrate<T>
where
    T: Clone + Sub<T, Output = T> + Zero,
{
    type Output = <T as Sub<T>>::Output;

    fn filter(&mut self, input: T) -> Self::Output {
        let state = self.state.value.clone() + input.clone();
        self.state.value = state.clone();
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let filter = Integrate::default();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(
            output,
            vec![
                0.0, 1.0, 8.0, 10.0, 15.0, 23.0, 39.0, 42.0, 61.0, 67.0, 81.0, 90.0, 99.0, 116.0,
                133.0, 137.0, 149.0, 169.0, 189.0, 196.0
            ]
        );
    }
}
