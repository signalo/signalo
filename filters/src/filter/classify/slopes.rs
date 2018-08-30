// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::{Ordering, PartialOrd};

use generic_array::typenum::*;
use generic_array::GenericArray;

use signalo_traits::filter::Filter;

use filter::classify::Classification;
use signalo_traits::{InitialState, Resettable, Stateful, StatefulUnsafe};

/// A slope's kind.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Slope {
    /// A rising slope.
    Rising,
    /// A flat slope.
    None,
    /// A falling slope.
    Falling,
}

impl Default for Slope {
    fn default() -> Self {
        Slope::None
    }
}

impl Classification<Slope, U3> for Slope {
    fn classes() -> GenericArray<Slope, U3> {
        arr![Slope; Slope::Rising, Slope::None, Slope::Falling]
    }
}

/// A slope detection filter's internal state.
#[derive(Clone, Debug)]
pub struct State<T> {
    pub input: Option<T>,
}

/// A slope detection filter.
#[derive(Clone, Debug)]
pub struct Slopes<T, U> {
    state: State<T>,
    /// [rising, flat, falling] outputs.
    outputs: GenericArray<U, U3>,
}

impl<T, U> Slopes<T, U>
where
    U: Clone,
{
    /// Creates a new `Slopes` filter with given `threshold` and `outputs` (`[rising, none, falling]`).
    #[inline]
    pub fn new(outputs: GenericArray<U, U3>) -> Self {
        let state = Self::initial_state(());
        Slopes { state, outputs }
    }
}

impl<T, U> Stateful for Slopes<T, U> {
    type State = State<T>;
}

unsafe impl<T, U> StatefulUnsafe for Slopes<T, U> {
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> InitialState<()> for Slopes<T, U> {
    fn initial_state(_: ()) -> Self::State {
        State { input: None }
    }
}

impl<T, U> Resettable for Slopes<T, U> {
    fn reset(&mut self) {
        self.state = Self::initial_state(());
    }
}

impl<T, U> Filter<T> for Slopes<T, U>
where
    T: Clone + PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let index = match self.state.input {
            None => 1, // None
            Some(ref state) => match state.partial_cmp(&input).unwrap() {
                Ordering::Less => 0,    // Rising
                Ordering::Equal => 1,   // None
                Ordering::Greater => 2, // Falling
            },
        };
        self.state.input = Some(input);
        self.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn test() {
        use self::Slope::*;

        let filter = Slopes::new(Slope::classes());
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
            0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(
            output,
            vec![
                None, Rising, Rising, Falling, Rising, Rising, Rising, Falling, Rising, Falling,
                Rising, Falling, None, Rising, None, Falling, Rising, Rising, None, Falling,
            ]
        );
    }
}
