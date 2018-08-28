// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialOrd;

use signalo_traits::filter::Filter;

use traits::{InitialState, Resettable, Stateful, StatefulUnsafe};

/// A Schmitt trigger's internal state.
#[derive(Clone, Debug)]
pub struct State {
    pub on: bool,
}

/// A [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger) filter.
#[derive(Clone, Debug)]
pub struct Schmitt<T, U> {
    /// [low, high] input thresholds.
    thresholds: [T; 2],
    /// [off, on] outputs.
    outputs: [U; 2],
    /// Current internal state.
    state: State,
}

impl<T, U> Schmitt<T, U>
where
    U: Clone,
{
    /// Creates a new `Schmitt` filter with given `thresholds` (`[low, high]`) and `outputs` (`[off, on]`).
    #[inline]
    pub fn new(thresholds: [T; 2], outputs: [U; 2]) -> Self {
        let state = Self::initial_state(());
        Schmitt {
            thresholds,
            outputs,
            state,
        }
    }
}

impl<T, U> Stateful for Schmitt<T, U> {
    type State = State;
}

unsafe impl<T, U> StatefulUnsafe for Schmitt<T, U> {
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> InitialState<()> for Schmitt<T, U> {
    fn initial_state(_: ()) -> Self::State {
        let on = false;
        State { on }
    }
}

impl<T, U> Resettable for Schmitt<T, U> {
    fn reset(&mut self) {
        self.state = Self::initial_state(());
    }
}

impl<T, U> Filter<T> for Schmitt<T, U>
where
    T: PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        self.state.on = match self.state.on {
            false => input > self.thresholds[1],
            true => input >= self.thresholds[0],
        };
        let index = self.state.on as usize;
        self.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn test() {
        let filter = Schmitt::new([5, 10], u8::classes());
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
            vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        );
    }
}
