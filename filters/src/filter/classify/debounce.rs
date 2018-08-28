// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialEq;

use num_traits::Zero;

use signalo_traits::filter::Filter;

use traits::{InitialState, Resettable, Stateful, StatefulUnsafe};

/// A [Debounce](https://en.wikipedia.org/wiki/Switch#Contact_bounce) filter.
#[derive(Clone, Debug)]
pub struct Debounce<T, U> {
    /// Threshold of how long input must remain same to be accepted.
    threshold: usize,
    /// [off, on] output.
    outputs: [U; 2],
    /// Value to debounce.
    predicate: T,
    /// Counter of how long input was the same.
    state: usize,
}

impl<T, U> Debounce<T, U>
where
    T: Clone + Zero,
{
    /// Creates a new `Schmitt` filter with given `threshold`, `predicate` and `outputs` (`[off, on]`).
    #[inline]
    pub fn new(threshold: usize, predicate: T, outputs: [U; 2]) -> Self {
        let state = Self::initial_state(());
        Debounce {
            threshold,
            outputs,
            predicate,
            state,
        }
    }
}

impl<T, U> Stateful for Debounce<T, U> {
    type State = usize;
}

unsafe impl<T, U> StatefulUnsafe for Debounce<T, U> {
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> InitialState<()> for Debounce<T, U> {
    fn initial_state(_: ()) -> Self::State {
        0
    }
}

impl<T, U> Resettable for Debounce<T, U> {
    fn reset(&mut self) {
        self.state = Self::initial_state(());
    }
}

impl<T, U> Filter<T> for Debounce<T, U>
where
    T: Clone + PartialEq<T>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        if input == self.predicate {
            self.state = (self.state + 1).min(self.threshold);
        } else {
            self.reset();
        }
        let index = (self.state >= self.threshold) as usize;
        self.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn test() {
        let filter = Debounce::new(3, 1, u8::classes());
        let input = vec![0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(
            output,
            vec![0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }
}
