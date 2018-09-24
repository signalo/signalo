// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialEq;

use generic_array::typenum::U2;
use generic_array::GenericArray;
use num_traits::Zero;

use signalo_traits::filter::Filter;

use signalo_traits::{Configurable, InitialState, Resettable, Stateful, StatefulUnsafe};

/// The [Debounce](https://en.wikipedia.org/wiki/Switch#Contact_bounce) filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, U> {
    /// Threshold of how long input must remain same to be accepted.
    pub threshold: usize,
    /// Value to debounce.
    pub predicate: T,
    /// [off, on] output.
    pub outputs: GenericArray<U, U2>,
}

/// The [Debounce](https://en.wikipedia.org/wiki/Switch#Contact_bounce) filter's state.
#[derive(Clone, Debug)]
pub struct State {
    /// Counter of how long input was the same.
    count: usize,
}

/// A [Debounce](https://en.wikipedia.org/wiki/Switch#Contact_bounce) filter.
#[derive(Clone, Debug)]
pub struct Debounce<T, U> {
    /// The filter's configuration.
    config: Config<T, U>,
    /// Counter of how long input was the same.
    state: State,
}

impl<T, U> Debounce<T, U>
where
    T: Clone + Zero,
{
    /// Creates a new `Debounce` filter with given `threshold`, `predicate` and `outputs` (`[off, on]`).
    #[inline]
    pub fn new(config: Config<T, U>) -> Self {
        let state = Self::initial_state(&config);
        Debounce { config, state }
    }
}

impl<T, U> Configurable for Debounce<T, U> {
    type Config = Config<T, U>;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> Stateful for Debounce<T, U> {
    type State = State;
}

unsafe impl<T, U> StatefulUnsafe for Debounce<T, U> {
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<'a, T, U> InitialState<&'a Config<T, U>> for Debounce<T, U> {
    fn initial_state(_config: &'a Config<T, U>) -> Self::State {
        let count = 0;
        State { count }
    }
}

impl<T, U> Resettable for Debounce<T, U> {
    fn reset(&mut self) {
        self.state = Self::initial_state(self.config());
    }
}

impl<T, U> Filter<T> for Debounce<T, U>
where
    T: Clone + PartialEq<T>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        if input == self.config.predicate {
            self.state.count = (self.state.count + 1).min(self.config.threshold);
        } else {
            self.reset();
        }
        let index = (self.state.count >= self.config.threshold) as usize;
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn test() {
        let filter = Debounce::new(Config {
            threshold: 3,
            predicate: 1,
            outputs: u8::classes(),
        });
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
