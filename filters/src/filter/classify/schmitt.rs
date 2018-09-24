// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialOrd;

use generic_array::typenum::U2;
use generic_array::GenericArray;

use signalo_traits::filter::Filter;

use signalo_traits::{Configurable, InitialState, Resettable, Stateful, StatefulUnsafe};

/// The [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger)'s configuration.
#[derive(Clone, Debug)]
pub struct Config<T, U> {
    /// [low, high] input thresholds.
    pub thresholds: GenericArray<T, U2>,
    /// [off, on] outputs.
    pub outputs: GenericArray<U, U2>,
}

/// The [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger)'s state.
#[derive(Clone, Debug)]
pub struct State {
    pub on: bool,
}

/// A [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger).
#[derive(Clone, Debug)]
pub struct Schmitt<T, U> {
    /// The filter's configuration.
    config: Config<T, U>,
    /// Current internal state.
    state: State,
}

impl<T, U> Schmitt<T, U>
where
    U: Clone,
{
    /// Creates a new `Schmitt` filter with given `thresholds` (`[low, high]`) and `outputs` (`[off, on]`).
    #[inline]
    pub fn new(config: Config<T, U>) -> Self {
        let state = Self::initial_state(&config);
        Schmitt { config, state }
    }
}

impl<T, U> Configurable for Schmitt<T, U> {
    type Config = Config<T, U>;

    fn config(&self) -> &Self::Config {
        &self.config
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

impl<'a, T, U> InitialState<&'a Config<T, U>> for Schmitt<T, U> {
    fn initial_state(_config: &'a Config<T, U>) -> Self::State {
        let on = false;
        State { on }
    }
}

impl<T, U> Resettable for Schmitt<T, U> {
    fn reset(&mut self) {
        self.state = Self::initial_state(self.config());
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
            false => input > self.config.thresholds[1],
            true => input >= self.config.thresholds[0],
        };
        let index = self.state.on as usize;
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn test() {
        let filter = Schmitt::new(Config {
            thresholds: arr![u8; 5, 10],
            outputs: u8::classes(),
        });
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
