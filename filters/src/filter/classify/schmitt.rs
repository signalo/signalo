// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Schmitt trigger filters.

use std::cmp::PartialOrd;

use generic_array::typenum::U2;
use generic_array::GenericArray;

use signalo_traits::filter::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigRef, Destruct, Reset, State as StateTrait, StateMut, WithConfig,
};

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
    /// The current state.
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

impl<T, U> ConfigTrait for Schmitt<T, U> {
    type Config = Config<T, U>;
}

impl<T, U> StateTrait for Schmitt<T, U> {
    type State = State;
}

impl<T, U> WithConfig for Schmitt<T, U> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let on = false;
            State { on }
        };
        Self { config, state }
    }
}

impl<T, U> ConfigRef for Schmitt<T, U> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> StateMut for Schmitt<T, U> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> Destruct for Schmitt<T, U> {
    type Output = (Config<T, U>, State);

    fn destruct(self) -> Self::Output {
        (self.config, self.state)
    }
}

impl<T, U> Reset for Schmitt<T, U> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
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
        let filter = Schmitt::with_config(Config {
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
