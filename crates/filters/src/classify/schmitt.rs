// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Schmitt trigger filters.

use core::cmp::PartialOrd;

use signalo_traits::{
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, FromGuts, Guts, IntoGuts, Reset,
    State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "derive")]
use signalo_traits::ResetMut;

/// The [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger)'s configuration.
#[derive(Clone, Debug)]
pub struct Config<T, U> {
    /// [low, high] input thresholds.
    pub thresholds: [T; 2],
    /// [off, on] outputs.
    pub outputs: [U; 2],
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

impl<T, U> ConfigClone for Schmitt<T, U>
where
    Config<T, U>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, U> StateMut for Schmitt<T, U> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> Guts for Schmitt<T, U> {
    type Guts = (Config<T, U>, State);
}

impl<T, U> FromGuts for Schmitt<T, U> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, U> IntoGuts for Schmitt<T, U> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, U> Reset for Schmitt<T, U> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, U> ResetMut for Schmitt<T, U> where Self: Reset {}

impl<T, U> Filter<T> for Schmitt<T, U>
where
    T: PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        self.state.on = if self.state.on {
            input >= self.config.thresholds[0]
        } else {
            input > self.config.thresholds[1]
        };
        let index: usize = self.state.on.into();
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::Classification;

    #[test]
    fn test() {
        let filter = Schmitt::with_config(Config {
            thresholds: [5, 10],
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
