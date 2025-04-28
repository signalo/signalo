// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Debounce filters.

use core::cmp::PartialEq;

use signalo_traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use signalo_traits::ResetMut;

/// The [Debounce](https://en.wikipedia.org/wiki/Switch#Contact_bounce) filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, U> {
    /// Threshold of how long input must remain same to be accepted.
    pub threshold: usize,
    /// Value to debounce.
    pub predicate: T,
    /// [off, on] output.
    pub outputs: [U; 2],
}

/// The [Debounce](https://en.wikipedia.org/wiki/Switch#Contact_bounce) filter's state.
#[derive(Clone, Debug)]
pub struct State {
    /// Counter of how long input was the same.
    pub count: usize,
}

/// A [Debounce](https://en.wikipedia.org/wiki/Switch#Contact_bounce) filter.
#[derive(Clone, Debug)]
pub struct Debounce<T, U> {
    /// The filter's configuration.
    config: Config<T, U>,
    /// Counter of how long input was the same.
    state: State,
}

impl<T, U> ConfigTrait for Debounce<T, U> {
    type Config = Config<T, U>;
}

impl<T, U> StateTrait for Debounce<T, U> {
    type State = State;
}

impl<T, U> WithConfig for Debounce<T, U> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let count = 0;
            State { count }
        };
        Self { config, state }
    }
}

impl<T, U> ConfigRef for Debounce<T, U> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> ConfigClone for Debounce<T, U>
where
    Config<T, U>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, U> StateMut for Debounce<T, U> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> HasGuts for Debounce<T, U> {
    type Guts = (Config<T, U>, State);
}

impl<T, U> FromGuts for Debounce<T, U> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, U> IntoGuts for Debounce<T, U> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, U> Reset for Debounce<T, U> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, U> ResetMut for Debounce<T, U> where Self: Reset {}

impl<T, U> Filter<T> for Debounce<T, U>
where
    T: Clone + PartialEq<T>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        if input == self.config.predicate {
            self.state.count = self.state.count.saturating_add(1);
        } else {
            self.state.count = 0;
        }
        let index: usize = (self.state.count >= self.config.threshold).into();
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use crate::classify::Classification;

    use super::*;

    #[test]
    fn test() {
        let filter = Debounce::with_config(Config {
            threshold: 3,
            predicate: 1,
            outputs: u8::classes(),
        });
        let input = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1];
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
