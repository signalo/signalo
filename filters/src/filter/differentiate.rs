// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Differentiation filters.

use std::ops::Sub;

use num_traits::Zero;

use signalo_traits::filter::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigRef, Destruct, Reset, State as StateTrait, StateMut, WithConfig,
};

/// The differentiate filter's configuration.
#[derive(Default, Clone, Debug)]
pub struct Config {}

/// The differentiate filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Current value.
    pub value: Option<T>,
}

/// A differentiate filter that produces the derivative of the signal.
#[derive(Clone, Debug)]
pub struct Differentiate<T> {
    config: Config,
    state: State<T>,
}

impl<T> Default for Differentiate<T>
where
    Config: Default,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T> ConfigTrait for Differentiate<T> {
    type Config = Config;
}

impl<T> StateTrait for Differentiate<T> {
    type State = State<T>;
}

impl<T> WithConfig for Differentiate<T> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let value = None;
            State { value }
        };
        Self { config, state }
    }
}

impl<T> ConfigRef for Differentiate<T>
where
    Config: Clone,
{
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> StateMut for Differentiate<T> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> Destruct for Differentiate<T> {
    type Output = (Config, State<T>);

    fn destruct(self) -> Self::Output {
        (self.config, self.state)
    }
}

impl<T> Reset for Differentiate<T> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

impl<T> Filter<T> for Differentiate<T>
where
    T: Clone + Sub<T, Output = T> + Zero,
{
    type Output = <T as Sub<T>>::Output;

    fn filter(&mut self, input: T) -> Self::Output {
        let output = match &self.state.value {
            None => T::zero(),
            Some(ref state) => input.clone() - state.clone(),
        };
        self.state.value = Some(input);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let filter = Differentiate::default();
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
                0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -13.0, 16.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0,
                -13.0, 8.0, 8.0, 0.0, -13.0
            ]
        );
    }
}
