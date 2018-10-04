// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Thresholding filters.

use std::cmp::PartialOrd;

use generic_array::typenum::U2;
use generic_array::GenericArray;

use signalo_traits::filter::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigRef, Destruct, Reset, State as StateTrait, StateMut, WithConfig,
};

/// The threshold filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, U> {
    /// input threshold.
    pub threshold: T,
    /// [off, on] outputs.
    pub outputs: GenericArray<U, U2>,
}

/// The threshold filter's state.
#[derive(Clone, Debug)]
pub struct State {}

/// A threshold filter.
#[derive(Clone, Debug)]
pub struct Threshold<T, U> {
    config: Config<T, U>,
    state: State,
}

impl<T, U> ConfigTrait for Threshold<T, U> {
    type Config = Config<T, U>;
}

impl<T, U> StateTrait for Threshold<T, U> {
    type State = State;
}

impl<T, U> WithConfig for Threshold<T, U> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = State {};
        Self { config, state }
    }
}

impl<T, U> ConfigRef for Threshold<T, U> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> StateMut for Threshold<T, U> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> Destruct for Threshold<T, U> {
    type Output = (Config<T, U>, State);

    fn destruct(self) -> Self::Output {
        (self.config, self.state)
    }
}

impl<T, U> Reset for Threshold<T, U> {
    fn reset(self) -> Self {
        self
    }
}

impl<T, U> Filter<T> for Threshold<T, U>
where
    T: PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let index = (input >= self.config.threshold) as usize;
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn test() {
        let filter = Threshold::with_config(Config {
            threshold: 10,
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
            vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]
        );
    }
}
