// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Slope detection filters.

use std::cmp::{Ordering, PartialOrd};

use generic_array::typenum::*;
use generic_array::GenericArray;

use classify::Classification;

use signalo_traits::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigRef, Destruct, Reset, State as StateTrait, StateMut, WithConfig,
};

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

/// The slope detection filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<U> {
    /// [rising, flat, falling] outputs.
    pub outputs: GenericArray<U, U3>,
}

/// The slope detection filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Current memorized input.
    pub input: Option<T>,
}

/// A slope detection filter.
#[derive(Clone, Debug)]
pub struct Slopes<T, U> {
    config: Config<U>,
    state: State<T>,
}

impl<T, U> ConfigTrait for Slopes<T, U> {
    type Config = Config<U>;
}

impl<T, U> WithConfig for Slopes<T, U> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = State { input: None };
        Self { config, state }
    }
}

impl<T, U> StateTrait for Slopes<T, U> {
    type State = State<T>;
}

impl<T, U> ConfigRef for Slopes<T, U> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> StateMut for Slopes<T, U> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> Destruct for Slopes<T, U> {
    type Output = (Config<U>, State<T>);

    fn destruct(self) -> Self::Output {
        (self.config, self.state)
    }
}

impl<T, U> Reset for Slopes<T, U> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
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
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use classify::Classification;

    #[test]
    fn test() {
        use self::Slope::*;

        let filter = Slopes::with_config(Config {
            outputs: Slope::classes(),
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
            vec![
                None, Rising, Rising, Falling, Rising, Rising, Rising, Falling, Rising, Falling,
                Rising, Falling, None, Rising, None, Falling, Rising, Rising, None, Falling,
            ]
        );
    }
}
