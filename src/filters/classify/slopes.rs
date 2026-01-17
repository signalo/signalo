// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Slope detection filters.

use core::cmp::{Ordering, PartialOrd};

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

use super::Classification;

/// A slope's kind.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum Slope {
    /// A rising slope.
    Rising,
    /// A flat slope.
    #[default]
    None,
    /// A falling slope.
    Falling,
}

impl Classification<Slope, 3> for Slope {
    fn classes() -> [Self; 3] {
        [Self::Rising, Self::None, Self::Falling]
    }
}

/// The slope detection filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<U> {
    /// [rising, flat, falling] outputs.
    pub outputs: [U; 3],
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

impl<T, U> ConfigClone for Slopes<T, U>
where
    Config<U>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, U> StateMut for Slopes<T, U> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> HasGuts for Slopes<T, U> {
    type Guts = (Config<U>, State<T>);
}

impl<T, U> FromGuts for Slopes<T, U> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, U> IntoGuts for Slopes<T, U> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, U> Reset for Slopes<T, U> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, U> ResetMut for Slopes<T, U> where Self: Reset {}

impl<T, U> Filter<T> for Slopes<T, U>
where
    T: Clone + PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        let index = match self.state.input {
            None => 1, // None
            Some(ref state) => {
                #[allow(clippy::match_same_arms)]
                match state.partial_cmp(&input) {
                    Some(Ordering::Less) => 0,    // Rising
                    Some(Ordering::Equal) => 1,   // None
                    Some(Ordering::Greater) => 2, // Falling
                    None => 1,                    // Non-fatal fallback.
                }
            }
        };
        self.state.input = Some(input);
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use super::Classification;

    use super::*;

    #[test]
    fn test() {
        use self::Slope::*;

        let filter = Slopes::with_config(Config {
            outputs: Slope::classes(),
        });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = [
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
