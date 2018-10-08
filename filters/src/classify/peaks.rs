// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Peak detection filters.

use std::cmp::PartialOrd;

use generic_array::typenum::*;
use generic_array::GenericArray;

use signalo_traits::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigClone, ConfigRef, Destruct, Reset, State as StateTrait, StateMut,
    WithConfig,
};

use classify::{
    slopes::{Config as SlopesConfig, Slope, Slopes, State as SlopesState},
    Classification,
};

/// A slope's kind.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Peak {
    /// A local maximum.
    Max,
    /// A local constant.
    None,
    /// A local minimum.
    Min,
}

impl Default for Peak {
    fn default() -> Self {
        Peak::None
    }
}

impl Classification<Peak, U3> for Peak {
    fn classes() -> GenericArray<Peak, U3> {
        arr![Peak; Peak::Max, Peak::None, Peak::Min]
    }
}

/// The peak detection filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<U> {
    /// [rising, flat, falling] outputs.
    pub outputs: GenericArray<U, U3>,
}

/// A peak detection filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Slope detection filter.
    pub slopes: Slopes<T, Slope>,
    /// Current slope.
    pub slope: Option<Slope>,
}

/// A peak detection filter.
#[derive(Clone, Debug)]
pub struct Peaks<T, U> {
    config: Config<U>,
    state: State<T>,
}

impl<T, U> Peaks<T, U>
where
    U: Clone,
{
    fn filter_internal(&mut self, slope: Slope) -> (Slope, usize) {
        let index = match self.state.slope {
            None => 1,
            Some(Slope::Rising) => {
                match &slope {
                    Slope::Rising => 1,  // None
                    Slope::None => 1,    // None
                    Slope::Falling => 0, // Max
                }
            }
            Some(Slope::None) => {
                match &slope {
                    Slope::Rising => 1,  // None
                    Slope::None => 1,    // None
                    Slope::Falling => 1, // None
                }
            }
            Some(Slope::Falling) => {
                match &slope {
                    Slope::Rising => 2,  // Min
                    Slope::None => 1,    // None
                    Slope::Falling => 1, // None
                }
            }
        };
        (slope, index)
    }
}

impl<T, U> ConfigTrait for Peaks<T, U> {
    type Config = Config<U>;
}

impl<T, U> StateTrait for Peaks<T, U> {
    type State = State<T>;
}

impl<T, U> WithConfig for Peaks<T, U> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let slopes = Slopes::with_config(SlopesConfig {
                outputs: Slope::classes(),
            });
            let slope = None;
            State { slopes, slope }
        };
        Self { config, state }
    }
}

impl<T, U> ConfigRef for Peaks<T, U> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> ConfigClone for Peaks<T, U>
where
    Config<U>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, U> StateMut for Peaks<T, U> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> Destruct for Peaks<T, U> {
    type Output = (Config<U>, State<T>);

    fn destruct(self) -> Self::Output {
        (self.config, self.state)
    }
}

impl<T, U> Reset for Peaks<T, U> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

impl<T, U> Filter<T> for Peaks<T, U>
where
    T: Clone + PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let slope = self.state.slopes.filter(input);
        let (state, index) = self.filter_internal(slope);
        self.state.slope = Some(state);
        self.config.outputs[index].clone()
    }
}

impl<U> Filter<Slope> for Peaks<Slope, U>
where
    U: Clone,
{
    type Output = U;

    #[inline]
    fn filter(&mut self, slope: Slope) -> Self::Output {
        let (state, index) = self.filter_internal(slope);
        unsafe {
            let inner_state = self.state.slopes.state_mut();
            *inner_state = SlopesState { input: Some(slope) };
        }
        self.state.slope = Some(state);
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use classify::Classification;

    #[test]
    fn values() {
        use self::Peak::*;

        let filter = Peaks::with_config(Config {
            outputs: Peak::classes(),
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
                None, None, None, Max, Min, None, None, Max, Min, Max, Min, Max, None, None, None,
                None, Min, None, None, None,
            ]
        );
    }

    #[test]
    fn slopes() {
        use self::Peak::*;

        let filter = Peaks::with_config(Config {
            outputs: Peak::classes(),
        });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = {
            use self::Slope::*;
            vec![
                None, Rising, Rising, Falling, Rising, Rising, Rising, Falling, Rising, Falling,
                Rising, Falling, None, Rising, None, Falling, Rising, Rising, None, Falling,
            ]
        };
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, input| Some(filter.filter(input.clone())))
            .collect();
        assert_eq!(
            output,
            vec![
                None, None, None, Max, Min, None, None, Max, Min, Max, Min, Max, None, None, None,
                None, Min, None, None, None,
            ]
        );
    }
}
