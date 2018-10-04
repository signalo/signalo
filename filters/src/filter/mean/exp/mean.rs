// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Exponential moving average filters.

use num_traits::Num;

use signalo_traits::filter::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigRef, Destruct, Reset, State as StateTrait, StateMut, WithConfig,
};

/// The mean filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// The inverse filter width.
    /// (`inverse_width = 1.0 / n` with `n` being the filter's width.)
    ///
    /// Important: `inverse_width` is required to be in the range between `0.0` and `1.0`.
    pub inverse_width: T,
}

/// A mean filter's internal state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// The current mean value.
    pub mean: Option<T>,
}

/// A mean filter producing the exponential moving average over a given signal.
#[derive(Clone, Debug)]
pub struct Mean<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> ConfigTrait for Mean<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for Mean<T> {
    type State = State<T>;
}

impl<T> WithConfig for Mean<T> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let mean = None;
            State { mean }
        };
        Self { config, state }
    }
}

impl<T> ConfigRef for Mean<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> StateMut for Mean<T> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> Destruct for Mean<T> {
    type Output = (Config<T>, State<T>);

    fn destruct(self) -> Self::Output {
        (self.config, self.state)
    }
}

impl<T> Reset for Mean<T> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

impl<T> Filter<T> for Mean<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let mean = match &self.state.mean {
            None => input,
            Some(ref state) => {
                state.clone() + ((input - state.clone()) * self.config.inverse_width.clone())
            }
        };
        self.state.mean = Some(mean.clone());
        mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.000, 0.250, 1.938, 1.953, 2.715, 4.036, 7.027, 6.020, 9.265, 8.449, 9.837, 9.628,
            9.471, 11.353, 12.765, 10.574, 10.930, 13.198, 14.898, 12.924, 11.443, 12.332, 12.999,
            12.249, 14.937, 13.703, 38.027, 33.020, 29.265, 26.449, 46.337, 36.003, 33.502, 28.376,
            24.532, 23.649, 22.987, 22.490, 25.368, 21.026, 43.019, 34.264, 32.948, 28.711, 25.533,
            23.150, 43.363, 35.272, 32.454, 30.340,
        ]
    }

    #[test]
    fn test() {
        let filter = Mean::with_config(Config {
            inverse_width: 0.25,
        });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
