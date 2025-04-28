// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Approximated exponential moving median filters.

use num_traits::Num;

use signalo_traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

use crate::mean::exp::mean::{Config as MeanConfig, Mean};

/// The median filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Pre-processing mean smoothing factor.
    ///
    /// Import: value required to be in the range between `0.0` and `1.0`.
    pub pre: MeanConfig<T>,
    /// Pre-processing median smoothing factor.
    ///
    /// Recommended value: `mid = pre`.
    /// Import: value required to be in the range between `0.0` and `1.0`.
    pub mid: T,
    /// Post-processing mean smoothing factor.
    ///
    /// Recommended value: `post = pre * 0.5`.
    /// Import: value required to be in the range between `0.0` and `1.0`.
    pub post: MeanConfig<T>,
}

/// The median filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Pre-processing low-pass filter.
    pub mean_pre: Mean<T>,
    /// Post-processing low-pass filter.
    pub mean_post: Mean<T>,
    /// Current median value.
    pub median: Option<T>,
}

/// A median filter producing the approximated exponential moving median over a given signal.
#[derive(Clone, Debug)]
pub struct Median<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> ConfigTrait for Median<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for Median<T> {
    type State = State<T>;
}

impl<T> WithConfig for Median<T>
where
    T: Clone,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let mean_pre = Mean::with_config(config.pre.clone());
            let mean_post = Mean::with_config(config.post.clone());
            let median = None;
            State {
                mean_pre,
                mean_post,
                median,
            }
        };
        Self { config, state }
    }
}

impl<T> ConfigRef for Median<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for Median<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for Median<T> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for Median<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for Median<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for Median<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for Median<T>
where
    T: Clone,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

impl<T> Filter<T> for Median<T>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        // We calculate the mean and use it as an estimate of the median:
        let mean = self.state.mean_pre.filter(input);
        // We then calculate the approximate of the median:
        let median = match &self.state.median {
            None => mean,
            Some(ref state) => state.clone() + ((mean - state.clone()) * self.config.mid.clone()),
        };
        // The approximated median tends to oscillate,
        // so we apply another mean to smoothen those out:
        let state = self.state.mean_post.filter(median);
        // And we're done. Store a copy and return the result:
        self.state.median = Some(state.clone());
        state
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use nearly_eq::assert_nearly_eq;

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
            0.000, 0.063, 0.523, 0.817, 1.207, 1.803, 2.950, 3.456, 4.648, 5.254, 6.066, 6.605,
            6.990, 7.784, 8.708, 8.817, 9.064, 9.856, 10.836, 11.025, 10.856, 11.042, 11.370,
            11.428, 12.177, 12.368, 18.616, 21.311, 22.284, 22.441, 27.733, 28.627, 28.854, 27.962,
            26.637, 25.705, 25.003, 24.446, 24.799, 23.904, 28.831, 29.684, 30.015, 29.284, 28.133,
            26.872, 31.140, 31.749, 31.531, 30.965,
        ]
    }

    #[test]
    fn test() {
        let filter = Median::with_config(Config {
            pre: MeanConfig { inverse_width: 0.5 },
            mid: 0.5,
            post: MeanConfig {
                inverse_width: 0.25,
            }, // 0.25,
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
