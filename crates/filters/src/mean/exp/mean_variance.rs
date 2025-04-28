// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Exponential moving average & variance filters.

use num_traits::{Num, Signed};

use signalo_traits::{
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, guts::{FromGuts, HasGuts, IntoGuts}, Reset,
    State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "derive")]
use signalo_traits::ResetMut;

use super::mean::{Config as MeanConfig, Mean};

/// Output of `MeanVariance` filter.
#[derive(Clone, Debug)]
pub struct Output<T> {
    /// Mean of values.
    pub mean: T,
    /// Variance of values.
    pub variance: T,
}

/// The mean/variance filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// The inverse filter width.
    /// (`inverse_width = 1.0 / n` with `n` being the filter's width.)
    ///
    /// Important: `inverse_width` is required to be in the range between `0.0` and `1.0`.
    pub inverse_width: T,
}

/// The mean/variance filter's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// The current mean value.
    pub mean: Mean<T>,
    /// The current variance value.
    pub variance: Mean<T>,
}

/// A mean/variance filter producing the exponential moving average and variance over a given signal.
#[derive(Clone, Debug)]
pub struct MeanVariance<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> ConfigTrait for MeanVariance<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for MeanVariance<T> {
    type State = State<T>;
}

impl<T> WithConfig for MeanVariance<T>
where
    T: Clone,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let mean = Mean::with_config(MeanConfig {
                inverse_width: config.inverse_width.clone(),
            });
            let variance = Mean::with_config(MeanConfig {
                inverse_width: config.inverse_width.clone(),
            });
            State { mean, variance }
        };
        Self { config, state }
    }
}

impl<T> ConfigRef for MeanVariance<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for MeanVariance<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for MeanVariance<T> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for MeanVariance<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for MeanVariance<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for MeanVariance<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for MeanVariance<T>
where
    T: Clone,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for MeanVariance<T> where Self: Reset {}

impl<T> Filter<T> for MeanVariance<T>
where
    T: Clone + Num + Signed,
{
    /// (mean, variance)
    type Output = Output<T>;

    fn filter(&mut self, input: T) -> Self::Output {
        let mean_old = unsafe {
            self.state
                .mean
                .state_mut()
                .mean
                .clone()
                .unwrap_or_else(|| input.clone())
        };
        let mean = self.state.mean.filter(input.clone());
        let deviation_old = (input.clone() - mean_old).abs();
        let deviation_new = (input - mean.clone()).abs();
        let squared = deviation_old * deviation_new;
        let variance = self.state.variance.filter(squared);
        Output { mean, variance }
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

    fn get_mean() -> Vec<f32> {
        vec![
            0.000, 0.250, 1.938, 1.953, 2.715, 4.036, 7.027, 6.020, 9.265, 8.449, 9.837, 9.628,
            9.471, 11.353, 12.765, 10.574, 10.930, 13.198, 14.898, 12.924, 11.443, 12.332, 12.999,
            12.249, 14.937, 13.703, 38.027, 33.020, 29.265, 26.449, 46.337, 36.003, 33.502, 28.376,
            24.532, 23.649, 22.987, 22.490, 25.368, 21.026, 43.019, 34.264, 32.948, 28.711, 25.533,
            23.150, 43.363, 35.272, 32.454, 30.340,
        ]
    }

    fn get_variance() -> Vec<f32> {
        vec![
            0.000, 0.188, 8.684, 6.513, 6.626, 10.207, 34.493, 28.910, 53.271, 41.953, 37.242,
            28.063, 21.121, 26.470, 25.832, 33.778, 25.715, 34.710, 34.709, 37.728, 34.875, 28.529,
            22.732, 18.735, 35.722, 31.362, 1798.539, 1424.107, 1110.382, 856.581, 1829.006,
            1_692.14, 1287.865, 1_044.71, 827.864, 623.237, 468.744, 352.298, 289.063, 273.354,
            1656.166, 1472.066, 1109.246, 885.793, 694.640, 538.022, 1629.149, 1418.237, 1087.501,
            829.026,
        ]
    }

    #[test]
    fn mean() {
        let filter = MeanVariance::with_config(Config {
            inverse_width: 0.25,
        });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).mean))
            .collect();
        assert_nearly_eq!(output, get_mean(), 0.001);
    }

    #[test]
    fn variance() {
        let filter = MeanVariance::with_config(Config {
            inverse_width: 0.25,
        });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).variance))
            .collect();
        assert_nearly_eq!(output, get_variance(), 0.001);
    }
}
