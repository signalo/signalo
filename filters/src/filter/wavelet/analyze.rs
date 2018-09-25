// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Wavelet analysis (i.e. decomposition) filters.

use generic_array::ArrayLength;
use num_traits::Num;

use signalo_traits::filter::Filter;
use signalo_traits::{
    Config as ConfigTrait, InitialState, Resettable, Stateful, StatefulUnsafe, WithConfig,
};

use filter::convolve::{Config as ConvolveConfig, Convolve};

use filter::wavelet::Decomposition;

/// The wavelet filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, N>
where
    N: ArrayLength<T>,
{
    /// The low-pass convolution' configuration.
    pub low_pass: ConvolveConfig<T, N>,
    /// The high-pass convolution' configuration.
    pub high_pass: ConvolveConfig<T, N>,
}

/// The wavelet filter's configuration.
#[derive(Clone, Debug)]
pub struct ConfigRef<'a, T, N>
where
    N: ArrayLength<T>,
{
    /// The low-pass convolution' configuration.
    pub low_pass: &'a ConvolveConfig<T, N>,
    /// The high-pass convolution' configuration.
    pub high_pass: &'a ConvolveConfig<T, N>,
}

impl<'a, T, N> ConfigRef<'a, T, N>
where
    T: Clone,
    N: ArrayLength<T>,
    ConvolveConfig<T, N>: Clone,
{
    fn to_owned(self) -> Config<T, N> {
        Config {
            low_pass: self.low_pass.clone(),
            high_pass: self.high_pass.clone(),
        }
    }
}

/// A wavelet filter's internal state.
#[derive(Clone, Debug)]
pub struct State<T, N>
where
    N: ArrayLength<T>,
{
    low_pass: Convolve<T, N>,
    high_pass: Convolve<T, N>,
}

/// A wavelet filter.
#[derive(Clone, Debug)]
pub struct Analyze<T, N>
where
    N: ArrayLength<T>,
{
    state: State<T, N>,
}

impl<T, N> WithConfig for Analyze<T, N>
where
    T: Clone,
    N: ArrayLength<T>,
    ConvolveConfig<T, N>: Clone,
{
    type Config = Config<T, N>;

    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = Self::initial_state(&config);
        Self { state }
    }
}

impl<'a, T, N> ConfigTrait for &'a Analyze<T, N>
where
    N: ArrayLength<T>,
{
    type ConfigRef = ConfigRef<'a, T, N>;

    fn config(self) -> Self::ConfigRef {
        ConfigRef {
            low_pass: self.state.low_pass.config(),
            high_pass: self.state.high_pass.config(),
        }
    }
}

impl<T, N> Stateful for Analyze<T, N>
where
    N: ArrayLength<T>,
{
    type State = State<T, N>;
}

unsafe impl<T, N> StatefulUnsafe for Analyze<T, N>
where
    N: ArrayLength<T>,
{
    unsafe fn state(&self) -> &Self::State {
        &self.state
    }

    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<'a, T, N> InitialState<&'a Config<T, N>> for Analyze<T, N>
where
    T: Clone,
    N: ArrayLength<T>,
    ConvolveConfig<T, N>: Clone,
{
    fn initial_state(config: &'a Config<T, N>) -> Self::State {
        let low_pass = Convolve::with_config(config.low_pass.clone());
        let high_pass = Convolve::with_config(config.high_pass.clone());
        State {
            low_pass,
            high_pass,
        }
    }
}

impl<T, N> Resettable for Analyze<T, N>
where
    T: Clone,
    N: ArrayLength<T>,
    ConvolveConfig<T, N>: Clone,
{
    fn reset(&mut self) {
        let config = self.config().to_owned();
        self.state = Self::initial_state(&config);
    }
}

impl<T, N> Filter<T> for Analyze<T, N>
where
    T: Clone + Num,
    N: ArrayLength<T>,
{
    type Output = Decomposition<T>;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let low = self.state.low_pass.filter(input.clone());
        let high = self.state.high_pass.filter(input.clone());
        Decomposition { low, high }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0,
        ]
    }

    fn get_low() -> Vec<f32> {
        vec![
            0.0, 0.5, 4.0, 4.5, 3.5, 6.5, 12.0, 14.5, 16.0, 12.5, 10.0, 11.5, 9.0, 13.0, 17.0,
            10.5, 8.0, 16.0, 20.0, 13.5, 7.0, 11.0, 15.0, 12.5, 16.5, 16.5, 60.5, 145.5, 144.0,
            63.0, 62.0, 55.5, 15.5, 19.5, 13.0, 17.0, 21.0, 21.0, 27.5, 21.0, 58.5, 58.5, 18.5,
            22.5, 16.0, 16.0, 60.0, 57.5, 17.5, 24.0,
        ]
    }

    fn get_high() -> Vec<f32> {
        vec![
            0.0, 0.5, 3.0, -2.5, 1.5, 1.5, 4.0, -1.5, 3.0, -6.5, 4.0, -2.5, 0.0, 4.0, 0.0, -6.5,
            4.0, 4.0, 0.0, -6.5, 0.0, 4.0, 0.0, -2.5, 6.5, -6.5, 50.5, 34.5, -36.0, -45.0, 44.0,
            -50.5, 10.5, -6.5, 0.0, 4.0, 0.0, 0.0, 6.5, -13.0, 50.5, -50.5, 10.5, -6.5, 0.0, 0.0,
            44.0, -46.5, 6.5, 0.0,
        ]
    }

    #[test]
    fn low() {
        // Effectively calculates the haar transform:
        let filter = Analyze::with_config(Config {
            low_pass: ConvolveConfig {
                coefficients: arr![f32; 0.5, 0.5],
            },
            high_pass: ConvolveConfig {
                coefficients: arr![f32; 0.5, -0.5],
            },
        });
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).low))
            .collect();

        assert_nearly_eq!(output, get_low(), 0.001);
    }

    #[test]
    fn high() {
        // Effectively calculates the haar transform:
        let filter = Analyze::with_config(Config {
            low_pass: ConvolveConfig {
                coefficients: arr![f32; 0.5, 0.5],
            },
            high_pass: ConvolveConfig {
                coefficients: arr![f32; 0.5, -0.5],
            },
        });
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).high))
            .collect();

        assert_nearly_eq!(output, get_high(), 0.001);
    }
}
