// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Wavelet synthesis (i.e. reconstruction) filters.

use generic_array::ArrayLength;

use num_traits::Num;

use signalo_traits::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigClone, FromGuts, Guts, IntoGuts, Reset, State as StateTrait,
    StateMut, WithConfig,
};

use convolve::{Config as ConvolveConfig, Convolve};
use wavelet::Decomposition;

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

/// A wavelet filter's internal state.
#[derive(Clone, Debug)]
pub struct State<T, N>
where
    N: ArrayLength<T>,
{
    /// Low-pass convolution.
    pub low_pass: Convolve<T, N>,
    /// Low-pass convolution.
    pub high_pass: Convolve<T, N>,
}

/// A wavelet filter.
#[derive(Clone, Debug)]
pub struct Synthesize<T, N>
where
    N: ArrayLength<T>,
{
    state: State<T, N>,
}

impl<T, N> ConfigTrait for Synthesize<T, N>
where
    N: ArrayLength<T>,
{
    type Config = Config<T, N>;
}

impl<T, N> StateTrait for Synthesize<T, N>
where
    N: ArrayLength<T>,
{
    type State = State<T, N>;
}

impl<T, N> WithConfig for Synthesize<T, N>
where
    N: ArrayLength<T>,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let low_pass = Convolve::with_config(config.low_pass);
            let high_pass = Convolve::with_config(config.high_pass);
            State {
                low_pass,
                high_pass,
            }
        };
        Self { state }
    }
}

impl<T, N> ConfigClone for Synthesize<T, N>
where
    N: ArrayLength<T>,
    Convolve<T, N>: ConfigClone<Config = ConvolveConfig<T, N>>,
{
    fn config(&self) -> Self::Config {
        let low_pass = self.state.low_pass.config();
        let high_pass = self.state.high_pass.config();
        Config {
            low_pass,
            high_pass,
        }
    }
}

impl<T, N> StateMut for Synthesize<T, N>
where
    N: ArrayLength<T>,
{
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, N> Guts for Synthesize<T, N>
where
    N: ArrayLength<T>,
{
    type Guts = State<T, N>;
}

impl<T, N> FromGuts for Synthesize<T, N>
where
    N: ArrayLength<T>,
{
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, N> IntoGuts for Synthesize<T, N>
where
    N: ArrayLength<T>,
{
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, N> Reset for Synthesize<T, N>
where
    N: ArrayLength<T>,
    Self: ConfigClone<Config = Config<T, N>> + WithConfig<Output = Self>,
{
    fn reset(self) -> Self {
        Self::with_config(self.config())
    }
}

#[cfg(feature = "derive_reset_mut")]
impl<T, N> ResetMut for Synthesize<T, N> where Self: Reset {}

impl<T, N> Filter<Decomposition<T>> for Synthesize<T, N>
where
    T: Clone + Num,
    N: ArrayLength<T>,
{
    type Output = T;

    fn filter(&mut self, input: Decomposition<T>) -> Self::Output {
        let Decomposition { low, high } = input;
        let low = self.state.low_pass.filter(low);
        let high = self.state.high_pass.filter(high);
        low + high
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<Decomposition<f32>> {
        get_low()
            .into_iter()
            .zip(get_high())
            .map(|(low, high)| Decomposition { low, high })
            .collect()
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

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0,
            4.0, 12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0,
            18.0, 106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0,
            16.0, 16.0, 16.0, 104.0, 11.0, 24.0,
        ]
    }

    #[test]
    fn test() {
        // Effectively calculates the haar transform:
        let filter = Synthesize::with_config(Config {
            low_pass: ConvolveConfig {
                coefficients: arr![f32; 0.5, 0.5],
            },
            high_pass: ConvolveConfig {
                coefficients: arr![f32; -0.5, 0.5],
            },
        });
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, input| Some(filter.filter(input.clone())))
            .collect();

        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
