// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Convolution filters.

use arraydeque::{ArrayDeque, Wrapping};
use generic_array::{ArrayLength, GenericArray};

use num_traits::Num;

use signalo_traits::Filter;

use signalo_traits::{
    Config as ConfigTrait, ConfigClone, ConfigRef, FromGuts, Guts, IntoGuts, Reset,
    State as StateTrait, StateMut, WithConfig,
};

pub mod savitzky_golay;

/// The convolution filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, N>
where
    N: ArrayLength<T>,
{
    /// The convolution coefficients.
    pub coefficients: GenericArray<T, N>,
}

/// The convolution filter's state.
#[derive(Clone, Debug)]
pub struct State<T, N>
where
    N: ArrayLength<T>,
{
    /// The filter's taps (i.e. buffered input).
    pub taps: ArrayDeque<GenericArray<T, N>, Wrapping>,
}

/// A convolution filter.
#[derive(Clone, Debug)]
pub struct Convolve<T, N>
where
    N: ArrayLength<T>,
{
    config: Config<T, N>,
    state: State<T, N>,
}

impl<T, N> Convolve<T, N>
where
    T: Clone + PartialOrd + Num,
    N: ArrayLength<T>,
{
    /// Creates a new `Convolve` filter with given `coefficients`, normalizing them.
    pub fn normalized(mut config: Config<T, N>) -> Self {
        // let mut coefficients: GenericArray<T, N>
        let sum = config
            .coefficients
            .as_slice()
            .iter()
            .fold(T::zero(), |sum, coeff| sum + coeff.clone());
        if !sum.is_zero() {
            for coeff in config.coefficients.as_mut_slice() {
                *coeff = coeff.clone() / sum.clone();
            }
        }
        Self::with_config(config)
    }
}

impl<T, N> ConfigTrait for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    type Config = Config<T, N>;
}

impl<T, N> StateTrait for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    type State = State<T, N>;
}

impl<T, N> WithConfig for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let taps = ArrayDeque::new();
            State { taps }
        };
        Self { config, state }
    }
}

impl<T, N> ConfigRef for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, N> ConfigClone for Convolve<T, N>
where
    N: ArrayLength<T>,
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, N> StateMut for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, N> Guts for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    type Guts = (Config<T, N>, State<T, N>);
}

impl<T, N> FromGuts for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    unsafe fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, N> IntoGuts for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, N> Reset for Convolve<T, N>
where
    N: ArrayLength<T>,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

impl<T, N> Filter<T> for Convolve<T, N>
where
    T: Clone + Num,
    N: ArrayLength<T>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        loop {
            if self.state.taps.push_back(input.clone()).is_some() {
                break;
            }
        }

        let state_iter = self.state.taps.iter();
        let coeff_iter = self.config.coefficients.as_slice().iter().rev();

        state_iter
            .zip(coeff_iter)
            .fold(T::zero(), |sum, (state, coeff)| {
                sum + (state.clone() * coeff.clone())
            })
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

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 1.0, 6.0, -5.0, 3.0, 3.0, 8.0, -3.0, 6.0, -13.0, 8.0, -5.0, 0.0, 8.0, 0.0, -13.0,
            8.0, 8.0, 0.0, -13.0, 0.0, 8.0, 0.0, -5.0, 13.0, -13.0, 101.0, 69.0, -72.0, -90.0,
            88.0, -101.0, 21.0, -13.0, 0.0, 8.0, 0.0, 0.0, 13.0, -26.0, 101.0, -101.0, 21.0, -13.0,
            0.0, 0.0, 88.0, -93.0, 13.0, 0.0,
        ]
    }

    #[test]
    fn test() {
        // Effectively calculates the derivative:
        let filter = Convolve::with_config(Config {
            coefficients: arr![f32; 1.000, -1.000],
        });
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
