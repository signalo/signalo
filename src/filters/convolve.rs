// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Convolution filters.

use circular_buffer::CircularBuffer;
use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Savitzky-Golay polynomial smoothing filter.
///
/// Smooths signals by fitting local polynomial patches, preserving signal features like edges
/// and peaks while reducing noise. Provides excellent results for spectral data and derivatives.
pub mod savitzky_golay;

/// The convolution filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, const N: usize> {
    /// The convolution coefficients.
    pub coefficients: [T; N],
}

/// The convolution filter's state.
#[derive(Clone, Debug)]
pub struct State<T, const N: usize> {
    /// The filter's taps (i.e. buffered input).
    pub taps: CircularBuffer<N, T>,
}

/// A convolution filter.
#[derive(Clone, Debug)]
pub struct Convolve<T, const N: usize> {
    config: Config<T, N>,
    state: State<T, N>,
}

impl<T, const N: usize> Convolve<T, N>
where
    T: Clone + PartialOrd + Num,
{
    /// Creates a new `Convolve` filter with given `coefficients`, normalizing them.
    pub fn normalized(mut config: Config<T, N>) -> Self {
        let mut sum = T::zero();
        for coeff in &config.coefficients {
            sum = sum + coeff.clone();
        }
        if !sum.is_zero() {
            for coeff in &mut config.coefficients {
                *coeff = coeff.clone() / sum.clone();
            }
        }
        Self::with_config(config)
    }
}

impl<T, const N: usize> ConfigTrait for Convolve<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for Convolve<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> WithConfig for Convolve<T, N> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let taps = CircularBuffer::default();
            State { taps }
        };
        Self { config, state }
    }
}

impl<T, const N: usize> ConfigRef for Convolve<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> ConfigClone for Convolve<T, N>
where
    Config<T, N>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> StateMut for Convolve<T, N> {
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Convolve<T, N> {
    type Guts = (Config<T, N>, State<T, N>);
}

impl<T, const N: usize> FromGuts for Convolve<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for Convolve<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for Convolve<T, N> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Convolve<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Convolve<T, N>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        // Note: push_back may return None until the circular buffer is full.
        // Once full, it returns Some with the evicted old value.
        // This loop is guaranteed to terminate when the buffer is full.
        loop {
            if self.state.taps.push_back(input.clone()).is_some() {
                break;
            }
        }

        let state_iter = self.state.taps.iter();
        let coeff_iter = self.config.coefficients.iter().rev();

        state_iter
            .zip(coeff_iter)
            .fold(T::zero(), |sum, (state, coeff)| {
                sum + (state.clone() * coeff.clone())
            })
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use nearly_eq::assert_nearly_eq;

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
            coefficients: [1.000, -1.000],
        });
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }

    #[test]
    fn test_normalized() {
        // Test normalizing coefficients
        let filter = Convolve::<f32, 3>::normalized(Config {
            coefficients: [2.0, 4.0, 6.0],
        });
        let config = filter.config_ref();
        // Sum is 12.0, so normalized coefficients should be [2/12, 4/12, 6/12] = [1/6, 1/3, 1/2]
        assert_nearly_eq!(config.coefficients[0], 1.0 / 6.0, 0.0001);
        assert_nearly_eq!(config.coefficients[1], 1.0 / 3.0, 0.0001);
        assert_nearly_eq!(config.coefficients[2], 1.0 / 2.0, 0.0001);
    }

    #[test]
    fn test_normalized_zero_sum() {
        // Test normalizing coefficients that sum to zero
        let filter = Convolve::<f32, 3>::normalized(Config {
            coefficients: [1.0, -1.0, 0.0],
        });
        let config = filter.config_ref();
        // Sum is 0.0, so coefficients should remain unchanged
        assert_nearly_eq!(config.coefficients[0], 1.0, 0.0001);
        assert_nearly_eq!(config.coefficients[1], -1.0, 0.0001);
        assert_nearly_eq!(config.coefficients[2], 0.0, 0.0001);
    }

    #[test]
    fn test_config_ref() {
        let config = Config {
            coefficients: [0.5, 0.25, 0.25],
        };
        let filter = Convolve::<f32, 3>::with_config(config.clone());
        let config_ref = filter.config_ref();
        assert_nearly_eq!(config_ref.coefficients[0], 0.5, 0.0001);
        assert_nearly_eq!(config_ref.coefficients[1], 0.25, 0.0001);
        assert_nearly_eq!(config_ref.coefficients[2], 0.25, 0.0001);
    }

    #[test]
    fn test_config_clone() {
        let config = Config {
            coefficients: [0.5, 0.25, 0.25],
        };
        let filter = Convolve::<f32, 3>::with_config(config.clone());
        let cloned_config = filter.config();
        assert_nearly_eq!(cloned_config.coefficients[0], 0.5, 0.0001);
        assert_nearly_eq!(cloned_config.coefficients[1], 0.25, 0.0001);
        assert_nearly_eq!(cloned_config.coefficients[2], 0.25, 0.0001);
    }

    #[test]
    fn test_state_mut() {
        use circular_buffer::CircularBuffer;

        let config = Config {
            coefficients: [1.0, 0.5],
        };
        let mut filter = Convolve::<f32, 2>::with_config(config);
        filter.filter(1.0);
        filter.filter(2.0);

        unsafe {
            let state = filter.state_mut();
            // Modify the internal taps buffer
            state.taps = CircularBuffer::from([3.0, 4.0]);
        }

        // Next filter call should use the modified state
        let output = filter.filter(5.0);
        // Output should be: 5.0 * 1.0 + 4.0 * 0.5 = 5.0 + 2.0 = 7.0
        assert_nearly_eq!(output, 7.0, 0.0001);
    }

    #[test]
    fn test_from_into_guts() {
        use crate::traits::guts::{FromGuts, IntoGuts};

        let config = Config {
            coefficients: [1.0, 0.5],
        };
        let mut filter = Convolve::<f32, 2>::with_config(config.clone());
        filter.filter(3.0);
        filter.filter(4.0);

        let (guts_config, guts_state) = filter.into_guts();
        assert_nearly_eq!(guts_config.coefficients[0], 1.0, 0.0001);
        assert_nearly_eq!(guts_config.coefficients[1], 0.5, 0.0001);

        let filter2 = Convolve::from_guts((guts_config, guts_state));
        let output = filter2.config_ref();
        assert_nearly_eq!(output.coefficients[0], 1.0, 0.0001);
    }

    #[test]
    fn test_reset() {
        let config = Config {
            coefficients: [1.0, 1.0, 1.0],
        };
        let mut filter = Convolve::<f32, 3>::with_config(config);

        // Fill the buffer
        filter.filter(1.0);
        filter.filter(2.0);
        filter.filter(3.0);
        let output = filter.filter(4.0);
        assert_nearly_eq!(output, 9.0, 0.0001); // 2 + 3 + 4

        // Reset the filter
        let mut reset_filter = filter.reset();

        // After reset, buffer should be empty and fill again
        reset_filter.filter(10.0);
        reset_filter.filter(20.0);
        reset_filter.filter(30.0);
        let output = reset_filter.filter(40.0);
        assert_nearly_eq!(output, 90.0, 0.0001); // 20 + 30 + 40
    }

    #[test]
    fn test_filter_buffer_filling() {
        // Test the loop that fills the circular buffer initially
        let config = Config {
            coefficients: [0.25, 0.25, 0.25, 0.25],
        };
        let mut filter = Convolve::<f32, 4>::with_config(config);

        // First input - buffer not full yet, loops until full
        let output1 = filter.filter(4.0);
        // All positions get 4.0, output = 4 * 4.0 * 0.25 = 4.0
        assert_nearly_eq!(output1, 4.0, 0.0001);

        // Second input - buffer now full
        let output2 = filter.filter(8.0);
        // Buffer: [4, 4, 4, 8], output = 4*0.25 + 4*0.25 + 4*0.25 + 8*0.25 = 5.0
        assert_nearly_eq!(output2, 5.0, 0.0001);
    }
}
