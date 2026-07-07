// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Wavelet analysis (decomposition) filter.
//!
//! Decomposes signals into low-frequency (approximation) and high-frequency (detail) components
//! using wavelet transforms for multi-resolution analysis.

use core::marker::PhantomData;

use circular_buffer::FixedCircularBuffer;
use num_traits::Num;

use crate::storage::{AsSlice, RingBuffer};
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, Filter, Reset, State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

use super::Decomposition;
use crate::filters::fir::convolve::{Config as ConvolveConfig, Convolve, ConvolveArray};

/// The wavelet filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, C> {
    /// The low-pass convolution' configuration.
    pub low_pass: ConvolveConfig<C>,
    /// The high-pass convolution' configuration.
    pub high_pass: ConvolveConfig<C>,
    /// Ensures `T` is captured in the type signature.
    pub _phantom: PhantomData<T>,
}

/// A wavelet filter's internal state.
#[derive(Clone, Debug)]
pub struct State<T, C, R> {
    /// Low-pass convolution.
    pub low_pass: Convolve<T, C, R>,
    /// High-pass convolution.
    pub high_pass: Convolve<T, C, R>,
}

/// A wavelet filter.
///
/// # Complexity
///
/// - **Time per sample:** O(N); two independent `Convolve<T, C, R>` calls (low-pass and high-pass),
///   each O(N).
/// - **Space:** O(N); two tap buffers of N elements each.
#[derive(Clone, Debug)]
pub struct Analyze<T, C, R> {
    state: State<T, C, R>,
}

/// A wavelet analysis filter backed by heap-allocated [`Vec`](alloc::vec::Vec) coefficients
/// and a [`HeapCircularBuffer`] tap buffer.
///
/// Requires the `alloc` feature.
#[cfg(feature = "alloc")]
pub type AnalyzeVec<T> = Analyze<T, alloc::vec::Vec<T>, HeapCircularBuffer<T>>;

/// A wavelet analysis filter that borrows a [`CircularBuffer`] tap buffer.
///
/// This alias allows sharing a caller-owned ring buffer without taking
/// ownership of it. The coefficient storage `C` remains generic.
pub type AnalyzeRefMut<'a, T, C> = Analyze<T, C, &'a mut circular_buffer::CircularBuffer<T>>;

impl<T, C, R> ConfigTrait for Analyze<T, C, R> {
    type Config = Config<T, C>;
}

impl<T, C, R> StateTrait for Analyze<T, C, R> {
    type State = State<T, C, R>;
}

impl<T, const N: usize> WithConfig for AnalyzeArray<T, N>
where
    T: Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let low_pass = ConvolveArray::with_config(config.low_pass);
            let high_pass = ConvolveArray::with_config(config.high_pass);
            State {
                low_pass,
                high_pass,
            }
        };
        Self { state }
    }
}

impl<T, const N: usize> ConfigClone for AnalyzeArray<T, N>
where
    ConvolveArray<T, N>: ConfigClone<Config = ConvolveConfig<[T; N]>>,
{
    fn config(&self) -> Self::Config {
        let low_pass = self.state.low_pass.config();
        let high_pass = self.state.high_pass.config();
        Config {
            low_pass,
            high_pass,
            _phantom: PhantomData,
        }
    }
}

impl<T, C, R> StateMut for Analyze<T, C, R> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, C, R> HasGuts for Analyze<T, C, R> {
    type Guts = State<T, C, R>;
}

impl<T, C, R> FromGuts for Analyze<T, C, R> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, C, R> IntoGuts for Analyze<T, C, R> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for AnalyzeArray<T, N>
where
    T: Num,
    Self: ConfigClone<Config = Config<T, [T; N]>> + WithConfig<Output = Self>,
{
    fn reset(self) -> Self {
        Self::with_config(self.config())
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for AnalyzeArray<T, N> where Self: Reset {}

impl<T, C, R> Analyze<T, C, R>
where
    C: AsSlice<T>,
    R: RingBuffer<T>,
{
    /// Creates an [`Analyze`] filter from an already-constructed [`State`].
    pub fn from_parts(state: State<T, C, R>) -> Self {
        Self { state }
    }
}

impl<T, C, R> Filter<T> for Analyze<T, C, R>
where
    T: Clone + Num,
    C: AsSlice<T>,
    R: RingBuffer<T>,
{
    type Output = Decomposition<T>;

    fn filter(&mut self, input: T) -> Self::Output {
        let low = self.state.low_pass.filter(input.clone());
        let high = self.state.high_pass.filter(input);
        Decomposition { low, high }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::marker::PhantomData;

    use approx::assert_abs_diff_eq;

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
        let filter = AnalyzeArray::with_config(Config {
            low_pass: ConvolveConfig {
                coefficients: [0.5, 0.5],
            },
            high_pass: ConvolveConfig {
                coefficients: [0.5, -0.5],
            },
            _phantom: PhantomData,
        });
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).low))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), get_low().as_slice(), epsilon = 0.001);
    }

    #[test]
    fn high() {
        // Effectively calculates the haar transform:
        let filter = AnalyzeArray::with_config(Config {
            low_pass: ConvolveConfig {
                coefficients: [0.5, 0.5],
            },
            high_pass: ConvolveConfig {
                coefficients: [0.5, -0.5],
            },
            _phantom: PhantomData,
        });
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input).high))
            .collect();

        assert_abs_diff_eq!(output.as_slice(), get_high().as_slice(), epsilon = 0.001);
    }
}
