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

pub mod differentiator;
pub mod lagrange;
pub mod moving_sum;
pub mod windowed_sinc;

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
///
/// # Coefficient ordering
///
/// Coefficients `h[k]` pair with taps `x[n−k]` so that `h[0]` multiplies the
/// newest sample and `h[N−1]` the oldest. The dot product computes
/// `y[n] = Σ_{k=0}^{N−1} h[k]·x[n−k]` using zero-padding for negative
/// time indices. This convention is verified by the `coefficient_ordering`
/// test.
///
/// # Complexity
///
/// - **Time per sample:** O(N); dot product of N taps with N coefficients.
/// - **Space:** O(N); circular tap buffer of N elements plus N coefficient array.
///
/// # Cold-start behaviour
///
/// On construction, the tap buffer is pre-filled with `N` zeros. The first
/// `N − 1` outputs therefore reflect implicit zero-padding `x[n] = 0` for
/// `n < 0`, as verified by `cold_start_is_zero_padded_partial_convolution`.
/// Discard the warm-up window if zero-padding bias is unacceptable
/// for your application.
#[derive(Clone, Debug)]
pub struct Convolve<T, const N: usize> {
    config: Config<T, N>,
    state: State<T, N>,
}

#[cfg(any(feature = "libm", feature = "std"))]
impl<T, const N: usize> Convolve<T, N>
where
    T: num_traits::Float,
{
    /// Creates a new `Convolve` filter with given `coefficients`, normalizing
    /// them to unity DC gain.
    ///
    /// This constructor is float-only. For integer types, use
    /// [`with_config`](Self::with_config) directly with manually pre-scaled
    /// coefficients.
    ///
    /// # Behaviour
    ///
    /// If `sum == 0` (exact), normalisation is skipped — this is the documented
    /// DC-blocker escape hatch. Otherwise the sum must be finite and its
    /// magnitude must be at or above `T::min_positive_value().sqrt()`; smaller
    /// denominators (near-zero) panic.
    pub fn normalized(mut config: Config<T, N>) -> Self
    where
        T: core::fmt::Debug,
    {
        let sum = config
            .coefficients
            .iter()
            .copied()
            .fold(T::zero(), |a, b| a + b);
        if !sum.is_zero() {
            // Exact zero is treated as an explicit DC-blocker request; near-zero is treated as numerical error and rejected by safe_normalise_divisor.
            let denom =
                crate::math::safe_normalise_divisor(sum, "Convolve::normalized: coefficient sum");
            for coeff in &mut config.coefficients {
                *coeff = *coeff / denom;
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

impl<T, const N: usize> WithConfig for Convolve<T, N>
where
    T: Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        assert!(N > 0, "Convolve: window size N must be > 0");
        let state = {
            let mut taps = CircularBuffer::new();
            for _ in 0..N {
                let _ = taps.push_back(T::zero());
            }
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
    fn state_mut(&mut self) -> &mut Self::State {
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

impl<T, const N: usize> Reset for Convolve<T, N>
where
    T: Num,
{
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
        self.state.taps.push_back(input);

        let state_iter = self.state.taps.iter();
        // See "Coefficient ordering" in the struct-level documentation.
        // coeff_iter.rev(): state iterates oldest->newest; reversing pairs h[N-1] with oldest, h[0] with newest. See struct-level "Coefficient ordering".
        let coeff_iter = self.config.coefficients.iter().rev();

        state_iter
            .zip(coeff_iter)
            .fold(T::zero(), |sum, (state, coeff)| {
                sum + (state.clone() * coeff.clone())
            })
    }
}

#[cfg(test)]
mod tests;
