// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Biquad (second-order IIR) filters using Direct Form II Transposed topology.
//!
//! Provides a core building block for designing more complex digital filters with
//! arbitrary frequency response characteristics. Can be cascaded for higher-order responses.
//!
//! A biquad filter is a second-order infinite impulse response (IIR) filter that processes
//! signals according to the difference equation:
//!
//! ```text
//! y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
//! ```
//!
//! This implementation uses the Direct Form II Transposed topology, which reduces
//! sensitivity to coefficient quantization and avoids overflow in intermediate calculations.
//!
//! A filter is stable if and only if all poles lie strictly inside the unit circle, i.e.,
//! the roots of `z^2 + a1*z + a2 = 0` have magnitude less than 1. Hand-crafted coefficients
//! that violate this condition will produce diverging (unstable) output.

use core::ops::{Add, Mul, Sub};

use num_traits::{Num, Zero};

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Cascaded biquad filters for higher-order IIR implementations.
pub mod cascade;

/// Coefficient factory traits for computing biquad filter coefficients from standard DSP design
/// equations (low-pass, high-pass, band-pass, notch, peaking, etc.).
#[cfg(any(feature = "libm", feature = "std"))]
pub mod coefficients;

/// The biquad filter's configuration.
///
/// Contains the five coefficients that define the biquad filter's frequency response.
/// The numerator coefficients (b0, b1, b2) control the zeros, while the denominator
/// coefficients (a1, a2) control the poles. The coefficient a0 is assumed to be 1.
///
/// `K` is the coefficient type. It may differ from the sample type used by
/// [`Biquad`], provided samples can be multiplied by coefficients.
#[derive(Clone, Debug)]
pub struct Config<K> {
    /// Numerator coefficient b0 (feedforward)
    pub b0: K,
    /// Numerator coefficient b1 (feedforward)
    pub b1: K,
    /// Numerator coefficient b2 (feedforward)
    pub b2: K,
    /// Denominator coefficient a1 (feedback)
    pub a1: K,
    /// Denominator coefficient a2 (feedback)
    pub a2: K,
}

impl<K> From<[K; 5]> for Config<K> {
    fn from([b0, b1, b2, a1, a2]: [K; 5]) -> Self {
        Self { b0, b1, b2, a1, a2 }
    }
}

impl<K> From<Config<K>> for [K; 5] {
    fn from(c: Config<K>) -> Self {
        [c.b0, c.b1, c.b2, c.a1, c.a2]
    }
}

impl<K> Default for Config<K>
where
    K: Num,
{
    fn default() -> Self {
        Self {
            b0: K::one(),
            b1: K::zero(),
            b2: K::zero(),
            a1: K::zero(),
            a2: K::zero(),
        }
    }
}

/// The biquad filter's state.
///
/// Contains the delay line values required for the DF2T implementation.
/// `T` is the sample type and therefore also the state type.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// First delay line value
    pub s1: T,
    /// Second delay line value
    pub s2: T,
}

impl<T> Default for State<T>
where
    T: Zero,
{
    fn default() -> Self {
        Self {
            s1: T::zero(),
            s2: T::zero(),
        }
    }
}

/// A biquad (second-order IIR) filter using Direct Form II Transposed topology.
///
/// Generic over sample/state type `T` and coefficient type `K`. `K` defaults
/// to `T`, preserving the common `Biquad<f32>`/`Biquad<f64>` usage. Use an
/// explicit `K` when coefficients have a different type than samples, for
/// example `Biquad<Complex32, f32>` with the `complex` feature enabled.
#[derive(Clone, Debug)]
pub struct Biquad<T, K = T> {
    config: Config<K>,
    state: State<T>,
}

impl<T, K> Default for Biquad<T, K>
where
    T: Zero,
    K: Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T, K> ConfigTrait for Biquad<T, K> {
    type Config = Config<K>;
}

impl<T, K> StateTrait for Biquad<T, K> {
    type State = State<T>;
}

impl<T, K> WithConfig for Biquad<T, K>
where
    T: Zero,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, K> ConfigRef for Biquad<T, K> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, K> ConfigClone for Biquad<T, K>
where
    K: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, K> StateMut for Biquad<T, K> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, K> HasGuts for Biquad<T, K> {
    type Guts = (Config<K>, State<T>);
}

impl<T, K> FromGuts for Biquad<T, K> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, K> IntoGuts for Biquad<T, K> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, K> Reset for Biquad<T, K>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, K> ResetMut for Biquad<T, K> where Self: Reset {}

impl<T, K> Filter<T> for Biquad<T, K>
where
    T: Clone + Zero + Add<Output = T> + Sub<Output = T> + Mul<K, Output = T>,
    K: Clone,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        df2t_step(&self.config, &mut self.state, input)
    }
}

pub(crate) fn df2t_step<T, K>(config: &Config<K>, state: &mut State<T>, input: T) -> T
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<K, Output = T>,
    K: Clone,
{
    let output = input.clone() * config.b0.clone() + state.s1.clone();
    state.s1 =
        input.clone() * config.b1.clone() - output.clone() * config.a1.clone() + state.s2.clone();
    state.s2 = input * config.b2.clone() - output.clone() * config.a2.clone();
    output
}

pub use cascade::BiquadCascade;

#[cfg(any(feature = "libm", feature = "std"))]
pub use coefficients::Butterworth;

#[cfg(test)]
mod tests;
