// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Envelope follower filter with asymmetric attack and release.
//!
//! Tracks the peak amplitude of a signal with fast attack and slow release characteristics,
//! useful for dynamic range compression, peak detection, and amplitude modulation.

use num_traits::{Num, Signed};

use crate::time::Timed;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The envelope follower's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// The attack smoothing coefficient (0..1).
    /// Higher values make the envelope respond faster to increasing input magnitudes.
    pub attack: T,
    /// The release smoothing coefficient (0..1).
    /// Higher values make the envelope respond faster to decreasing input magnitudes.
    pub release: T,
}

impl<T> Default for Config<T>
where
    T: Num,
{
    fn default() -> Self {
        Self {
            attack: T::one(),
            release: T::zero(),
        }
    }
}

/// The envelope follower's internal state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// The current envelope value.
    pub envelope: T,
}

/// An envelope follower filter.
///
/// Tracks the envelope (peak amplitude) of a signal with asymmetric attack and release times.
/// The envelope rises quickly on attack and falls more slowly on release.
///
/// The filter computes:
/// - `abs_input = |input|`
/// - if `abs_input > envelope`:
///   - `envelope = attack * abs_input + (1 - attack) * envelope`
/// - else:
///   - `envelope = release * abs_input + (1 - release) * envelope`
///
/// The `attack` and `release` coefficients should be in the range [0.0, 1.0].
/// Typical values: attack ≈ 0.9 (fast), release ≈ 0.1 (slow).
///
/// # Complexity
///
/// - **Time per sample:** O(1); one absolute value, one comparison, and two multiply-add steps.
/// - **Space:** O(1); stores one envelope value.
#[derive(Clone, Debug)]
pub struct Envelope<T> {
    config: Config<T>,
    state: State<T>,
}

impl<T> Default for Envelope<T>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T> ConfigTrait for Envelope<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for Envelope<T> {
    type State = State<T>;
}

impl<T> WithConfig for Envelope<T>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let envelope = T::zero();
            State { envelope }
        };
        Self { config, state }
    }
}

impl<T> ConfigRef for Envelope<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for Envelope<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for Envelope<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for Envelope<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for Envelope<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for Envelope<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for Envelope<T>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for Envelope<T>
where
    T: Clone + Num,
    Self: Reset,
{
}

impl<T> Filter<T> for Envelope<T>
where
    T: Clone + Num + Signed + PartialOrd,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let abs_input = input.abs();
        let coeff = if abs_input > self.state.envelope {
            &self.config.attack
        } else {
            &self.config.release
        };
        let one_minus_coeff = T::one() - coeff.clone();
        self.state.envelope =
            coeff.clone() * abs_input + one_minus_coeff * self.state.envelope.clone();
        self.state.envelope.clone()
    }
}

/// An envelope follower for irregularly-timed samples, parameterized by time constants.
///
/// Tracks the envelope (peak amplitude) of a signal with asymmetric attack and
/// release characteristics, deriving its per-sample smoothing coefficient from
/// each sample's elapsed time `dt` rather than from a fixed coefficient. This
/// keeps the response consistent under non-uniform sampling, unlike [`Envelope`]
/// whose fixed `attack`/`release` coefficients assume a constant sample rate.
///
/// Per sample the smoothing coefficient is `alpha = 1 − exp(−dt / tau)`, using
/// `attack_tau` when the input magnitude exceeds the current envelope and
/// `release_tau` otherwise. The envelope then updates as
/// `envelope = alpha · |input| + (1 − alpha) · envelope`, mirroring [`Envelope`].
///
/// # Complexity
///
/// - **Time per sample:** O(1); one absolute value, one comparison, one
///   exponential, and two multiply-add steps.
/// - **Space:** O(1); stores one envelope value and two time constants.
#[derive(Clone, Debug)]
pub struct TimedEnvelope<T> {
    attack_tau: T,
    release_tau: T,
    envelope: T,
}

impl<T> TimedEnvelope<T>
where
    T: num_traits::Float,
{
    /// Creates an envelope follower with the given attack and release time
    /// constants (same units as the samples' `dt`).
    ///
    /// The initial envelope is `T::zero()`.
    pub fn from_time_constants(attack_tau: T, release_tau: T) -> Self {
        let envelope = T::zero();

        Self {
            attack_tau,
            release_tau,
            envelope,
        }
    }
}

impl<T> Filter<Timed<T, T>> for TimedEnvelope<T>
where
    T: num_traits::Float + Signed,
{
    type Output = T;

    fn filter(&mut self, input: Timed<T, T>) -> Self::Output {
        let Timed { value, dt } = input;

        let abs_input = value.abs();

        let tau = if abs_input > self.envelope {
            self.attack_tau
        } else {
            self.release_tau
        };

        let coeff = crate::time::smoothing_alpha(dt, tau);

        self.envelope = coeff * abs_input + (T::one() - coeff) * self.envelope;

        self.envelope
    }
}

#[cfg(test)]
mod tests;
