// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Cascaded biquad filters for higher-order IIR implementations.
//!
//! Combines multiple biquad stages in series to achieve higher-order filter responses
//! while maintaining stability and reducing computational complexity of direct higher-order implementations.
//! A cascade of N biquad filters applied sequentially. Each stage is a second-order IIR filter.
//! Stages are applied in index order: `sections[0]` receives the input first,
//! its output feeds `sections[1]`, and so on through `sections[N-1]`.
//!
//! This is useful for higher-order filtering without explicit state-space implementations,
//! as each biquad stage can be designed independently (e.g., using `sos` format from filter design).

use num_traits::Num;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

use super::{df2t_step, Config as BiquadConfig, State as BiquadState};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The biquad cascade configuration.
///
/// Holds the configuration (coefficients) for each of the N biquad stages.
#[derive(Clone, Debug)]
pub struct Config<T, const N: usize> {
    /// Array of biquad configurations (one per stage).
    pub sections: [BiquadConfig<T>; N],
}

impl<T, const N: usize> From<[[T; 5]; N]> for Config<T, N> {
    fn from(sections: [[T; 5]; N]) -> Self {
        Self {
            sections: sections.map(BiquadConfig::from),
        }
    }
}

impl<T, const N: usize> From<Config<T, N>> for [[T; 5]; N] {
    fn from(c: Config<T, N>) -> Self {
        c.sections.map(<[T; 5]>::from)
    }
}

impl<T, const N: usize> Default for Config<T, N>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self {
            sections: core::array::from_fn(|_| BiquadConfig::default()),
        }
    }
}

/// The biquad cascade state.
///
/// Holds the state (delay lines) for each of the N biquad stages.
#[derive(Clone, Debug)]
pub struct State<T, const N: usize> {
    /// Array of biquad states (one per stage).
    pub sections: [BiquadState<T>; N],
}

impl<T, const N: usize> Default for State<T, N>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self {
            sections: core::array::from_fn(|_| BiquadState::default()),
        }
    }
}

/// A cascade of N biquad filters applied sequentially.
#[derive(Clone, Debug)]
pub struct BiquadCascade<T, const N: usize> {
    config: Config<T, N>,
    state: State<T, N>,
}

impl<T, const N: usize> Default for BiquadCascade<T, N>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T, const N: usize> ConfigTrait for BiquadCascade<T, N> {
    type Config = Config<T, N>;
}

impl<T, const N: usize> StateTrait for BiquadCascade<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> WithConfig for BiquadCascade<T, N>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self {
            config,
            state: State::default(),
        }
    }
}

impl<T, const N: usize> ConfigClone for BiquadCascade<T, N>
where
    T: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, const N: usize> ConfigRef for BiquadCascade<T, N> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, const N: usize> StateMut for BiquadCascade<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for BiquadCascade<T, N> {
    type Guts = (Config<T, N>, State<T, N>);
}

impl<T, const N: usize> FromGuts for BiquadCascade<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, const N: usize> IntoGuts for BiquadCascade<T, N> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for BiquadCascade<T, N>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for BiquadCascade<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for BiquadCascade<T, N>
where
    T: Clone + Num,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let mut x = input;
        for i in 0..N {
            let cfg = &self.config.sections[i];
            let st = &mut self.state.sections[i];
            x = df2t_step(cfg, st, x);
        }
        x
    }
}

#[cfg(test)]
mod tests;
