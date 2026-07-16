// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Cascaded biquad filters for higher-order IIR implementations.
//!
//! Combines multiple biquad stages in series to achieve higher-order filter responses
//! while maintaining stability and reducing computational complexity of direct higher-order
//! implementations. A cascade of biquad filters applied sequentially. Each stage is a
//! second-order IIR filter. Stages are applied in index order: `sections[0]` receives the
//! input first, its output feeds `sections[1]`, and so on through the last section.
//!
//! This is useful for higher-order filtering without explicit state-space implementations,
//! as each biquad stage can be designed independently (e.g., using `sos` format from filter
//! design).

use core::ops::{Add, Mul, Sub};

use num_traits::{Num, Zero};

use crate::storage::AsSlice;
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
/// Holds the configuration (coefficients) for each biquad stage. The storage
/// container `CS` must implement [`AsSlice<BiquadConfig<K>>`]; use the
/// [`BiquadCascadeArray`] alias for fixed-size stack storage or the
/// [`BiquadCascadeVec`] alias for heap-allocated, runtime-sized storage.
///
/// `K` is the coefficient type for every section.
#[derive(Clone, Debug)]
pub struct Config<K, CS> {
    /// Storage for biquad configurations (one per stage).
    pub sections: CS,
    _phantom: core::marker::PhantomData<K>,
}

impl<K, CS> Config<K, CS> {
    /// Creates a new cascade configuration from the given sections storage.
    pub fn new(sections: CS) -> Self {
        Self {
            sections,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<K, const N: usize> From<[[K; 5]; N]> for Config<K, [BiquadConfig<K>; N]> {
    fn from(sections: [[K; 5]; N]) -> Self {
        Self::new(sections.map(BiquadConfig::from))
    }
}

impl<K, const N: usize> From<Config<K, [BiquadConfig<K>; N]>> for [[K; 5]; N] {
    fn from(c: Config<K, [BiquadConfig<K>; N]>) -> Self {
        c.sections.map(<[K; 5]>::from)
    }
}

impl<K, const N: usize> Default for Config<K, [BiquadConfig<K>; N]>
where
    K: Num,
{
    fn default() -> Self {
        Self::new(core::array::from_fn(|_| BiquadConfig::default()))
    }
}

/// The biquad cascade state.
///
/// Holds the state (delay lines) for each biquad stage. The storage
/// container `SS` must implement [`AsSlice<BiquadState<T>>`]; use the
/// [`BiquadCascadeArray`] alias for fixed-size stack storage or the
/// [`BiquadCascadeVec`] alias for heap-allocated, runtime-sized storage.
///
/// `T` is the sample type and therefore also the per-section state type.
#[derive(Clone, Debug)]
pub struct State<T, SS> {
    /// Storage for biquad states (one per stage).
    pub sections: SS,
    _phantom: core::marker::PhantomData<T>,
}

impl<T, SS> State<T, SS> {
    /// Creates a new cascade state from the given sections storage.
    pub fn new(sections: SS) -> Self {
        Self {
            sections,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<T, const N: usize> Default for State<T, [BiquadState<T>; N]>
where
    T: Zero,
{
    fn default() -> Self {
        Self::new(core::array::from_fn(|_| BiquadState::default()))
    }
}

/// A cascade of biquad filters applied sequentially.
///
/// Generic over sample/state type `T` and coefficient type `K`. `K` defaults
/// to `T`, preserving the common same-type cascade usage.
///
/// `CS` is the config sections storage (must implement [`AsSlice<BiquadConfig<K>>`])
/// and `SS` is the state sections storage (must implement [`AsSlice<BiquadState<T>>`]).
///
/// Use the [`BiquadCascadeArray`] type alias for fixed-size stack allocation or
/// [`BiquadCascadeVec`] for heap-allocated, runtime-sized storage.
#[derive(Clone, Debug)]
pub struct BiquadCascade<T, CS, SS, K = T> {
    config: Config<K, CS>,
    state: State<T, SS>,
}

/// A [`BiquadCascade`] backed by fixed-size arrays `[BiquadConfig<K>; N]` and
/// `[BiquadState<T>; N]`.
///
/// Provides stack-allocated, `no_std`-friendly storage. Use [`BiquadCascadeVec`]
/// when the number of sections is only known at runtime.
pub type BiquadCascadeArray<T, const N: usize, K = T> =
    BiquadCascade<T, [BiquadConfig<K>; N], [BiquadState<T>; N], K>;

/// A [`BiquadCascade`] backed by heap-allocated `Vec<BiquadConfig<K>>` and
/// `Vec<BiquadState<T>>`.
///
/// Requires the `alloc` feature. Use [`BiquadCascadeArray`] for `no_std` contexts
/// where the number of sections is known at compile time.
#[cfg(feature = "alloc")]
pub type BiquadCascadeVec<T, K = T> =
    BiquadCascade<T, alloc::vec::Vec<BiquadConfig<K>>, alloc::vec::Vec<BiquadState<T>>, K>;

/// A [`BiquadCascade`] that borrows `[BiquadConfig<K>]` and `[BiquadState<T>]`
/// slices for its section storage.
///
/// This alias allows sharing caller-owned coefficient and state slices without
/// taking ownership. Construct via [`BiquadCascade::from_guts`], passing
/// [`Config::new`] and [`State::new`] each wrapping mutable slices.
pub type BiquadCascadeRefMut<'a, T, K = T> =
    BiquadCascade<T, &'a mut [BiquadConfig<K>], &'a mut [BiquadState<T>], K>;

impl<T, const N: usize, K> Default for BiquadCascadeArray<T, N, K>
where
    T: Zero,
    K: Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T, CS, SS, K> ConfigTrait for BiquadCascade<T, CS, SS, K> {
    type Config = Config<K, CS>;
}

impl<T, CS, SS, K> StateTrait for BiquadCascade<T, CS, SS, K> {
    type State = State<T, SS>;
}

impl<T, const N: usize, K> WithConfig for BiquadCascadeArray<T, N, K>
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

impl<T, CS, SS, K> ConfigClone for BiquadCascade<T, CS, SS, K>
where
    K: Clone,
    CS: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, CS, SS, K> ConfigRef for BiquadCascade<T, CS, SS, K> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, CS, SS, K> StateMut for BiquadCascade<T, CS, SS, K> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, CS, SS, K> HasGuts for BiquadCascade<T, CS, SS, K> {
    type Guts = (Config<K, CS>, State<T, SS>);
}

impl<T, CS, SS, K> FromGuts for BiquadCascade<T, CS, SS, K> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, CS, SS, K> IntoGuts for BiquadCascade<T, CS, SS, K> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize, K> Reset for BiquadCascadeArray<T, N, K>
where
    T: Zero,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize, K> ResetMut for BiquadCascadeArray<T, N, K> where Self: Reset {}

impl<T, CS, SS, K> Filter<T> for BiquadCascade<T, CS, SS, K>
where
    T: Clone + Zero + Add<Output = T> + Sub<Output = T> + Mul<K, Output = T>,
    K: Clone,
    CS: AsSlice<BiquadConfig<K>>,
    SS: AsSlice<BiquadState<T>>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let mut x = input;
        let n = self.config.sections.as_slice().len();
        for i in 0..n {
            let cfg = &self.config.sections.as_slice()[i];
            let st = &mut self.state.sections.as_mut_slice()[i];
            x = df2t_step(cfg, st, x);
        }
        x
    }
}

#[cfg(test)]
mod tests;
