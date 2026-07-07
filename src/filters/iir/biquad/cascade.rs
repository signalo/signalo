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

use num_traits::Num;

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
/// container `CS` must implement [`AsSlice<BiquadConfig<T>>`]; use the
/// [`BiquadCascadeArray`] alias for fixed-size stack storage or the
/// [`BiquadCascadeVec`] alias for heap-allocated, runtime-sized storage.
#[derive(Clone, Debug)]
pub struct Config<T, CS> {
    /// Storage for biquad configurations (one per stage).
    pub sections: CS,
    _phantom: core::marker::PhantomData<T>,
}

impl<T, CS> Config<T, CS> {
    /// Creates a new cascade configuration from the given sections storage.
    pub fn new(sections: CS) -> Self {
        Self {
            sections,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<T, const N: usize> From<[[T; 5]; N]> for Config<T, [BiquadConfig<T>; N]> {
    fn from(sections: [[T; 5]; N]) -> Self {
        Self::new(sections.map(BiquadConfig::from))
    }
}

impl<T, const N: usize> From<Config<T, [BiquadConfig<T>; N]>> for [[T; 5]; N] {
    fn from(c: Config<T, [BiquadConfig<T>; N]>) -> Self {
        c.sections.map(<[T; 5]>::from)
    }
}

impl<T, const N: usize> Default for Config<T, [BiquadConfig<T>; N]>
where
    T: Clone + Num,
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
    T: Clone + Num,
{
    fn default() -> Self {
        Self::new(core::array::from_fn(|_| BiquadState::default()))
    }
}

/// A cascade of biquad filters applied sequentially.
///
/// `CS` is the config sections storage (must implement [`AsSlice<BiquadConfig<T>>`])
/// and `SS` is the state sections storage (must implement [`AsSlice<BiquadState<T>>`]).
///
/// Use the [`BiquadCascadeArray`] type alias for fixed-size stack allocation or
/// [`BiquadCascadeVec`] for heap-allocated, runtime-sized storage.
#[derive(Clone, Debug)]
pub struct BiquadCascade<T, CS, SS> {
    config: Config<T, CS>,
    state: State<T, SS>,
}

/// A [`BiquadCascade`] backed by fixed-size arrays `[BiquadConfig<T>; N]` and
/// `[BiquadState<T>; N]`.
///
/// Provides stack-allocated, `no_std`-friendly storage. Use [`BiquadCascadeVec`]
/// when the number of sections is only known at runtime.
pub type BiquadCascadeArray<T, const N: usize> =
    BiquadCascade<T, [BiquadConfig<T>; N], [BiquadState<T>; N]>;

/// A [`BiquadCascade`] backed by heap-allocated `Vec<BiquadConfig<T>>` and
/// `Vec<BiquadState<T>>`.
///
/// Requires the `alloc` feature. Use [`BiquadCascadeArray`] for `no_std` contexts
/// where the number of sections is known at compile time.
#[cfg(feature = "alloc")]
pub type BiquadCascadeVec<T> =
    BiquadCascade<T, alloc::vec::Vec<BiquadConfig<T>>, alloc::vec::Vec<BiquadState<T>>>;

/// A [`BiquadCascade`] that borrows `[BiquadConfig<T>]` and `[BiquadState<T>]`
/// slices for its section storage.
///
/// This alias allows sharing caller-owned coefficient and state slices without
/// taking ownership. Construct via [`BiquadCascade::from_guts`], passing
/// [`Config::new`] and [`State::new`] each wrapping a `&mut [T]` slice.
pub type BiquadCascadeRefMut<'a, T> =
    BiquadCascade<T, &'a mut [BiquadConfig<T>], &'a mut [BiquadState<T>]>;

impl<T, const N: usize> Default for BiquadCascadeArray<T, N>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config::default())
    }
}

impl<T, CS, SS> ConfigTrait for BiquadCascade<T, CS, SS> {
    type Config = Config<T, CS>;
}

impl<T, CS, SS> StateTrait for BiquadCascade<T, CS, SS> {
    type State = State<T, SS>;
}

impl<T, const N: usize> WithConfig for BiquadCascadeArray<T, N>
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

impl<T, CS, SS> ConfigClone for BiquadCascade<T, CS, SS>
where
    T: Clone,
    CS: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, CS, SS> ConfigRef for BiquadCascade<T, CS, SS> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, CS, SS> StateMut for BiquadCascade<T, CS, SS> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, CS, SS> HasGuts for BiquadCascade<T, CS, SS> {
    type Guts = (Config<T, CS>, State<T, SS>);
}

impl<T, CS, SS> FromGuts for BiquadCascade<T, CS, SS> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, CS, SS> IntoGuts for BiquadCascade<T, CS, SS> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, const N: usize> Reset for BiquadCascadeArray<T, N>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for BiquadCascadeArray<T, N> where Self: Reset {}

impl<T, CS, SS> Filter<T> for BiquadCascade<T, CS, SS>
where
    T: Clone + Num,
    CS: AsSlice<BiquadConfig<T>>,
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
