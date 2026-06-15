// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Caching wrapper filters.

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The cache filter's state.
#[derive(Clone, Debug)]
pub struct State<T, U> {
    /// Inner filter.
    pub inner: T,
    /// Cached value.
    pub cached: Option<U>,
}

/// A filter wrapper that caches the wrapped inner filter.
///
/// # Complexity
///
/// - **Time per sample:** same as the wrapped inner filter; `Last` adds O(1) overhead (one clone of the output).
/// - **Space:** O(1) extra; stores one `Option<U>` alongside the inner filter's own state.
#[derive(Clone, Debug)]
pub struct Last<T, U> {
    state: State<T, U>,
}

impl<T, U> Last<T, U> {
    /// Returns the most recent value returned from `self.filter(…)`, otherwise `None`.
    pub fn cached(&self) -> Option<&U> {
        self.state.cached.as_ref()
    }
}

impl<T, U> From<T> for Last<T, U> {
    fn from(inner: T) -> Self {
        let cached = None;
        let state = State { inner, cached };
        Self { state }
    }
}

impl<T, U> Default for Last<T, U>
where
    T: Default,
{
    fn default() -> Self {
        let inner = T::default();
        Self::from(inner)
    }
}

impl<T, U> ConfigTrait for Last<T, U>
where
    T: ConfigTrait,
{
    type Config = T::Config;
}

impl<T, U> StateTrait for Last<T, U>
where
    T: StateTrait,
{
    type State = State<T, U>;
}

impl<T, U> WithConfig for Last<T, U>
where
    T: WithConfig<Output = T>,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = State {
            inner: T::with_config(config),
            cached: None,
        };
        Self { state }
    }
}

impl<T, U> ConfigRef for Last<T, U>
where
    T: ConfigRef,
{
    fn config_ref(&self) -> &Self::Config {
        self.state.inner.config_ref()
    }
}

impl<T, U> ConfigClone for Last<T, U>
where
    T: ConfigClone,
{
    fn config(&self) -> Self::Config {
        self.state.inner.config()
    }
}

impl<T, U> StateMut for Last<T, U>
where
    T: StateMut,
{
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> HasGuts for Last<T, U> {
    type Guts = State<T, U>;
}

impl<T, U> FromGuts for Last<T, U> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, U> IntoGuts for Last<T, U> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, U> Reset for Last<T, U>
where
    T: Reset,
{
    fn reset(mut self) -> Self {
        let State { inner, .. } = self.state;
        self.state = State {
            inner: inner.reset(),
            cached: None,
        };
        self
    }
}

#[cfg(feature = "derive")]
impl<T, U> ResetMut for Last<T, U> where Self: Reset {}

impl<T, U, V> Filter<V> for Last<T, U>
where
    T: Filter<V, Output = U>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: V) -> Self::Output {
        let cached = self.state.inner.filter(input);
        self.state.cached = Some(cached.clone());
        cached
    }
}

#[cfg(test)]
mod tests;
