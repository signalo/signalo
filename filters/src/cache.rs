// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Caching wrapper filters.

use signalo_traits::Filter;
use signalo_traits::{
    Config as ConfigTrait, ConfigClone, ConfigRef, FromGuts, Guts, IntoGuts, Reset,
    State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "derive_reset_mut")]
use signalo_traits::ResetMut;

/// The cache filter's state.
#[derive(Clone, Debug)]
pub struct State<T, U> {
    /// Inner filter.
    pub inner: T,
    /// Cached value.
    pub cached: Option<U>,
}

/// A filter wrapper that caches the wrapped inner filter.
#[derive(Clone, Debug)]
pub struct Cache<T, U> {
    state: State<T, U>,
}

impl<T, U> Cache<T, U> {
    /// Returns the most recent value returned from `self.filter(…)`, otherwise `None`.
    pub fn cached(&self) -> Option<&U> {
        self.state.cached.as_ref()
    }
}

impl<T, U> From<T> for Cache<T, U> {
    fn from(inner: T) -> Self {
        let cached = None;
        let state = State { inner, cached };
        Self { state }
    }
}

impl<T, U> Default for Cache<T, U>
where
    T: Default,
{
    fn default() -> Self {
        let inner = T::default();
        Self::from(inner)
    }
}

impl<T, U> ConfigTrait for Cache<T, U>
where
    T: ConfigTrait,
{
    type Config = T::Config;
}

impl<T, U> StateTrait for Cache<T, U>
where
    T: StateTrait,
{
    type State = State<T, U>;
}

impl<T, U> WithConfig for Cache<T, U>
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

impl<T, U> ConfigRef for Cache<T, U>
where
    T: ConfigRef,
{
    fn config_ref(&self) -> &Self::Config {
        self.state.inner.config_ref()
    }
}

impl<T, U> ConfigClone for Cache<T, U>
where
    T: ConfigClone,
{
    fn config(&self) -> Self::Config {
        self.state.inner.config()
    }
}

impl<T, U> StateMut for Cache<T, U>
where
    T: StateMut,
{
    unsafe fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> Guts for Cache<T, U> {
    type Guts = State<T, U>;
}

impl<T, U> FromGuts for Cache<T, U> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, U> IntoGuts for Cache<T, U> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, U> Reset for Cache<T, U>
where
    T: Reset,
{
    fn reset(mut self) -> Self {
        let State { inner, .. } = self.state;
        self.state = State {
            inner: inner.reset(),
            cached: None
        };
        self
    }
}

#[cfg(feature = "derive_reset_mut")]
impl<T, U> ResetMut for Cache<T, U>
where
    Self: Reset,
{}

impl<T, U, V> Filter<V> for Cache<T, U>
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
mod tests {
    use super::*;

    struct AddFourPointTwo;

    impl Filter<f32> for AddFourPointTwo {
        type Output = f32;

        fn filter(&mut self, input: f32) -> Self::Output {
            input + 4.2
        }
    }

    #[test]
    fn test() {
        let add_fourty_two = AddFourPointTwo;

        let mut cache = Cache::from(add_fourty_two);
        assert_nearly_eq!(cache.cached(), None);

        cache.filter(0.0);
        assert_nearly_eq!(cache.cached(), Some(4.2));

        cache.filter(1.0);
        assert_nearly_eq!(cache.cached(), Some(5.2));
    }
}
