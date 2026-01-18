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

impl<T, U> HasGuts for Cache<T, U> {
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
            cached: None,
        };
        self
    }
}

#[cfg(feature = "derive")]
impl<T, U> ResetMut for Cache<T, U> where Self: Reset {}

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
    use nearly_eq::assert_nearly_eq;

    use crate::traits::{
        Config as ConfigTrait, ConfigClone, ConfigRef, Reset, State as StateTrait, StateMut,
        WithConfig,
    };

    use super::*;

    #[derive(Default)]
    struct AddFourPointTwo;

    impl Filter<f32> for AddFourPointTwo {
        type Output = f32;

        fn filter(&mut self, input: f32) -> Self::Output {
            input + 4.2
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct TestConfig {
        offset: f32,
    }

    #[derive(Debug, Clone)]
    struct ConfigurableAdd {
        config: TestConfig,
        state_placeholder: (),
    }

    impl ConfigTrait for ConfigurableAdd {
        type Config = TestConfig;
    }

    impl WithConfig for ConfigurableAdd {
        type Output = Self;

        fn with_config(config: Self::Config) -> Self::Output {
            Self {
                config,
                state_placeholder: (),
            }
        }
    }

    impl ConfigRef for ConfigurableAdd {
        fn config_ref(&self) -> &Self::Config {
            &self.config
        }
    }

    impl ConfigClone for ConfigurableAdd {
        fn config(&self) -> Self::Config {
            self.config.clone()
        }
    }

    impl StateTrait for ConfigurableAdd {
        type State = ();
    }

    impl StateMut for ConfigurableAdd {
        unsafe fn state_mut(&mut self) -> &mut Self::State {
            &mut self.state_placeholder
        }
    }

    impl Reset for ConfigurableAdd {
        fn reset(self) -> Self {
            Self::with_config(self.config)
        }
    }

    impl Filter<f32> for ConfigurableAdd {
        type Output = f32;

        fn filter(&mut self, input: f32) -> Self::Output {
            input + self.config.offset
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

    #[test]
    fn test_default() {
        let mut cache: Cache<AddFourPointTwo, f32> = Cache::default();
        assert_nearly_eq!(cache.cached(), None);

        cache.filter(10.0);
        assert_nearly_eq!(cache.cached(), Some(14.2));
    }

    #[test]
    fn test_with_config() {
        let config = TestConfig { offset: 3.5 };
        let mut cache: Cache<ConfigurableAdd, f32> = Cache::with_config(config);
        assert_nearly_eq!(cache.cached(), None);

        cache.filter(2.0);
        assert_nearly_eq!(cache.cached(), Some(5.5));
    }

    #[test]
    fn test_config_ref() {
        let config = TestConfig { offset: 3.5 };
        let cache: Cache<ConfigurableAdd, f32> = Cache::with_config(config.clone());
        let config_ref = cache.config_ref();
        assert_nearly_eq!(config_ref.offset, 3.5);
    }

    #[test]
    fn test_config_clone() {
        let config = TestConfig { offset: 3.5 };
        let cache: Cache<ConfigurableAdd, f32> = Cache::with_config(config.clone());
        let cloned_config = cache.config();
        assert_nearly_eq!(cloned_config.offset, 3.5);
    }

    #[test]
    fn test_state_mut() {
        let add = ConfigurableAdd::with_config(TestConfig { offset: 1.0 });
        let mut cache: Cache<ConfigurableAdd, f32> = Cache::from(add);
        cache.filter(5.0);

        unsafe {
            let state = cache.state_mut();
            assert_nearly_eq!(state.cached, Some(6.0));
            state.cached = Some(10.0);
        }

        assert_nearly_eq!(cache.cached(), Some(10.0));
    }

    #[test]
    fn test_from_into_guts() {
        use crate::traits::guts::{FromGuts, IntoGuts};

        let add = ConfigurableAdd::with_config(TestConfig { offset: 2.0 });
        let mut cache: Cache<ConfigurableAdd, f32> = Cache::from(add);
        cache.filter(3.0);

        let guts = cache.into_guts();
        assert_nearly_eq!(guts.cached, Some(5.0));

        let cache2 = Cache::from_guts(guts);
        assert_nearly_eq!(cache2.cached(), Some(5.0));
    }

    #[test]
    fn test_reset() {
        let add = ConfigurableAdd::with_config(TestConfig { offset: 1.5 });
        let mut cache: Cache<ConfigurableAdd, f32> = Cache::from(add);
        cache.filter(4.0);
        assert_nearly_eq!(cache.cached(), Some(5.5));

        let reset_cache = cache.reset();
        assert_nearly_eq!(reset_cache.cached(), None);

        // After reset, filtering should still work
        let mut reset_cache = reset_cache;
        reset_cache.filter(10.0);
        assert_nearly_eq!(reset_cache.cached(), Some(11.5));
    }
}
