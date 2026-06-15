// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use approx::assert_abs_diff_eq;

use crate::traits::{
    Config as ConfigTrait, ConfigClone, ConfigRef, Reset, State as StateTrait, StateMut, WithConfig,
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
    fn state_mut(&mut self) -> &mut Self::State {
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

    let mut cache = Last::from(add_fourty_two);
    assert_eq!(cache.cached(), None);

    cache.filter(0.0);
    assert_eq!(cache.cached(), Some(&4.2));

    cache.filter(1.0);
    assert_eq!(cache.cached(), Some(&5.2));
}

#[test]
fn test_default() {
    let mut cache: Last<AddFourPointTwo, f32> = Last::default();
    assert_eq!(cache.cached(), None);

    cache.filter(10.0);
    assert_eq!(cache.cached(), Some(&14.2));
}

#[test]
fn test_with_config() {
    let config = TestConfig { offset: 3.5 };
    let mut cache: Last<ConfigurableAdd, f32> = Last::with_config(config);
    assert_eq!(cache.cached(), None);

    cache.filter(2.0);
    assert_eq!(cache.cached(), Some(&5.5));
}

#[test]
fn test_config_ref() {
    let config = TestConfig { offset: 3.5 };
    let cache: Last<ConfigurableAdd, f32> = Last::with_config(config.clone());
    let config_ref = cache.config_ref();
    assert_abs_diff_eq!(config_ref.offset, 3.5, epsilon = 1e-6);
}

#[test]
fn test_config_clone() {
    let config = TestConfig { offset: 3.5 };
    let cache: Last<ConfigurableAdd, f32> = Last::with_config(config.clone());
    let cloned_config = cache.config();
    assert_abs_diff_eq!(cloned_config.offset, 3.5, epsilon = 1e-6);
}

#[test]
fn test_state_mut() {
    let add = ConfigurableAdd::with_config(TestConfig { offset: 1.0 });
    let mut cache: Last<ConfigurableAdd, f32> = Last::from(add);
    cache.filter(5.0);

    let state = cache.state_mut();
    assert_eq!(state.cached, Some(6.0));
    state.cached = Some(10.0);

    assert_eq!(cache.cached(), Some(&10.0));
}

#[test]
fn test_from_into_guts() {
    use crate::traits::guts::{FromGuts, IntoGuts};

    let add = ConfigurableAdd::with_config(TestConfig { offset: 2.0 });
    let mut cache: Last<ConfigurableAdd, f32> = Last::from(add);
    cache.filter(3.0);

    let guts = cache.into_guts();
    assert_eq!(guts.cached, Some(5.0));

    let cache2 = Last::from_guts(guts);
    assert_eq!(cache2.cached(), Some(&5.0));
}

#[test]
fn test_reset() {
    let add = ConfigurableAdd::with_config(TestConfig { offset: 1.5 });
    let mut cache: Last<ConfigurableAdd, f32> = Last::from(add);
    cache.filter(4.0);
    assert_eq!(cache.cached(), Some(&5.5));

    let reset_cache = cache.reset();
    assert_eq!(reset_cache.cached(), None);

    // After reset, filtering should still work
    let mut reset_cache = reset_cache;
    reset_cache.filter(10.0);
    assert_eq!(reset_cache.cached(), Some(&11.5));
}
