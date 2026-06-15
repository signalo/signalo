// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Schmitt trigger hysteresis filter.
//!
//! Implements hysteresis with separate threshold values for rising and falling edges,
//! preventing oscillation around a single threshold and improving noise immunity.

use core::cmp::PartialOrd;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// A validated pair of threshold values for a Schmitt trigger,
/// guaranteeing that `low <= high`.
#[derive(Clone, Debug)]
pub struct Thresholds<T> {
    low: T,
    high: T,
}

impl<T> Thresholds<T>
where
    T: PartialOrd,
{
    /// Creates a new `Thresholds` pair.
    ///
    /// Returns `None` if `low > high`.
    pub fn new(low: T, high: T) -> Option<Self> {
        if low <= high {
            Some(Self { low, high })
        } else {
            None
        }
    }

    /// Returns the low threshold.
    pub fn low(&self) -> &T {
        &self.low
    }

    /// Returns the high threshold.
    pub fn high(&self) -> &T {
        &self.high
    }
}

/// The [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger)'s configuration.
#[derive(Clone, Debug)]
pub struct Config<T, U> {
    /// Validated [low, high] input thresholds.
    pub thresholds: Thresholds<T>,
    /// [off, on] outputs.
    pub outputs: [U; 2],
}

/// The [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger)'s state.
#[derive(Clone, Debug)]
pub struct State {
    /// The current state.
    pub on: bool,
}

/// A [Schmitt trigger](https://en.wikipedia.org/wiki/Schmitt_trigger).
///
/// # Complexity
///
/// - **Time per sample:** O(1) — two threshold comparisons and an array index.
/// - **Space:** O(1) — stores one `bool`.
#[derive(Clone, Debug)]
pub struct Schmitt<T, U> {
    /// The filter's configuration.
    config: Config<T, U>,
    /// Current internal state.
    state: State,
}

impl<T, U> ConfigTrait for Schmitt<T, U> {
    type Config = Config<T, U>;
}

impl<T, U> StateTrait for Schmitt<T, U> {
    type State = State;
}

impl<T, U> WithConfig for Schmitt<T, U> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = {
            let on = false;
            State { on }
        };
        Self { config, state }
    }
}

impl<T, U> ConfigRef for Schmitt<T, U> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> ConfigClone for Schmitt<T, U>
where
    Config<T, U>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, U> StateMut for Schmitt<T, U> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, U> HasGuts for Schmitt<T, U> {
    type Guts = (Config<T, U>, State);
}

impl<T, U> FromGuts for Schmitt<T, U> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T, U> IntoGuts for Schmitt<T, U> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T, U> Reset for Schmitt<T, U> {
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T, U> ResetMut for Schmitt<T, U> where Self: Reset {}

impl<T, U> Filter<T> for Schmitt<T, U>
where
    T: PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        self.state.on = if self.state.on {
            input >= *self.config.thresholds.low()
        } else {
            input > *self.config.thresholds.high()
        };
        let index: usize = self.state.on.into();
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use crate::filters::classify::Classification;

    use super::*;

    #[test]
    fn test() {
        let filter = Schmitt::with_config(Config {
            thresholds: Thresholds::new(5, 10).unwrap(),
            outputs: u8::classes(),
        });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = [
            0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();
        assert_eq!(
            output,
            vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        );
    }

    #[test]
    fn test_config_ref() {
        let config = Config {
            thresholds: Thresholds::new(3.0, 7.0).unwrap(),
            outputs: [false, true],
        };
        let filter = Schmitt::with_config(config);
        let config_ref = filter.config_ref();
        assert_eq!(*config_ref.thresholds.low(), 3.0);
        assert_eq!(*config_ref.thresholds.high(), 7.0);
        assert_eq!(config_ref.outputs, [false, true]);
    }

    #[test]
    fn test_config_clone() {
        let config = Config {
            thresholds: Thresholds::new(10, 20).unwrap(),
            outputs: [0, 100],
        };
        let filter = Schmitt::with_config(config.clone());
        let cloned_config = filter.config();
        assert_eq!(*cloned_config.thresholds.low(), 10);
        assert_eq!(*cloned_config.thresholds.high(), 20);
        assert_eq!(cloned_config.outputs, [0, 100]);
    }

    #[test]
    fn test_state_mut() {
        let config = Config {
            thresholds: Thresholds::new(5, 10).unwrap(),
            outputs: [0, 1],
        };
        let mut filter = Schmitt::with_config(config);

        // Initially off
        assert_eq!(filter.filter(3), 0);

        let state = filter.state_mut();
        assert_eq!(state.on, false);
        // Force the state to on
        state.on = true;

        // Now with state forced to on, even low values should keep it on until below threshold
        let output = filter.filter(6);
        assert_eq!(output, 1); // 6 >= 5, so stays on
    }

    #[test]
    fn test_from_into_guts() {
        use crate::traits::guts::{FromGuts, IntoGuts};

        let config = Config {
            thresholds: Thresholds::new(2.5, 7.5).unwrap(),
            outputs: [10.0, 20.0],
        };
        let mut filter = Schmitt::with_config(config);
        filter.filter(10.0); // Turn on

        let (guts_config, guts_state) = filter.into_guts();
        assert_eq!(*guts_config.thresholds.low(), 2.5);
        assert_eq!(*guts_config.thresholds.high(), 7.5);
        assert_eq!(guts_state.on, true);

        let filter2 = Schmitt::from_guts((guts_config, guts_state));
        let mut filter2 = filter2;
        // State is on, so it should output 20.0 for values >= 2.5
        let output = filter2.filter(5.0);
        assert_eq!(output, 20.0);
    }

    #[test]
    fn test_reset() {
        let config = Config {
            thresholds: Thresholds::new(5, 10).unwrap(),
            outputs: [0, 1],
        };
        let mut filter = Schmitt::with_config(config);

        // Turn the schmitt trigger on
        filter.filter(15);
        let output = filter.filter(8);
        assert_eq!(output, 1); // Still on because 8 >= 5

        // Reset the filter
        let mut reset_filter = filter.reset();

        // After reset, state should be off again
        let output = reset_filter.filter(8);
        assert_eq!(output, 0); // Off because 8 is not > 10
    }

    #[test]
    fn thresholds_rejects_inverted() {
        assert!(Thresholds::<i32>::new(10, 5).is_none());
        assert!(Thresholds::<f32>::new(1.0, 0.0).is_none());
    }

    #[test]
    fn thresholds_accepts_equal() {
        // low == high is valid (degenerate trigger that never produces hysteresis,
        // but does not break the invariant).
        assert!(Thresholds::<i32>::new(5, 5).is_some());
    }
}
