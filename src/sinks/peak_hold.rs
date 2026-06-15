// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Peak hold sinks.
//!
//! Tracks the maximum absolute value of a signal, with optional decay when no larger signal arrives.

use num_traits::{Num, Signed};

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Finalize, Reset, Sink,
    State as StateTrait, StateMut, WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The peak hold sink's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Decay factor applied each sample. `T::one()` = no decay, `0.99` = gentle decay.
    pub decay: T,
}

/// The peak hold sink's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// The current peak value.
    pub peak: Option<T>,
}

/// A peak hold sink that tracks the maximum absolute value of a signal,
/// with optional decay when no larger signal arrives.
#[derive(Clone, Debug)]
pub struct PeakHold<T> {
    /// The sink's configuration.
    config: Config<T>,
    /// Current internal state.
    state: State<T>,
}

impl<T> ConfigTrait for PeakHold<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for PeakHold<T> {
    type State = State<T>;
}

impl<T> WithConfig for PeakHold<T>
where
    T: Clone + Num,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = State { peak: None };
        Self { config, state }
    }
}

impl<T> ConfigRef for PeakHold<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T> ConfigClone for PeakHold<T>
where
    Config<T>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for PeakHold<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for PeakHold<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for PeakHold<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for PeakHold<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for PeakHold<T>
where
    T: Clone + Num,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for PeakHold<T> where Self: Reset {}

impl<T> Filter<T> for PeakHold<T>
where
    T: Clone + Num + Signed + PartialOrd,
{
    type Output = T;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let abs_input = input.abs();

        let new_peak = if let Some(ref current_peak) = self.state.peak {
            let decayed_peak = current_peak.clone() * self.config.decay.clone();
            if abs_input >= decayed_peak {
                abs_input.clone()
            } else {
                decayed_peak
            }
        } else {
            abs_input.clone()
        };

        self.state.peak = Some(new_peak.clone());
        new_peak
    }
}

impl<T> Sink<T> for PeakHold<T>
where
    Self: Filter<T>,
{
    #[inline]
    fn sink(&mut self, input: T) {
        let _ = self.filter(input);
    }
}

impl<T> Finalize for PeakHold<T> {
    type Output = Option<T>;

    #[inline]
    fn finalize(self) -> Self::Output {
        self.state.peak
    }
}

impl<T> Default for PeakHold<T>
where
    T: Clone + Num,
{
    fn default() -> Self {
        Self::with_config(Config { decay: T::one() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_hold_no_decay() {
        let mut sink = PeakHold::default();
        sink.sink(0.5);
        sink.sink(1.0);
        sink.sink(0.8);
        sink.sink(0.3);
        let result = sink.finalize();
        assert_eq!(result, Some(1.0));
    }

    #[test]
    fn test_peak_hold_with_decay() {
        let config = Config { decay: 0.5f32 };
        let mut sink = PeakHold::with_config(config);
        sink.sink(1.0);
        // After first sample: peak = 1.0
        sink.sink(0.0);
        // After second sample: peak = max(0, 1.0 * 0.5) = 0.5
        let result = sink.finalize();
        assert!(result.is_some());
        let peak = result.unwrap();
        assert!(peak > 0.4 && peak < 0.6); // approximately 0.5
    }

    #[test]
    fn test_peak_hold_negative_inputs() {
        let mut sink = PeakHold::default();
        sink.sink(-0.5);
        sink.sink(-2.0);
        sink.sink(-0.3);
        let result = sink.finalize();
        // Should track absolute value, so peak is 2.0
        assert_eq!(result, Some(2.0));
    }

    #[test]
    fn test_filter_output() {
        let mut filter = PeakHold::default();
        let out1 = filter.filter(0.5);
        assert_eq!(out1, 0.5);
        let out2 = filter.filter(1.0);
        assert_eq!(out2, 1.0);
        let out3 = filter.filter(0.8);
        assert_eq!(out3, 1.0); // Peak holds at 1.0
        let out4 = filter.filter(0.3);
        assert_eq!(out4, 1.0); // Still holding
    }

    #[test]
    fn test() {
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let mut sink: PeakHold<f32> = PeakHold::default();
        for value in input {
            sink.sink(value);
        }
        let result = sink.finalize();
        assert_eq!(result, Some(20.0));
    }
}
