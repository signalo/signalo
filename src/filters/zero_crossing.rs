// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Zero-crossing detection filters.

use num_traits::Signed;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, State as StateTrait, StateMut,
    WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The zero-crossing detector's configuration.
#[derive(Clone, Debug)]
pub struct Config<T> {
    /// Hysteresis threshold around zero.
    /// A zero crossing is detected when `sign(prev) != sign(input)` AND `|input| > hysteresis`.
    /// Default value of 0 means no hysteresis applied.
    pub hysteresis: T,
}

impl<T: Default> Default for Config<T> {
    fn default() -> Self {
        Self {
            hysteresis: T::default(),
        }
    }
}

/// The zero-crossing detector's state.
#[derive(Clone, Debug)]
pub struct State<T> {
    /// Previous sample value. `None` on the first sample.
    pub prev: Option<T>,
}

/// A zero-crossing detector that identifies when a signal crosses zero.
#[derive(Clone, Debug)]
pub struct ZeroCrossing<T> {
    /// The filter's configuration.
    config: Config<T>,
    /// Current internal state.
    state: State<T>,
}

impl<T> ConfigTrait for ZeroCrossing<T> {
    type Config = Config<T>;
}

impl<T> StateTrait for ZeroCrossing<T> {
    type State = State<T>;
}

impl<T> WithConfig for ZeroCrossing<T>
where
    T: Clone,
{
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        let state = State { prev: None };
        Self { config, state }
    }
}

impl<T> ConfigRef for ZeroCrossing<T> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T: Clone> ConfigClone for ZeroCrossing<T> {
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T> StateMut for ZeroCrossing<T> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T> HasGuts for ZeroCrossing<T> {
    type Guts = (Config<T>, State<T>);
}

impl<T> FromGuts for ZeroCrossing<T> {
    fn from_guts(guts: Self::Guts) -> Self {
        let (config, state) = guts;
        Self { config, state }
    }
}

impl<T> IntoGuts for ZeroCrossing<T> {
    fn into_guts(self) -> Self::Guts {
        (self.config, self.state)
    }
}

impl<T> Reset for ZeroCrossing<T>
where
    T: Clone,
{
    fn reset(self) -> Self {
        Self::with_config(self.config)
    }
}

#[cfg(feature = "derive")]
impl<T> ResetMut for ZeroCrossing<T> where Self: Reset {}

impl<T> Filter<T> for ZeroCrossing<T>
where
    T: Signed + PartialOrd,
{
    type Output = bool;

    /// Returns `true` when a zero-crossing is detected, `false` otherwise.
    fn filter(&mut self, input: T) -> Self::Output {
        let crossing = match &self.state.prev {
            None => false,
            Some(prev) => {
                let prev_sign = prev.signum();
                let input_sign = input.signum();
                prev_sign != input_sign && input.abs() > self.config.hysteresis
            }
        };
        self.state.prev = Some(input);
        crossing
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use super::*;

    #[test]
    fn test_basic_zero_crossing_detection() {
        let filter = ZeroCrossing::with_config(Config { hysteresis: 0.0f32 });
        let inputs = [-1.0f32, -0.5, 0.5, 1.0, -1.0, -2.0];
        let outputs: Vec<_> = inputs
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        // Expected: [false (first), false (no crossing), true (crossing), false, true, false]
        assert_eq!(outputs[0], false); // First sample
        assert_eq!(outputs[1], false); // -1.0 to -0.5, same sign
        assert_eq!(outputs[2], true); // -0.5 to 0.5, crossing
        assert_eq!(outputs[3], false); // 0.5 to 1.0, same sign
        assert_eq!(outputs[4], true); // 1.0 to -1.0, crossing
        assert_eq!(outputs[5], false); // -1.0 to -2.0, same sign
    }

    #[test]
    fn test_hysteresis_prevents_spurious_crossings() {
        let filter = ZeroCrossing::with_config(Config { hysteresis: 0.3f32 });
        let inputs = [-1.0f32, 0.2, -0.2, 1.0];
        let outputs: Vec<_> = inputs
            .iter()
            .scan(filter, |filter, &input| Some(filter.filter(input)))
            .collect();

        // Expected: [false (first), false (0.2 < 0.3), false (−0.2 < 0.3), true (1.0 > 0.3)]
        assert_eq!(outputs[0], false); // First sample
        assert_eq!(outputs[1], false); // -1.0 to 0.2: crosses, but |0.2| < 0.3
        assert_eq!(outputs[2], false); // 0.2 to -0.2: crosses, but |-0.2| < 0.3
        assert_eq!(outputs[3], true); // -0.2 to 1.0: crosses and |1.0| > 0.3
    }

    #[test]
    fn test_reset() {
        let mut filter = ZeroCrossing::with_config(Config { hysteresis: 0.0f32 });
        let _ = filter.filter(-1.0f32);
        let _ = filter.filter(1.0f32); // Crossing detected

        let filter_reset = filter.reset();
        let outputs: Vec<_> = vec![-1.0f32, 1.0f32]
            .iter()
            .scan(filter_reset, |filter, &input| Some(filter.filter(input)))
            .collect();

        // After reset, first sample should not produce a crossing
        assert_eq!(outputs[0], false);
        assert_eq!(outputs[1], true); // Second sample should detect crossing
    }

    #[test]
    fn test_non_negative_sequence_no_crossing() {
        let filter = ZeroCrossing::with_config(Config { hysteresis: 0.0 });
        let input = [
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0,
        ];
        let output: Vec<_> = input
            .iter()
            .scan(filter, |f, &x| Some(f.filter(x)))
            .collect();
        // All non-negative, so signum() treats zero as positive: no crossings
        assert_eq!(output, vec![false; 20]);
    }

    #[test]
    fn test_zero_edge_cases() {
        // 0.0 -> -1.0: signum(0.0)=1, signum(-1.0)=-1, crossing detected
        let mut f = ZeroCrossing::with_config(Config { hysteresis: 0.0f32 });
        assert_eq!(f.filter(0.0f32), false);
        assert_eq!(f.filter(-1.0f32), true);

        // -1.0 -> 0.0: signum(-1.0)=-1, signum(0.0)=1, signs differ
        // but |0.0| > 0.0 is false → no crossing
        let mut f = ZeroCrossing::with_config(Config { hysteresis: 0.0f32 });
        assert_eq!(f.filter(-1.0f32), false);
        assert_eq!(f.filter(0.0f32), false);

        // 0.0 -> 0.0: signum(0.0)=1, signum(0.0)=1, no sign change
        let mut f = ZeroCrossing::with_config(Config { hysteresis: 0.0f32 });
        assert_eq!(f.filter(0.0f32), false);
        assert_eq!(f.filter(0.0f32), false);

        // -0.5 -> 0.0 -> -0.5: crossing on exit from zero only
        // (|0.0| > 0.0 is false, so entry to zero is not detected)
        let mut f = ZeroCrossing::with_config(Config { hysteresis: 0.0f32 });
        assert_eq!(f.filter(-0.5f32), false);
        assert_eq!(f.filter(0.0f32), false);
        assert_eq!(f.filter(-0.5f32), true);
    }
}
