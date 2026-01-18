// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Thresholding filters.

use core::cmp::PartialOrd;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Config as ConfigTrait, ConfigClone, ConfigRef, Filter, Reset, WithConfig,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The threshold filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, U> {
    /// input threshold.
    pub threshold: T,
    /// [off, on] outputs.
    pub outputs: [U; 2],
}

/// A threshold filter.
#[derive(Clone, Debug)]
pub struct Threshold<T, U> {
    config: Config<T, U>,
}

impl<T, U> ConfigTrait for Threshold<T, U> {
    type Config = Config<T, U>;
}

impl<T, U> WithConfig for Threshold<T, U> {
    type Output = Self;

    fn with_config(config: Self::Config) -> Self::Output {
        Self { config }
    }
}

impl<T, U> ConfigRef for Threshold<T, U> {
    fn config_ref(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> ConfigClone for Threshold<T, U>
where
    Config<T, U>: Clone,
{
    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl<T, U> HasGuts for Threshold<T, U> {
    type Guts = Config<T, U>;
}

impl<T, U> FromGuts for Threshold<T, U> {
    fn from_guts(guts: Self::Guts) -> Self {
        let config = guts;
        Self { config }
    }
}

impl<T, U> IntoGuts for Threshold<T, U> {
    fn into_guts(self) -> Self::Guts {
        self.config
    }
}

impl<T, U> Reset for Threshold<T, U> {
    fn reset(self) -> Self {
        self
    }
}

#[cfg(feature = "derive")]
impl<T, U> ResetMut for Threshold<T, U> where Self: Reset {}

impl<T, U> Filter<T> for Threshold<T, U>
where
    T: PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    fn filter(&mut self, input: T) -> Self::Output {
        let index: usize = (input >= self.config.threshold).into();
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
        let filter = Threshold::with_config(Config {
            threshold: 10,
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
            vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]
        );
    }

    #[test]
    fn test_from_guts() {
        let config = Config {
            threshold: 5,
            outputs: [10, 20],
        };
        let filter: Threshold<i32, i32> = FromGuts::from_guts(config);
        assert_eq!(filter.config.threshold, 5);
    }

    #[test]
    fn test_into_guts() {
        let config = Config {
            threshold: 5,
            outputs: [10, 20],
        };
        let filter = Threshold::with_config(config);
        let guts = filter.into_guts();
        assert_eq!(guts.threshold, 5);
        assert_eq!(guts.outputs, [10, 20]);
    }

    #[test]
    fn test_reset() {
        let config = Config {
            threshold: 5,
            outputs: [10, 20],
        };
        let filter = Threshold::with_config(config);
        let reset_filter = filter.reset();
        assert_eq!(reset_filter.config.threshold, 5);
    }

    #[test]
    fn test_threshold_at_boundary() {
        let mut filter = Threshold::with_config(Config {
            threshold: 10,
            outputs: [0, 1],
        });

        assert_eq!(filter.filter(9), 0);
        assert_eq!(filter.filter(10), 1);
        assert_eq!(filter.filter(11), 1);
    }

    #[test]
    fn test_string_outputs() {
        let mut filter = Threshold::with_config(Config {
            threshold: 5.0,
            outputs: ["low", "high"],
        });

        assert_eq!(filter.filter(3.0), "low");
        assert_eq!(filter.filter(7.0), "high");
        assert_eq!(filter.filter(5.0), "high");
    }
}
