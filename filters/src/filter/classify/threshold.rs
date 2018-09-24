// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::PartialOrd;

use generic_array::typenum::U2;
use generic_array::GenericArray;

use signalo_traits::filter::Filter;

use signalo_traits::Configurable;

/// The threshold filter's configuration.
#[derive(Clone, Debug)]
pub struct Config<T, U> {
    /// input threshold.
    pub threshold: T,
    /// [off, on] outputs.
    pub outputs: GenericArray<U, U2>,
}

/// A threshold filter.
#[derive(Clone, Debug)]
pub struct Threshold<T, U> {
    /// The filter's configuration.
    config: Config<T, U>,
}

impl<T, U> Threshold<T, U>
where
    U: Clone,
{
    /// Creates a new `Threshold` filter with given `threshold` and `outputs` (`[off, on]`).
    #[inline]
    pub fn new(config: Config<T, U>) -> Self {
        Threshold { config }
    }
}

impl<T, U> Configurable for Threshold<T, U> {
    type Config = Config<T, U>;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U> Filter<T> for Threshold<T, U>
where
    T: PartialOrd<T>,
    U: Clone,
{
    type Output = U;

    #[inline]
    fn filter(&mut self, input: T) -> Self::Output {
        let index = (input >= self.config.threshold) as usize;
        self.config.outputs[index].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filter::classify::Classification;

    #[test]
    fn test() {
        let filter = Threshold::new(Config {
            threshold: 10,
            outputs: u8::classes(),
        });
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
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
}
