// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::Ordering;

use signalo_traits::sink::Sink;

/// A sink that computes the max and max of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct Max<T> {
    max: Option<T>,
}

impl<T> Sink<T> for Max<T>
where
    T: Clone + PartialOrd,
{
    type Output = Option<T>;

    #[inline]
    fn sink(&mut self, input: T) {
        self.max = match &self.max {
            Some(ref max) => match max.partial_cmp(&input) {
                Some(Ordering::Less) => Some(input),
                _ => Some(max.clone()),
            },
            None => Some(input),
        };
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![
            0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7,
        ];
        let mut sink = Max::default();
        for input in input {
            sink.sink(input);
        }
        let max = sink.finalize().unwrap();
        assert_eq!(max, 20);
    }
}
