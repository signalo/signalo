// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::cmp::Ordering;

use signalo_traits::sink::Sink;

/// A sink that computes the min and max of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct Min<T> {
    min: Option<T>,
}

impl<T> Sink<T> for Min<T>
where
    T: Copy + PartialOrd,
{
    type Output = Option<T>;

    #[inline]
    fn sink(&mut self, input: T) {
        self.min = match self.min {
            Some(min) => match min.partial_cmp(&input) {
                Some(Ordering::Greater) => Some(input),
                _ => Some(min),
            },
            None => Some(input),
        };
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.min
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
        let mut sink = Min::default();
        for input in input {
            sink.sink(input);
        }
        let min = sink.finalize().unwrap();
        assert_eq!(min, 0);
    }
}
