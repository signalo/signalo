// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo_traits::sink::Sink;

use sink::{Max, Min};

/// A sink that computes the min and max of all received values of a signal.
#[derive(Clone, Default, Debug)]
pub struct Bounds<T> {
    min: Min<T>,
    max: Max<T>,
}

impl<T> Sink<T> for Bounds<T>
where
    T: Copy + PartialOrd,
{
    type Output = Option<(T, T)>;

    #[inline]
    fn sink(&mut self, input: T) {
        self.min.sink(input);
        self.max.sink(input);
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        let min = self.min.finalize();
        let max = self.max.finalize();
        match (min, max) {
            (Some(min), Some(max)) => Some((min, max)),
            (None, None) => None,
            (Some(_), None) => unreachable!(),
            (None, Some(_)) => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let mut sink = Bounds::default();
        for input in input {
            sink.sink(input);
        }
        let (min, max) = sink.finalize().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 20);
    }
}
