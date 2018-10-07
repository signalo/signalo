// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! "Last value" sinks.

use signalo_traits::Sink;

/// A sink that memorizes the most recently received value of a signal.
#[derive(Default, Clone, Debug)]
pub struct Last<T> {
    state: Option<T>,
}

impl<T> Last<T> {
    /// Creates a new `Last` sink.
    #[inline]
    pub fn new() -> Self {
        Last { state: None }
    }
}

impl<T> Sink<T> for Last<T> {
    type Output = Option<T>;

    #[inline]
    fn sink(&mut self, input: T) {
        self.state = Some(input);
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.state
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
        let mut sink = Last::new();
        for input in input {
            sink.sink(input);
        }
        let subject = sink.finalize();
        assert_eq!(subject, Some(7));
    }
}
