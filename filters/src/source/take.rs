// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo_traits::source::Source;

/// A source that returns only up to a specified number of values.
#[derive(Clone, Debug)]
pub struct Take<S> {
    inner: S,
    count: usize,
}

impl<S> Take<S> {
    /// Creates a new `Take` source for a given `value`.
    #[inline]
    pub fn new(inner: S, count: usize) -> Self {
        Take { inner, count }
    }
}

impl<S, T> Source for Take<S>
where
    S: Source<Output = T>,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        if self.count != 0 {
            self.count -= 1;
            self.inner.source()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        use source::Constant;

        const VALUE: f32 = 4.2;
        const COUNT: usize = 3;

        const EXCESS_COUNT: usize = COUNT + 10;

        let constant = Constant::new(VALUE);
        let source = Take::new(constant, COUNT);
        let subject: Vec<_> = (0..EXCESS_COUNT).scan(source, |source, _| {
            source.source()
        }).collect();
        let expected = vec![VALUE; COUNT];
        assert_eq!(subject, expected);
    }
}
