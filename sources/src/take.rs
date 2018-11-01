// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Trimming sources.

use signalo_traits::Source;

/// A source that returns only up to a specified number of values.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_sources;
/// #
/// # fn main() {
/// use signalo_sources::increment::Increment;
///
/// let increment: Increment<_> = Increment::new(0, 1);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 3 │─▶│ 4 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_sources::take::Take;
///
/// let take: Take<_> = Take::new(increment, 3);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
/// # }
///```
#[derive(Clone, Debug)]
pub struct Take<S> {
    inner: S,
    count: usize,
}

impl<S> Take<S> {
    /// Creates a new `Take` source for a given `value`.
    #[inline]
    pub fn new(inner: S, count: usize) -> Self {
        Self { inner, count }
    }
}

impl<S, T> Source for Take<S>
where
    S: Source<Output = T>,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        self.inner.source()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        use increment::Increment;

        let increment = Increment::new(0, 2);
        let mut source = Take::new(increment, 5);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![0, 2, 4, 6, 8];
        assert_eq!(subject, expected);
    }
}
