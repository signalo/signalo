// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Skipping sources.

use signalo_traits::Source;

/// A source that returns only up to a specified number of values.
/// A source that returns an auto-incremented value on each call.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_sources;
/// #
/// # fn main() {
/// use signalo_sources::increment::Increment;
///
/// let increment = Increment::new(0, 1);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 3 │─▶│ 4 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_sources::skip::Skip;
///
/// let skip = Skip::new(increment, 2);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 2 │─▶│ 3 │─▶│ 4 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯
/// # }
///```
#[derive(Clone, Debug)]
pub struct Skip<S> {
    inner: S,
    count: usize,
}

impl<S> Skip<S> {
    /// Creates a new `Skip` source for a given `value`.
    #[inline]
    pub fn new(inner: S, count: usize) -> Self {
        Self { inner, count }
    }
}

impl<S, T> Source for Skip<S>
where
    S: Source<Output = T>,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        while (self.count > 0) && (self.inner.source().is_some()) {
            self.count -= 1;
        }
        self.count = 0;
        self.inner.source()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use from_iter::FromIter;

    #[test]
    fn test() {
        let inner = FromIter::from(vec![0, 1, 2, 3, 4, 5]);
        let mut source = Skip::new(inner, 3);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![3, 4, 5];
        assert_eq!(subject, expected);
    }
}
