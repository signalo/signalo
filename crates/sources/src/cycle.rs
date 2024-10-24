// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Cycle sources.

use signalo_traits::Source;

/// A source that repeats an auto-incremented value on each call.
///
/// ### Example:
///
/// ```
/// # fn main() {
/// use signalo_sources::from_iter::FromIter;
///
/// let iter = FromIter::from(vec![0, 1, 2]);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_sources::cycle::Cycle;
/// let cycle = Cycle::new(iter);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 0 │─▶│ 1 │─▶│ 2 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Cycle<S> {
    orig: S,
    inner: S,
}

impl<S> Cycle<S>
where
    S: Clone,
{
    /// Creates a new `Cycle` source for a given `initial` value and an `interval`.
    #[inline]
    pub fn new(orig: S) -> Self {
        let inner = orig.clone();
        Self { orig, inner }
    }
}

impl<S, T> Source for Cycle<S>
where
    S: Clone + Source<Output = T>,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        match self.inner.source() {
            None => {
                self.inner = self.orig.clone();
                self.inner.source()
            }
            y => y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::from_iter::FromIter;

    #[test]
    fn non_empty() {
        let input = vec![0, 1, 2, 3];
        let inner = FromIter::from(input);
        let source = Cycle::new(inner);
        let subject: Vec<_> = (0..6).scan(source, |source, _| source.source()).collect();
        let expected = vec![0, 1, 2, 3, 0, 1];
        assert_eq!(subject, expected);
    }
}
