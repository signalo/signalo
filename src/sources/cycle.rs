// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Cyclic buffering source repeating a fixed sequence.
//!
//! Cycles through a predefined array of values repeatedly, useful for generating periodic
//! waveforms and test patterns.

use crate::traits::Source;

/// A source that cycles through an inner source, resetting it when exhausted.
///
/// # Complexity
///
/// - **Time per sample:** same as the inner source `S`; one clone of `S` per cycle reset.
/// - **Space:** O(2 × |S|); stores the original and the live copy of the inner source.
///
/// ### Example:
///
/// ```
/// # fn main() {
/// use signalo::sources::from_iter::FromIter;
///
/// let iter = FromIter::from(vec![0, 1, 2]);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
///
/// use signalo::sources::cycle::Cycle;
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
    use std::vec;
    use std::vec::Vec;

    use crate::sources::from_iter::FromIter;
    use crate::traits::Source;

    use super::*;

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
