// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sequential chaining of two sources in series.
//!
//! Combines two sources such that the second source begins generating values after the first
//! source is exhausted, useful for concatenating signal sequences.

use crate::traits::Source;

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum ChainState {
    // source values from front
    Front,
    // source values from back
    Back,
}

/// A source that drains its front source, then drains its back source.
///
/// ### Example:
///
/// ```
/// # fn main() {
/// use signalo::sources::from_iter::FromIter;
/// let front = FromIter::from(vec![0, 1, 2]);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
/// let back = FromIter::from(vec![3, 4]);
/// // ╭───╮  ╭───╮
/// // │ 3 │─▶│ 4 │
/// // ╰───╯  ╰───╯
///
/// use signalo::sources::chain::Chain;
/// let chain = Chain::new(front, back);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 3 │─▶│ 4 │
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
/// # }
/// ```
///
/// # Complexity
///
/// - **Time per sample:** same as the active source (`F` or `B`); O(1) for the dispatch itself.
/// - **Space:** same as `F` plus `B`; O(1) for the internal state flag.
#[derive(Clone, Debug)]
pub struct Chain<F, B> {
    front: F,
    back: B,
    state: ChainState,
}

impl<F, B> Chain<F, B> {
    /// Creates a new `Chain` source from a given pair of sources.
    #[inline]
    pub fn new(front: F, back: B) -> Self {
        let state = ChainState::Front;
        Self { front, back, state }
    }
}

impl<F, B, T> Source for Chain<F, B>
where
    F: Source<Output = T>,
    B: Source<Output = T>,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        match self.state {
            ChainState::Front => {
                if let Some(value) = self.front.source() {
                    Some(value)
                } else {
                    self.state = ChainState::Back;
                    self.back.source()
                }
            }
            ChainState::Back => self.back.source(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::vec;
    use std::vec::Vec;

    use crate::sources::from_iter::FromIter;

    use super::*;

    #[test]
    fn head_empty() {
        let head = FromIter::from(vec![]);
        let tail = FromIter::from(vec![3, 4]);
        let mut source = Chain::new(head, tail);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![3, 4];
        assert_eq!(subject, expected);
    }

    #[test]
    fn tail_empty() {
        let head = FromIter::from(vec![0, 1, 2]);
        let tail = FromIter::from(vec![]);
        let mut source = Chain::new(head, tail);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![0, 1, 2];
        assert_eq!(subject, expected);
    }

    #[test]
    fn non_empty() {
        let head = FromIter::from(vec![0, 1, 2]);
        let tail = FromIter::from(vec![3, 4]);
        let mut source = Chain::new(head, tail);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![0, 1, 2, 3, 4];
        assert_eq!(subject, expected);
    }
}
