// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo_traits::source::Source;

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum ChainState {
    // source values from front
    Front,
    // source values from back
    Back,
}

/// A source that returns only a specified number of values.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_filters;
/// #
/// # fn main() {
/// use signalo_filters::source::FromIter;
/// let front = FromIter::from(vec![0, 1, 2]);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
/// let back = FromIter::from(vec![3, 4]);
/// // ╭───╮  ╭───╮
/// // │ 3 │─▶│ 4 │
/// // ╰───╯  ╰───╯
///
/// use signalo_filters::source::Chain;
/// let chain = Chain::new(front, back);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 3 │─▶│ 4 │
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
/// # }
/// ```
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
        Chain { front, back, state }
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
            ChainState::Front => match self.front.source() {
                value @ Some(..) => value,
                None => {
                    self.state = ChainState::Back;
                    self.back.source()
                }
            },
            ChainState::Back => self.back.source(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use source::FromIter;

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
