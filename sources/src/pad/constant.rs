// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Constant padding sources.

use signalo_traits::Source;

use crate::repeat::Repeat;

#[derive(Clone, Debug)]
enum PadState {
    // source values from front
    Front,
    // source values from inner
    Inner,
    // source values from back
    Back,
}

/// A source that pads an inner source with a specified number of constant values at the edges.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_sources;
/// #
/// # fn main() {
/// use signalo_sources::increment::Increment;
/// let increment = Increment::new(0, 1);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 3 │─▶│ 4 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_sources::take::Take;
/// let take = Take::new(increment, 3);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_sources::pad::constant::Pad;
/// let pad_constant = Pad::new(take, 42, 2);
/// // ╭────╮  ╭────╮  ╭───╮  ╭───╮  ╭───╮  ╭────╮  ╭────╮
/// // │ 42 │─▶│ 42 │─▶│ 0 │─▶│ 1 │─▶│ 2 │─▶│ 42 │─▶│ 42 │
/// // ╰────╯  ╰────╯  ╰───╯  ╰───╯  ╰───╯  ╰────╯  ╰────╯
/// # }
///```
#[derive(Clone, Debug)]
pub struct Pad<S, T> {
    inner: S,
    front: Repeat<T>,
    back: Repeat<T>,
    state: PadState,
}

impl<S, T> Pad<S, T>
where
    T: Clone,
{
    /// Creates a new `Pad` source from an inner source and specified padding.
    #[inline]
    pub fn new(inner: S, value: T, count: usize) -> Self {
        let front = Repeat::new(value.clone(), count);
        let back = Repeat::new(value, count);
        let state = PadState::Front;
        Self {
            inner,
            front,
            back,
            state,
        }
    }
}

impl<S, T> Source for Pad<S, T>
where
    T: Clone,
    S: Source<Output = T>,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        match self.state {
            PadState::Front => {
                if let Some(output) = self.front.source() {
                    Some(output)
                } else {
                    self.state = PadState::Inner;
                    self.source()
                }
            }
            PadState::Inner => {
                if let Some(output) = self.inner.source() {
                    Some(output)
                } else {
                    self.state = PadState::Back;
                    self.source()
                }
            }
            PadState::Back => self.back.source(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::from_iter::FromIter;

    #[test]
    fn empty() {
        let inner = FromIter::from(vec![]);
        let mut source = Pad::new(inner, 42, 2);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![42, 42, 42, 42];
        assert_eq!(subject, expected);
    }

    #[test]
    fn non_empty() {
        let inner = FromIter::from(vec![0, 1, 2, 3, 4]);
        let mut source = Pad::new(inner, 42, 2);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![42, 42, 0, 1, 2, 3, 4, 42, 42];
        assert_eq!(subject, expected);
    }
}
