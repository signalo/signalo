// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Edge padding sources.

use signalo_traits::Source;

use repeat::Repeat;

#[derive(Clone, Debug)]
enum PadState<T> {
    // source values from front
    Before,
    // source values from front
    Front(Repeat<T>),
    // source values from inner
    Inner(T),
    // source values from back
    Back(Repeat<T>),
    // reached end of back padding
    After,
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
/// use signalo_sources::pad::edge::Pad;
/// let pad_edge = Pad::new(take, 2);
/// // ╭────╮  ╭────╮  ╭───╮  ╭───╮  ╭───╮  ╭────╮  ╭────╮
/// // │ 42 │─▶│ 42 │─▶│ 0 │─▶│ 1 │─▶│ 2 │─▶│ 42 │─▶│ 42 │
/// // ╰────╯  ╰────╯  ╰───╯  ╰───╯  ╰───╯  ╰────╯  ╰────╯
/// # }
///```
#[derive(Clone, Debug)]
pub struct Pad<S, T> {
    inner: S,
    count: usize,
    state: PadState<T>,
}

impl<S, T> Pad<S, T>
where
    S: Source<Output = T>,
    T: Clone,
{
    /// Creates a new `Pad` source from an inner source and specified padding.
    #[inline]
    pub fn new(inner: S, count: usize) -> Self {
        let state = PadState::Before;
        Pad {
            inner,
            count,
            state,
        }
    }
}

impl<S, T> Source for Pad<S, T>
where
    S: Source<Output = T>,
    T: Clone,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        let (state, output) = match self.state {
            PadState::Before => {
                if let Some(value) = self.inner.source() {
                    let front = Repeat::new(value.clone(), self.count);
                    (Some(PadState::Front(front)), Some(value))
                } else {
                    (Some(PadState::After), None)
                }
            }
            PadState::Front(ref mut front) => {
                if let Some(value) = front.source() {
                    (None, Some(value))
                } else if let Some(value) = self.inner.source() {
                    (Some(PadState::Inner(value.clone())), Some(value))
                } else {
                    (Some(PadState::After), None)
                }
            }
            PadState::Inner(ref value) => {
                if let Some(value) = self.inner.source() {
                    (Some(PadState::Inner(value.clone())), Some(value))
                } else {
                    let count = if self.count > 0 { self.count - 1 } else { 0 };
                    let back = Repeat::new(value.clone(), count);
                    (Some(PadState::Back(back)), Some(value.clone()))
                }
            }
            PadState::Back(ref mut back) => {
                if let Some(value) = back.source() {
                    (None, Some(value))
                } else {
                    (Some(PadState::After), None)
                }
            }
            PadState::After => (None, None),
        };
        if let Some(state) = state {
            self.state = state;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use from_iter::FromIter;

    #[test]
    fn empty() {
        let inner = FromIter::from(vec![]);
        let mut source = Pad::new(inner, 2);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![];
        assert_eq!(subject, expected);
    }

    #[test]
    fn non_empty() {
        let inner = FromIter::from(vec![0, 1, 2, 3, 4]);
        let mut source = Pad::new(inner, 2);
        let mut subject: Vec<usize> = vec![];
        while let Some(value) = source.source() {
            subject.push(value);
        }
        let expected = vec![0, 0, 0, 1, 2, 3, 4, 4, 4];
        assert_eq!(subject, expected);
    }
}
