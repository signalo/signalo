// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo_traits::source::Source;

use source::Repeat;

#[derive(Clone, Debug)]
enum PadState<T> {
    // source values from front
    Before,
    // source values from front
    Front(Repeat<T>),
    // source values from inner
    Inner,
    // source values from back
    Back(Repeat<T>),
    // reached end of back padding
    After,
}

/// A source that pads an inner source with a specified number of constant values at the edges.
#[derive(Clone, Debug)]
pub struct Pad<S, T> {
    inner: S,
    count: usize,
    state: PadState<T>,
}

impl<S, T> Pad<S, T>
where
    T: Clone,
{
    /// Creates a new `Pad` source from an inner source and specified padding.
    #[inline]
    pub fn new(inner: S, count: usize) -> Self {
        let state = PadState::Before;
        Pad { inner, count, state }
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
        let (state, output) = match self.state {
            PadState::Before => {
                if let Some(value) = self.inner.source() {
                    let front = Repeat::new(value.clone(), self.count);
                    (Some(PadState::Front(front)), Some(value))
                } else {
                    (Some(PadState::After), None)
                }
            },
            PadState::Front(ref mut front) => {
                if let Some(value) = front.source() {
                    (None, Some(value))
                } else {
                    (Some(PadState::Inner), self.inner.source())
                }
            },
            PadState::Inner => {
                if let Some(value) = self.inner.source() {
                    let back = Repeat::new(value.clone(), self.count);
                    (Some(PadState::Back(back)), Some(value))
                } else {
                    (Some(PadState::After), None)
                }
            },
            PadState::Back(ref mut back) => {
                if let Some(value) = back.source() {
                    (None, Some(value))
                } else {
                    (Some(PadState::After), None)
                }
            },
            PadState::After => {
                (None, None)
            },
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

    use source::Iter;

    #[test]
    fn test() {
        let inner = Iter::from(vec![0, 1, 2]);
        let source = Pad::new(inner, 2);
        let subject: Vec<_> = (0..7).scan(source, |source, _| {
            source.source()
        }).collect();
        let expected = vec![0, 0, 0, 1, 2, 2, 2];
        assert_eq!(subject, expected);
    }
}
