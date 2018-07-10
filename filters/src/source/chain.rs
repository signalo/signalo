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

    use source::Iter;

    #[test]
    fn test() {
        let head = Iter::from(vec![0, 1, 2]);
        let tail = Iter::from(vec![3, 4]);
        let source = Chain::new(head, tail);
        let subject: Vec<_> = (0..5).scan(source, |source, _| {
            source.source()
        }).collect();
        let expected = vec![0, 1, 2, 3, 4];
        assert_eq!(subject, expected);
    }
}
