// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo_traits::source::Source;

/// A wrapper type for turning iterators into sources.
#[derive(Clone, Debug)]
pub struct Iter<I> {
    iter: I,
}

impl<I, J, T> From<J> for Iter<I>
where
    I: Iterator<Item=T>,
    J: IntoIterator<IntoIter=I, Item=T>,
{
    #[inline]
    fn from(into_iter: J) -> Self {
        Self { iter: into_iter.into_iter() }
    }
}

impl<I> Source for Iter<I>
where
    I: Iterator,
{
    type Output = I::Item;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        self.iter.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source() {
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let source = Iter::from(input.clone());
        let subject: Vec<_> = (0..input.len()).scan(source, |source, _| {
            source.source()
        }).collect();
        let expected = input;
        assert_eq!(subject, expected);
    }
}
