// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo_traits::source::Source;

/// A wrapper type for turning iterators into sources.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_filters;
/// #
/// # fn main() {
/// use signalo_filters::prelude::Source;
/// use signalo_filters::prelude::Sink;
///
/// use signalo_filters::source::Increment;
/// let increment: Increment<_> = Increment::new(0, 1);
/// // ╭───╮  ╭───╮  ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │─▶│ 3 │─▶│ 4 │─▶ ...
/// // ╰───╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_filters::source::Take;
/// let mut take: Take<_> = Take::new(increment, 3);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
///
/// use signalo_filters::source::IntoIter;
/// let iter = IntoIter::from(take);
/// // ╭───╮  ╭───╮  ╭───╮
/// // │ 0 │─▶│ 1 │─▶│ 2 │
/// // ╰───╯  ╰───╯  ╰───╯
///
/// let vec: Vec<_> = iter.collect();
/// # }
/// ```
#[derive(Default, Clone, Debug)]
pub struct IntoIter<S> {
    source: S,
}

impl<S, T> From<S> for IntoIter<S>
where
    S: Source<Output=T>,
{
    #[inline]
    fn from(source: S) -> Self {
        Self { source }
    }
}

impl<S, T> Iterator for IntoIter<S>
where
    S: Source<Output=T>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.source.source()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use source::FromIter;

    #[test]
    fn test() {
        let input = vec![0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 14, 9, 9, 17, 17, 4, 12, 20, 20, 7];
        let source = FromIter::from(input.clone());
        let iter = IntoIter::from(source);
        let subject: Vec<_> = iter.collect();
        let expected = input;
        assert_eq!(subject, expected);
    }
}
