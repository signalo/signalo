// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Constant value sources.

use signalo_traits::Source;

/// A source that returns an constant value on each call.
///
/// ### Example:
///
/// ```
/// # extern crate signalo_sources;
/// #
/// # fn main() {
/// use signalo_sources::constant::Constant;
/// let constant = Constant::new(42);
/// // ╭────╮  ╭────╮  ╭────╮  ╭────╮  ╭────╮
/// // │ 42 │─▶│ 42 │─▶│ 42 │─▶│ 42 │─▶│ 42 │─▶ ...
/// // ╰────╯  ╰────╯  ╰────╯  ╰────╯  ╰────╯
/// # }
///```
#[derive(Default, Clone, Debug)]
pub struct Constant<T> {
    value: T,
}

impl<T> Constant<T> {
    /// Creates a new `Constant` source for a given `value`.
    #[inline]
    pub fn new(value: T) -> Self {
        Constant { value }
    }
}

impl<T> From<T> for Constant<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self { value }
    }
}

impl<T> Source for Constant<T>
where
    T: Clone,
{
    type Output = T;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        Some(self.value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        const VALUE: usize = 42;
        const COUNT: usize = 3;
        let source = Constant::new(VALUE);
        let subject: Vec<_> = (0..COUNT)
            .scan(source, |source, _| source.source())
            .collect();
        let expected = vec![VALUE; COUNT];
        assert_eq!(subject, expected);
    }
}
