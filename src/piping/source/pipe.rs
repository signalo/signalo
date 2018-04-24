use std::ops::BitOr;

use source::Source;
use filter::Filter;

/// A `Pipe` is a simple container joining a pair of a `Source` and a `Filter`
///
/// ```plain
/// ╠════════════ + ════════════
/// ║ ╭────────╮  +  ╭────────╮
/// ║ │ Source │  +  │ Filter │
/// ║ ╰────────╯  +  ╰────────╯
/// ╠════════════ + ════════════
/// └─┬───────────────────────┘
///   └ Pipe
/// ```
#[derive(Clone, Debug)]
pub struct Pipe<T, U> {
    lhs: T,
    rhs: U,
}

impl<T, U> Pipe<T, U>
{
    #[inline]
    pub fn new(lhs: T, rhs: U) -> Self {
        Self { lhs, rhs }
    }
}

impl<T, U> From<(T, U)> for Pipe<T, U> {
    #[inline]
    fn from(parts: (T, U)) -> Self {
        let (lhs, rhs) = parts;
        Self::new(lhs, rhs)
    }
}

impl<T, U, Rhs> BitOr<Rhs> for Pipe<T, U> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, rhs: Rhs) -> Self::Output {
        Pipe::new(self, rhs)
    }
}

impl<T, U> Iterator for Pipe<T, U>
where
    T: Source,
    U: Filter<T::Output>,
{
    type Item = U::Output;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.lhs.source().map(|input| self.rhs.filter(input))
    }
}
