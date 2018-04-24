use std::ops::BitOr;

use source::Source;
use piping::source::Pipe;

/// A `UnitPipe` is a simple container wrapping a `Source`
///
/// ```plain
/// ╠════════════
/// ║ ╭────────╮
/// ║ │ Source │
/// ║ ╰────────╯
/// ╠════════════
/// └─┬────────┘
///   └ UnitPipe
/// ```
#[derive(Clone, Debug)]
pub struct UnitPipe<T> {
    source: T,
}

impl<T> UnitPipe<T>
{
    #[inline]
    pub fn new(source: T) -> Self {
        Self { source }
    }
}

impl<T, Rhs> BitOr<Rhs> for UnitPipe<T> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, rhs: Rhs) -> Self::Output {
        Pipe::new(self, rhs)
    }
}

impl<T> Source for UnitPipe<T>
where
    T: Source,
{
    type Output = T::Output;

    #[inline]
    fn source(&mut self) -> Option<Self::Output> {
        self.source.source()
    }
}

impl<T> Iterator for UnitPipe<T>
where
    T: Source,
{
    type Item = T::Output;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.source.source()
    }
}
