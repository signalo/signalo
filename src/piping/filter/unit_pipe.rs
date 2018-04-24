use std::ops::BitOr;

use filter::Filter;
use piping::filter::Pipe;

/// A `UnitPipe` is a simple container wrapping a simple `Filter`
///
/// ```plain
/// ════════════
///  ╭────────╮
///  │ Filter │
///  ╰────────╯
/// ════════════
/// └─┬────────┘
///   └ UnitPipe
/// ```
#[derive(Clone, Debug)]
pub struct UnitPipe<T> {
    filter: T,
}

impl<T> UnitPipe<T>
{
    #[inline]
    pub fn new(filter: T) -> Self {
        UnitPipe { filter }
    }
}

impl<T, Rhs> BitOr<Rhs> for UnitPipe<T> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, rhs: Rhs) -> Self::Output {
        Pipe::new(self, rhs)
    }
}

impl<T, I> Filter<I> for UnitPipe<T>
where
    T: Filter<I>,
{
    type Output = T::Output;

    #[inline]
    fn filter(&mut self, input: I) -> Self::Output {
        self.filter.filter(input)
    }

    #[inline]
    fn phase_shift(&self) -> isize {
        self.filter.phase_shift()
    }
}
