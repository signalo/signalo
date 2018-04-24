use std::ops::BitOr;

use filter::Filter;
use sink::Sink;

/// A `Pipe` is a simple container joining a pair of a `Filter` and a `Sink`
///
/// ```plain
/// ╠════════════ + ════════════
/// ║ ╭────────╮  +  ╭────────╮
/// ║ │ Filter │  +  │  Sink  │
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
        Pipe { lhs, rhs }
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

impl<T, U, I> Sink<I> for Pipe<T, U>
where
    T: Filter<I>,
    U: Sink<T::Output>,
{
    type Output = U::Output;

    #[inline]
    fn sink(&mut self, input: I) {
        self.rhs.sink(self.lhs.filter(input))
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.rhs.finalize()
    }
}
