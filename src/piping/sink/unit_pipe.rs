use std::ops::BitOr;

use piping::sink::Pipe;
use sink::Sink;

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
    sink: T,
}

impl<T> UnitPipe<T>
{
    #[inline]
    pub fn new(sink: T) -> Self {
        UnitPipe { sink }
    }
}

impl<T, Rhs> BitOr<Rhs> for UnitPipe<T> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, rhs: Rhs) -> Self::Output {
        Pipe::new(self, rhs)
    }
}

impl<T, I> Sink<I> for UnitPipe<T>
where
    T: Sink<I>,
{
    type Output = T::Output;

    #[inline]
    fn sink(&mut self, input: I) {
        self.sink.sink(input)
    }

    #[inline]
    fn finalize(self) -> Self::Output {
        self.sink.finalize()
    }
}
