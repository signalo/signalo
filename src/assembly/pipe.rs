use std::ops::BitOr;

use filter::Filter;

/// A `Pipe` is a simple container joining a pair of `Pipe`s
///
/// ```plain
/// ════════════ + ════════════
///  ╭────────╮  +  ╭────────╮
///  │ Filter │  +  │ Filter │
///  ╰────────╯  +  ╰────────╯
/// ════════════ + ════════════
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
    fn from(filters: (T, U)) -> Self {
        let (lhs, rhs) = filters;
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

impl<T, U, I> Filter<I> for Pipe<T, U>
where
    T: Filter<I>,
    U: Filter<T::Output>,
{
    type Output = U::Output;

    #[inline]
    fn filter(&mut self, input: I) -> Self::Output {
        self.rhs.filter(self.lhs.filter(input))
    }

    fn phase_shift(&self) -> isize {
        self.rhs.phase_shift() + self.lhs.phase_shift()
    }
}

// impl<T, U> Iterator for Pipe<T, U>
// where
//     T: Iterator,
//     U: Filter<T::Item>,
// {
//     type Item = U::Output;
//
//     #[inline]
//     fn next(&mut self) -> Option<Self::Item> {
//         self.lhs.next().map(|input| self.rhs.filter(input))
//     }
// }
