use std::ops::BitOr;

use filter::Filter;

#[derive(Clone, Debug)]
pub struct Pipe<T, U> {
    signal: T,
    filter: U,
}

impl<T, U> Pipe<T, U> {
    #[inline]
    pub fn new(signal: T, filter: U) -> Self {
        Pipe { signal, filter }
    }
}

impl<T, U, Rhs> BitOr<Rhs> for Pipe<T, U> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, filter: Rhs) -> Self::Output {
        Pipe::new(self, filter)
    }
}

impl<T, U, I> Filter<I> for Pipe<T, U>
where
    T: Filter<I>,
    U: Filter<T::Output>,
{
    type Output = U::Output;

    #[inline]
    fn apply(&mut self, input: I) -> Self::Output {
        self.filter.apply(self.signal.apply(input))
    }

    fn phase_shift(&self) -> isize {
        self.signal.phase_shift() + self.filter.phase_shift()
    }
}

impl<T, U> Iterator for Pipe<T, U>
where
    T: Iterator,
    U: Filter<T::Item>,
{
    type Item = U::Output;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.signal.next().map(|input| self.filter.apply(input))
    }
}
