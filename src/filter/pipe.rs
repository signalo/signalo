#![macro_use]

use std::ops::BitOr;

use filter::Filter;

#[macro_export]
macro_rules! impl_pipe {
    ($type:ident) => {
        impl<Rhs> BitOr<Rhs> for $type {
            type Output = Pipe<Self, Rhs>;

            #[inline]
            fn bitor(self, filter: Rhs) -> Self::Output {
                Pipe::new(self, filter)
            }
        }
    };
    ($type:ident<$($arg:ident),+>) => {
        impl<$($arg),+, Rhs> BitOr<Rhs> for $type<$($arg),+> {
            type Output = Pipe<Self, Rhs>;

            #[inline]
            fn bitor(self, filter: Rhs) -> Self::Output {
                Pipe::new(self, filter)
            }
        }
    };
}

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

impl_pipe!(Pipe<T, U>);

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
