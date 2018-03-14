#![macro_use]

use std::ops::BitOr;

use filter::Filter;

#[macro_export]
macro_rules! impl_pipe {
    ($type:ident) => {
        impl<Rhs> BitOr<Rhs> for $type {
            type Output = Pipe<Self, Rhs>;

            #[inline]
            fn bitor(self, rhs: Rhs) -> Self::Output {
                Pipe::new(self, rhs)
            }
        }
    };
    ($type:ident<$($arg:ident),+>) => {
        impl<$($arg),+, Rhs> BitOr<Rhs> for $type<$($arg),+> {
            type Output = Pipe<Self, Rhs>;

            #[inline]
            fn bitor(self, rhs: Rhs) -> Self::Output {
                Pipe::new(self, rhs)
            }
        }
    };
}

#[derive(Clone, Debug)]
pub struct Pipe<T, U> {
    lhs: T,
    rhs: U,
}

impl<T, U> Pipe<T, U> {
    #[inline]
    pub fn new(lhs: T, rhs: U) -> Self {
        Pipe { lhs, rhs }
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
        self.rhs.apply(self.lhs.apply(input))
    }

    fn phase_shift(&self) -> isize {
        self.lhs.phase_shift() + self.rhs.phase_shift()
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
        self.lhs.next().map(|input| self.rhs.apply(input))
    }
}
