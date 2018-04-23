use std::ops::BitOr;

use filter::Pipe;

#[derive(Clone, Debug)]
pub struct Iter<I> {
    iter: I,
}

impl<I, Rhs> BitOr<Rhs> for Iter<I> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, filter: Rhs) -> Self::Output {
        Pipe::new(self, filter)
    }
}

impl<I> From<I> for Iter<I> where I: Iterator {
    fn from(iter: I) -> Self {
        Iter { iter }
    }
}

impl<I> Iterator for Iter<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
