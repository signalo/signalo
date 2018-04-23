use std::ops::BitOr;

use filter::pipe::Pipe;

#[derive(Clone, Debug)]
pub struct Source<I> {
    iter: I,
}

impl<I, Rhs> BitOr<Rhs> for Source<I> {
    type Output = Pipe<Self, Rhs>;

    #[inline]
    fn bitor(self, filter: Rhs) -> Self::Output {
        Pipe::new(self, filter)
    }
}

impl<I> From<I> for Source<I> where I: Iterator {
    fn from(iter: I) -> Self {
        Source { iter }
    }
}

impl<I> Iterator for Source<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
