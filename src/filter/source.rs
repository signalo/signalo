use std::ops::BitOr;

use filter::pipe::Pipe;

#[derive(Clone)]
pub struct Source<I> {
    iter: I,
}

impl_pipe!(Source<I>);

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
