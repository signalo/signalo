use std::{
    iter::FromIterator,
    mem::{self, MaybeUninit},
    ptr,
};

#[derive(Debug)]
pub struct CircularBuffer<T, const N: usize> {
    array: [MaybeUninit<T>; N],
    read: usize,
    write: usize,
}

impl<T, const N: usize> Clone for CircularBuffer<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        // FIXME: use `uninit_array` instead, once stable:
        let mut array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

        let capacity = Self::capacity();

        for index in self.read..self.write {
            let index = index % capacity;
            let source = &self.array[index];
            let destination = &mut array[index];
            let item = unsafe { source.assume_init_ref() };
            destination.write(item.clone());
        }

        Self {
            array,
            read: self.read,
            write: self.write,
        }
    }
}

impl<T, const N: usize> Default for CircularBuffer<T, N> {
    fn default() -> Self {
        // FIXME: use `uninit_array` instead, once stable:
        let array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

        Self {
            array,
            read: 0,
            write: 0,
        }
    }
}

impl<T, const N: usize> FromIterator<T> for CircularBuffer<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::default();

        for value in iter {
            buffer.push_back(value);
        }

        buffer
    }
}

impl<T, const N: usize> IntoIterator for CircularBuffer<T, N> {
    type Item = <Self::IntoIter as Iterator>::Item;

    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { buffer: self }
    }
}

impl<T, const N: usize> Drop for CircularBuffer<T, N> {
    fn drop(&mut self) {
        let capacity = Self::capacity();
        for index in self.read..self.write {
            let maybe_uninit = &mut self.array[index % capacity];
            unsafe {
                // FIXME: use `maybe_uninit.assume_init_drop()` once stabilized.
                ptr::drop_in_place(maybe_uninit.as_mut_ptr());
            }
        }
    }
}

impl<T, const N: usize> CircularBuffer<T, N> {
    pub fn push_back(&mut self, value: T) -> Option<T> {
        let result = if self.is_full() {
            self.pop_front()
        } else {
            None
        };

        let capacity = Self::capacity();
        let index = self.write % capacity;

        self.write += 1;

        self.array[index] = MaybeUninit::new(value);

        assert!(self.read <= self.write);

        result
    }

    #[must_use]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let capacity = Self::capacity();
        let index = self.read % capacity;

        self.read += 1;

        let slot = &mut self.array[index];
        let maybe_uninit = mem::replace(slot, MaybeUninit::uninit());
        let value = unsafe { maybe_uninit.assume_init() };

        assert!(self.read <= self.write);

        Some(value)
    }

    pub fn is_empty(&self) -> bool {
        self.read == self.write
    }

    pub fn is_full(&self) -> bool {
        self.len() == Self::capacity()
    }

    pub fn len(&self) -> usize {
        self.write - self.read
    }

    pub const fn capacity() -> usize {
        N
    }

    pub fn iter(&self) -> Iter<'_, T, N> {
        Iter {
            start: self.read,
            end: self.write,
            buffer: self,
        }
    }
}

pub struct Iter<'a, T, const N: usize> {
    start: usize,
    end: usize,
    buffer: &'a CircularBuffer<T, N>,
}

impl<'a, T, const N: usize> Iterator for Iter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            return None;
        }

        let capacity = CircularBuffer::<T, N>::capacity();
        let index = self.start;

        self.start += 1;

        let value = {
            let maybe_uninit = &self.buffer.array[index % capacity];
            unsafe { maybe_uninit.assume_init_ref() }
        };

        Some(value)
    }
}

pub struct IntoIter<T, const N: usize> {
    buffer: CircularBuffer<T, N>,
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_input() -> Vec<f32> {
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        vec![0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0]
    }

    fn get_output() -> Vec<Option<f32>> {
        vec![
            None,
            None,
            None,
            Some(0.0),
            Some(1.0),
            Some(7.0),
            Some(2.0),
            Some(5.0),
        ]
    }

    #[test]
    fn test() {
        // Effectively delays input by length of buffer:
        let buffer: CircularBuffer<f32, 3> = CircularBuffer::default();
        let input = get_input();
        let output: Vec<_> = input
            .iter()
            .scan(buffer, |buffer, &input| Some(buffer.push_back(input)))
            .collect();
        assert_eq!(output, get_output());
    }

    #[test]
    fn pop_front() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();

        assert_eq!(buffer.pop_front(), None);

        buffer.push_back(42.0);
        assert_eq!(buffer.pop_front(), Some(42.0));

        assert_eq!(buffer.pop_front(), None);
    }

    #[test]
    fn is_empty() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();

        assert_eq!(buffer.is_empty(), true);

        buffer.push_back(42.0);
        assert_eq!(buffer.is_empty(), false);

        let _ = buffer.pop_front();
        assert_eq!(buffer.is_empty(), true);
    }

    #[test]
    fn is_full() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();

        assert_eq!(buffer.is_full(), false);

        buffer.push_back(1.0);
        assert_eq!(buffer.is_full(), false);

        buffer.push_back(2.0);
        assert_eq!(buffer.is_full(), false);

        buffer.push_back(3.0);
        assert_eq!(buffer.is_full(), true);

        let _ = buffer.pop_front();
        assert_eq!(buffer.is_full(), false);
    }

    #[test]
    fn len() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();

        assert_eq!(buffer.len(), 0);

        buffer.push_back(1.0);
        assert_eq!(buffer.len(), 1);

        buffer.push_back(2.0);
        assert_eq!(buffer.len(), 2);

        buffer.push_back(3.0);
        assert_eq!(buffer.len(), 3);

        let _ = buffer.pop_front();
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn capacity() {
        assert_eq!(CircularBuffer::<f32, 3>::capacity(), 3);
    }

    #[test]
    fn from_iter() {
        let mut buffer: CircularBuffer<f32, 3> =
            vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();

        assert_eq!(buffer.pop_front(), Some(3.0));

        assert_eq!(buffer.pop_front(), Some(4.0));

        assert_eq!(buffer.pop_front(), Some(5.0));
    }

    #[test]
    fn iter() {
        let buffer: CircularBuffer<f32, 3> = vec![1.0, 2.0, 3.0].into_iter().collect();

        let items: Vec<&f32> = buffer.iter().collect();
        assert_eq!(items, vec![&1.0, &2.0, &3.0]);
    }

    #[test]
    fn into_iter() {
        let buffer: CircularBuffer<f32, 3> = vec![1.0, 2.0, 3.0].into_iter().collect();

        let items: Vec<f32> = buffer.into_iter().collect();
        assert_eq!(items, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn clone() {
        let buffer: CircularBuffer<f32, 3> = vec![1.0, 2.0].into_iter().collect();

        let clone = buffer.clone();

        assert_eq!(buffer.read, clone.read);
        assert_eq!(buffer.write, clone.write);

        for (original, clone) in buffer.iter().zip(clone.iter()) {
            assert_eq!(original, clone);
        }
    }

    #[test]
    fn drop() {
        use testdrop::{Item, TestDrop};

        let td = TestDrop::new();

        let buffer: CircularBuffer<Item, 5> = (0..3).map(|_| td.new_item().1).collect();

        std::mem::drop(buffer);

        assert_eq!(3, td.num_tracked_items());
        assert_eq!(3, td.num_dropped_items());
    }
}
