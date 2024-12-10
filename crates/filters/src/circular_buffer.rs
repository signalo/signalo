// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use core::{
    iter::FromIterator,
    mem::{self, MaybeUninit},
};

#[derive(Debug)]
pub struct CircularBuffer<T, const N: usize> {
    array: [MaybeUninit<T>; N],
    // The index of the buffer's "front" item.
    start: usize,
    // The index after the buffer's "back" item.
    end: usize,
}

impl<T, const N: usize> Clone for CircularBuffer<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        // FIXME: use `uninit_array` instead, once stable:
        // https://github.com/rust-lang/rust/issues/96097
        let mut array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

        let capacity = Self::capacity();

        for index in self.start..self.end {
            let index = index % capacity;
            let source = &self.array[index];
            let destination = &mut array[index];
            let item = unsafe { source.assume_init_ref() };
            destination.write(item.clone());
        }

        Self {
            array,
            start: self.start,
            end: self.end,
        }
    }
}

impl<T, const N: usize> Default for CircularBuffer<T, N> {
    fn default() -> Self {
        // FIXME: use `uninit_array` instead, once stable:
        // https://github.com/rust-lang/rust/issues/96097
        let array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

        Self {
            array,
            start: 0,
            end: 0,
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
        for index in self.start..self.end {
            let maybe_uninit = &mut self.array[index % capacity];
            unsafe {
                maybe_uninit.assume_init_drop();
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

        let index = self.end_index();
        self.array[index] = MaybeUninit::new(value);

        debug_assert!(self.start <= self.end);

        if self.end < usize::MAX {
            self.end += 1;
        } else {
            let delta = self.end - self.start + 1;

            // Jump to largest possible `end` index (within capacity) on overflow:
            self.end = Self::capacity();
            // Shift `start` index accordingly:
            self.start = self.end - delta;
        }

        debug_assert!(self.start < self.end);

        result
    }

    pub fn push_front(&mut self, value: T) -> Option<T> {
        let result = if self.is_full() {
            self.pop_back()
        } else {
            None
        };

        debug_assert!(self.start <= self.end);

        if self.start > usize::MIN {
            self.start -= 1;
        } else {
            let delta = self.end - self.start + 1;

            // Jump to smallest possible `start` index (within capacity) on overflow:
            self.start = 0;
            // Shift `end` index accordingly:
            self.end = self.start + delta;
        }

        debug_assert!(self.start < self.end);

        let index = self.start_index();
        self.array[index] = MaybeUninit::new(value);

        result
    }

    pub fn pop_front(&mut self) -> Option<T> {
        let Some(index) = self.front_index() else {
            return None;
        };

        let slot = &mut self.array[index];
        let maybe_uninit = mem::replace(slot, MaybeUninit::uninit());
        let value = unsafe { maybe_uninit.assume_init() };

        assert!(self.start < self.end);

        self.start += 1;

        Some(value)
    }

    pub fn pop_back(&mut self) -> Option<T> {
        let Some(index) = self.back_index() else {
            return None;
        };

        let slot = &mut self.array[index];
        let maybe_uninit = mem::replace(slot, MaybeUninit::uninit());
        let value = unsafe { maybe_uninit.assume_init() };

        assert!(self.start < self.end);

        self.end -= 1;

        Some(value)
    }

    pub fn back(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let capacity = Self::capacity();

        if capacity == 0 {
            return None;
        }

        let index = (self.end - 1) % capacity;
        let maybe_uninit = &self.array[index];
        let value = unsafe { maybe_uninit.assume_init_ref() };

        Some(value)
    }

    pub fn front(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let capacity = Self::capacity();

        if capacity == 0 {
            return None;
        }

        let index = self.start % capacity;
        let maybe_uninit = &self.array[index];
        let value = unsafe { maybe_uninit.assume_init_ref() };

        Some(value)
    }

    pub fn is_empty(&self) -> bool {
        (Self::capacity() == 0) || (self.start == self.end)
    }

    pub fn is_full(&self) -> bool {
        (Self::capacity() == 0) || (self.len() == Self::capacity())
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub const fn capacity() -> usize {
        N
    }

    pub fn iter(&self) -> Iter<'_, T, N> {
        Iter {
            start: self.start,
            end: self.end,
            buffer: self,
        }
    }

    fn start_index(&self) -> usize {
        self.start % Self::capacity()
    }

    fn end_index(&self) -> usize {
        self.end % Self::capacity()
    }

    fn front_index(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            Some(self.start_index())
        }
    }

    fn back_index(&self) -> Option<usize> {
        if self.is_empty() {
            None
        } else {
            let end_index = self.end_index();

            if end_index > 0 {
                Some(end_index - 1)
            } else {
                Some(Self::capacity() - 1)
            }
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.buffer.len();
        (len, Some(len))
    }

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.pop_front()
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.buffer.pop_back()
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}

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
    fn push_front() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();
        assert_eq!(buffer.len(), 0);

        buffer.push_front(1.0);
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.front(), Some(&1.0));

        buffer.push_front(2.0);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.front(), Some(&2.0));

        buffer.push_front(3.0);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.front(), Some(&3.0));

        buffer.push_front(4.0);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.front(), Some(&4.0));

        buffer.push_front(5.0);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.front(), Some(&5.0));
    }

    #[test]
    fn push_back() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();
        assert_eq!(buffer.len(), 0);

        buffer.push_back(1.0);
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.back(), Some(&1.0));

        buffer.push_back(2.0);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.back(), Some(&2.0));

        buffer.push_back(3.0);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.back(), Some(&3.0));

        buffer.push_back(4.0);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.back(), Some(&4.0));

        buffer.push_back(5.0);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.back(), Some(&5.0));
    }

    #[test]
    fn pop_front() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();

        assert_eq!(buffer.front(), None);
        assert_eq!(buffer.pop_front(), None);

        buffer.push_back(1.0);
        buffer.push_back(2.0);

        assert_eq!(buffer.front(), Some(&1.0));
        assert_eq!(buffer.pop_front(), Some(1.0));

        assert_eq!(buffer.front(), Some(&2.0));
        assert_eq!(buffer.pop_front(), Some(2.0));

        assert_eq!(buffer.front(), None);
        assert_eq!(buffer.pop_front(), None);
    }

    #[test]
    fn pop_back() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();

        assert_eq!(buffer.back(), None);
        assert_eq!(buffer.pop_back(), None);

        buffer.push_back(1.0);
        buffer.push_back(2.0);

        assert_eq!(buffer.back(), Some(&2.0));
        assert_eq!(buffer.pop_back(), Some(2.0));

        assert_eq!(buffer.back(), Some(&1.0));
        assert_eq!(buffer.pop_back(), Some(1.0));

        assert_eq!(buffer.back(), None);
        assert_eq!(buffer.pop_back(), None);
    }

    #[test]
    fn front() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();
        assert_eq!(buffer.front(), None);

        buffer.push_back(1.0);
        assert_eq!(buffer.front(), Some(&1.0));

        buffer.push_back(2.0);
        assert_eq!(buffer.front(), Some(&1.0));

        buffer.pop_front();
        assert_eq!(buffer.back(), Some(&2.0));

        buffer.pop_front();
        assert_eq!(buffer.back(), None);
    }

    #[test]
    fn back() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();
        assert_eq!(buffer.back(), None);

        buffer.push_back(1.0);
        assert_eq!(buffer.back(), Some(&1.0));

        buffer.push_back(2.0);
        assert_eq!(buffer.back(), Some(&2.0));

        buffer.pop_front();
        assert_eq!(buffer.back(), Some(&2.0));

        buffer.pop_front();
        assert_eq!(buffer.back(), None);
    }

    #[test]
    fn is_empty() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();

        assert!(buffer.is_empty());

        buffer.push_back(42.0);
        assert!(!buffer.is_empty());

        buffer.pop_front();
        assert!(buffer.is_empty());
    }

    #[test]
    fn is_full() {
        let mut buffer: CircularBuffer<f32, 3> = CircularBuffer::default();

        assert!(!buffer.is_full());

        buffer.push_back(1.0);
        assert!(!buffer.is_full());

        buffer.push_back(2.0);
        assert!(!buffer.is_full());

        buffer.push_back(3.0);
        assert!(buffer.is_full());

        buffer.pop_front();
        assert!(!buffer.is_full());
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

        buffer.pop_front();
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn capacity() {
        assert_eq!(CircularBuffer::<f32, 3>::capacity(), 3);
    }

    #[test]
    fn from_iter() {
        let mut buffer: CircularBuffer<f32, 3> =
            CircularBuffer::from_iter(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(buffer.pop_front(), Some(3.0));

        assert_eq!(buffer.pop_front(), Some(4.0));

        assert_eq!(buffer.pop_front(), Some(5.0));
    }

    #[test]
    fn iter() {
        let buffer: CircularBuffer<f32, 3> = CircularBuffer::from_iter(vec![1.0, 2.0, 3.0]);

        let items: Vec<&f32> = buffer.iter().collect();
        assert_eq!(items, vec![&1.0, &2.0, &3.0]);
    }

    #[test]
    fn into_iter() {
        let buffer: CircularBuffer<f32, 3> = CircularBuffer::from_iter(vec![1.0, 2.0, 3.0]);

        let items: Vec<f32> = buffer.into_iter().collect();
        assert_eq!(items, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn clone() {
        let buffer: CircularBuffer<f32, 3> = CircularBuffer::from_iter(vec![1.0, 2.0]);

        let clone = buffer.clone();

        assert_eq!(buffer.start, clone.start);
        assert_eq!(buffer.end, clone.end);

        for (original, clone) in buffer.iter().zip(clone.iter()) {
            assert_eq!(original, clone);
        }
    }

    #[test]
    fn drop() {
        use droptest::prelude::*;

        let registry = DropRegistry::default();

        let mut buffer: CircularBuffer<DropGuard<()>, 5> =
            (0..3).map(|_| registry.new_guard()).collect();

        let drop_stats = registry.stats();

        // Buffer holds 3 guards, out of capacity of 5.
        // Thus 0 guard should have been dropped by now:
        assert_eq!(3, drop_stats.created);
        assert_eq!(0, drop_stats.dropped);

        for _ in 0..2 {
            buffer.push_back(registry.new_guard());
        }

        // Buffer holds 5 guards, out of capacity of 5:
        // Thus 0 guard should have been dropped by now:
        let drop_stats = registry.stats();

        assert_eq!(5, drop_stats.created);
        assert_eq!(0, drop_stats.dropped);

        for _ in 0..2 {
            buffer.push_back(registry.new_guard());
        }

        // Buffer holds 5 guards, out of capacity of 5:
        // Thus 2 guard should have been dropped by now:
        let drop_stats = registry.stats();

        assert_eq!(7, drop_stats.created);
        assert_eq!(2, drop_stats.dropped);

        core::mem::drop(buffer);

        // Buffer held 5 guards, out of capacity of 5:
        // Thus 2 + 5 = 7 guard should have been dropped by now:
        let drop_stats = registry.stats();

        assert_eq!(7, drop_stats.created);
        assert_eq!(7, drop_stats.dropped);
    }
}
