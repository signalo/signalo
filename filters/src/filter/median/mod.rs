// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving median filters.

use signalo_traits::filter::Filter;

use std::fmt;
use std::ptr;

use generic_array::{ArrayLength, GenericArray, ArrayBuilder};

pub mod exp;

/// Implementation detail.
/// (Once we have value generics we will hopefully be able to un-leak it.)
#[doc(hidden)]
#[derive(Clone, PartialEq, Eq)]
pub struct ListNode<T> {
    value: Option<T>,
    previous: usize,
    next: usize,
}

impl<T> fmt::Debug for ListNode<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "@{:?}-{:?}-@{:?}", self.previous, self.value, self.next)
    }
}

/// An implementation of a median filter of fixed width with linear complexity.
///
/// While the common naïve implementation of a median filter
/// has a worst-case complexity of `O(n^2)` (due to having to sort the sliding window)
/// the use of a combination of linked list and ring buffer allows for
/// a worst-case complexity of `O(n)`.
///
/// # Implementation
///
/// The algorithm makes use of a ring buffer of the same size as its filter window.
/// Inserting values into the ring buffer appends them to a linked list that is *embedded*
/// inside said ring buffer (using relative integer jump offsets as links).
///
/// # Example
///
/// Given a sequence of values `[3, 2, 4, 6, 5, 1]` and a buffer of size 5,
/// the buffer would be filled like this:
///
/// ```plain
/// new(5)  consume(3)  consume(2)  consume(4)  consume(6)  consume(5)  consume(1)
/// ▶︎[ ]      ▷[3]       ┌→[3]       ┌→[3]─┐     ┌→[3]─┐    ▶︎┌→[3]─┐      ▷[1]─┐
///  [ ]      ▶︎[ ]      ▷└─[2]      ▷└─[2] │    ▷└─[2] │    ▷└─[2] │    ▶︎┌─[2]←┘
///  [ ]       [ ]        ▶︎[ ]         [4]←┘     ┌─[4]←┘     ┌─[4]←┘     └→[4]─┐
///  [ ]       [ ]         [ ]        ▶︎[ ]       └→[6]       │ [6]←┐     ┌→[6] │
///  [ ]       [ ]         [ ]         [ ]        ▶︎[ ]       └→[5]─┘     └─[5]←┘
/// ```
///
/// # Algorithm
///
/// 1. **Remove node** at current cursor (`▶︎`) from linked list, if it exists.
///    (by re-wiring its predecessor to its successor).
/// 2. **Initialize** `current` and `median` index to first node of linked list (`▷`).
/// 3. **Walk through** linked list, **searching** for insertion point.
/// 4. **Shift median index** on every other hop (thus ending up in the list's median).
/// 5. **Insert value** into ring buffer and linked list respectively.
/// 6. **Update index** to linked list's first node, if necessary.
/// 7. **Update ring buffer**'s cursor.
/// 8. **Return median value**.
///
/// (_Based on Phil Ekstrom, Embedded Systems Programming, November 2000._)
pub struct Median<T, N>
where
    N: ArrayLength<ListNode<T>>,
{
    // Buffer of list nodes:
    buffer: GenericArray<ListNode<T>, N>,
    // Cursor into circular buffer of data:
    cursor: usize,
    // Cursor to beginning of circular list:
    head: usize,
    // Cursor to median of circular list:
    median: usize,
}

impl<T, N> Clone for Median<T, N>
where
    T: Clone,
    N: ArrayLength<ListNode<T>>,
{
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            cursor: self.cursor.clone(),
            head: self.head.clone(),
            median: self.median.clone(),
        }
    }
}

impl<T, N> fmt::Debug for Median<T, N>
where
    T: fmt::Debug,
    N: ArrayLength<ListNode<T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Median")
            .field("buffer", &self.buffer)
            .field("cursor", &self.cursor)
            .field("head", &self.head)
            .field("median", &self.median)
            .finish()
    }
}

impl<T, N> Median<T, N>
where
    T: Clone + PartialOrd,
    N: ArrayLength<ListNode<T>>,
{
    /// Creates a new median filter with a given window size.
    pub fn new() -> Self {
        let buffer = unsafe {
            let mut builder = ArrayBuilder::new();
            {
                let size = N::to_usize();
                let (mut iter, index) = builder.iter_position();
                while let Some(destination) = iter.next() {
                    let node = ListNode {
                        value: None,
                        previous: (*index + size - 1) % size,
                        next: (*index + 1) % size,
                    };
                    ptr::write(destination, node);
                    *index += 1;
                }
            }
            builder.into_inner()
        };
        Median {
            buffer: buffer,
            cursor: 0,
            head: 0,
            median: 0,
        }
    }

    /// Returns the window size of the filter.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the filter buffer's current median value, panicking if empty.
    #[inline]
    pub fn median(&self) -> Option<T> {
        let index = self.median;
        self.buffer[index].value.clone()
    }

    /// Returns the filter buffer's current min value, panicking if empty.
    #[inline]
    pub fn min(&self) -> Option<T> {
        let index = self.head;
        self.buffer[index].value.clone()
    }

    /// Returns the filter buffer's current max value, panicking if empty.
    #[inline]
    pub fn max(&self) -> Option<T> {
        let index = (self.cursor + self.len() - 1) % (self.len());
        self.buffer[index].value.clone()
    }

    #[inline]
    fn should_insert(&self, value: &T, current: usize, index: usize) -> bool {
        if let Some(ref v) = self.buffer[current].value {
            (index + 1 == self.len()) || !(v < value)
        } else {
            true
        }
    }

    #[inline]
    unsafe fn move_head_forward(&mut self) {
        if self.cursor == self.head {
            self.head = self.buffer[self.head].next;
        }
    }

    #[inline]
    unsafe fn remove_node(&mut self) {
        let (predecessor, successor) = {
            let node = &self.buffer[self.cursor];
            (node.previous, node.next)
        };
        self.buffer[predecessor].next = successor;
        self.buffer[self.cursor] = ListNode {
            previous: usize::max_value(),
            value: None,
            next: usize::max_value(),
        };
        self.buffer[successor].previous = predecessor;
    }

    #[inline]
    unsafe fn initialize_median(&mut self) {
        self.median = self.head;
    }

    #[inline]
    unsafe fn insert_value(&mut self, value: &T) {
        let mut current = self.head;
        let buffer_len = self.len();
        let mut has_inserted = false;
        for index in 0..buffer_len {
            if !has_inserted {
                let should_insert = self.should_insert(value, current, index);
                if should_insert {
                    // Insert previously removed node with new value
                    // into linked list at given insertion index.
                    self.insert(value, current);
                    has_inserted = true;
                }
            }

            // Shift median on every other element in the list,
            // so that it ends up in the middle, eventually:
            self.shift_median(index, current);

            current = self.buffer[current].next;
        }
    }

    #[inline]
    unsafe fn insert(&mut self, value: &T, current: usize) {
        let successor = current;
        let predecessor = self.buffer[current].previous;
        debug_assert!(self.buffer.len() == 1 || current != self.cursor);
        self.buffer[predecessor].next = self.cursor;
        self.buffer[self.cursor] = ListNode {
            previous: predecessor,
            value: Some(value.clone()),
            next: successor,
        };
        self.buffer[successor].previous = self.cursor;
    }

    #[inline]
    unsafe fn shift_median(&mut self, index: usize, current: usize) {
        if (index & 0b1 == 0b1) && (self.buffer[current].value.is_some()) {
            self.median = self.buffer[self.median].next;
        }
    }

    #[inline]
    unsafe fn update_head(&mut self, value: &T) {
        let should_update_head = if let Some(ref head) = self.buffer[self.head].value {
            value <= head
        } else {
            true
        };

        if should_update_head {
            self.head = self.cursor;
            self.median = self.buffer[self.median].previous;
        }
    }

    #[inline]
    unsafe fn adjust_median_for_even_length(&mut self) {
        if self.len() % 2 == 0 {
            self.median = self.buffer[self.median].previous;
        }
    }

    #[inline]
    unsafe fn increment_cursor(&mut self) {
        self.cursor = (self.cursor + 1) % (self.len());
    }

    #[inline]
    unsafe fn median_unchecked(&mut self) -> T {
        self.median().unwrap()
    }
}

impl<T, N> Filter<T> for Median<T, N>
where
    T: Clone + PartialOrd,
    N: ArrayLength<ListNode<T>>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        // If the current head is about to be overwritten
        // we need to make sure to have the head point to
        // the next node after the current head:
        unsafe { self.move_head_forward(); }

        // Remove the node that is about to be overwritten
        // from the linked list:
        unsafe { self.remove_node(); }

        // Initialize `self.median` pointing
        // to the first (smallest) node in the sorted list:
        unsafe { self.initialize_median(); }

        // Search for the insertion index in the linked list
        // in regards to `value` as the insertion index.
        unsafe { self.insert_value(&input); }

        // Update head to newly inserted node if
        // cursor's value <= head's value or head is empty:
        unsafe { self.update_head(&input); }

        // If the filter has an even window size, then shift the median
        // back one slot, so that it points to the left one
        // of the middle pair of median values
        unsafe { self.adjust_median_for_even_length(); }

        // Increment and wrap data in pointer:
        unsafe { self.increment_cursor(); }

        // Read node value from buffer at `self.medium`:
        unsafe { self.median_unchecked() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use generic_array::typenum::*;

    macro_rules! test_filter {
        ($size:ident, $input:expr, $output:expr) => {
            let filter: Median<_, $size> = Median::new();
            let output: Vec<_> = $input.iter().scan(filter, |filter, &input| {
                Some(filter.filter(input))
            }).collect();
            assert_eq!(output, $output);
        }
    }

    #[test]
    fn single_peak_4() {
        let input = vec![10, 20, 30, 100, 30, 20, 10];
        let output = vec![10, 10, 20, 20, 30, 30, 20];

        test_filter!(U4, input, output);
    }

    #[test]
    fn single_peak_5() {
        let input = vec![10, 20, 30, 100, 30, 20, 10];
        let output = vec![10, 10, 20, 20, 30, 30, 30];
        test_filter!(U5, input, output);
    }

    #[test]
    fn single_valley_4() {
        let input = vec![90, 80, 70, 10, 70, 80, 90];
        let output = vec![90, 80, 80, 70, 70, 70, 70];
        test_filter!(U4, input, output);
    }

    #[test]
    fn single_valley_5() {
        let input = vec![90, 80, 70, 10, 70, 80, 90];
        let output = vec![90, 80, 80, 70, 70, 70, 70];
        test_filter!(U5, input, output);
    }

    #[test]
    fn single_outlier_4() {
        let input = vec![10, 10, 10, 100, 10, 10, 10];
        let output = vec![10, 10, 10, 10, 10, 10, 10];
        test_filter!(U4, input, output);
    }

    #[test]
    fn single_outlier_5() {
        let input = vec![10, 10, 10, 100, 10, 10, 10];
        let output = vec![10, 10, 10, 10, 10, 10, 10];
        test_filter!(U5, input, output);
    }

    #[test]
    fn triple_outlier_4() {
        let input = vec![10, 10, 100, 100, 100, 10, 10];
        let output = vec![10, 10, 10, 10, 100, 100, 10];
        test_filter!(U4, input, output);
    }

    #[test]
    fn triple_outlier_5() {
        let input = vec![10, 10, 100, 100, 100, 10, 10];
        let output = vec![10, 10, 10, 10, 100, 100, 100];
        test_filter!(U5, input, output);
    }

    #[test]
    fn quintuple_outlier_4() {
        let input = vec![10, 100, 100, 100, 100, 100, 10];
        let output = vec![10, 10, 100, 100, 100, 100, 100];
        test_filter!(U4, input, output);
    }

    #[test]
    fn quintuple_outlier_5() {
        let input = vec![10, 100, 100, 100, 100, 100, 10];
        let output = vec![10, 10, 100, 100, 100, 100, 100];
        test_filter!(U5, input, output);
    }

    #[test]
    fn alternating_4() {
        let input = vec![10, 20, 10, 20, 10, 20, 10];
        let output = vec![10, 10, 10, 10, 10, 10, 10];
        test_filter!(U4, input, output);
    }

    #[test]
    fn alternating_5() {
        let input = vec![10, 20, 10, 20, 10, 20, 10];
        let output = vec![10, 10, 10, 10, 10, 20, 10];
        test_filter!(U5, input, output);
    }

    #[test]
    fn ascending_4() {
        let input = vec![10, 20, 30, 40, 50, 60, 70];
        let output = vec![10, 10, 20, 20, 30, 40, 50];
        test_filter!(U4, input, output);
    }

    #[test]
    fn ascending_5() {
        let input = vec![10, 20, 30, 40, 50, 60, 70];
        let output = vec![10, 10, 20, 20, 30, 40, 50];
        test_filter!(U5, input, output);
    }

    #[test]
    fn descending_4() {
        let input = vec![70, 60, 50, 40, 30, 20, 10];
        let output = vec![70, 60, 60, 50, 40, 30, 20];
        test_filter!(U4, input, output);
    }

    #[test]
    fn descending_5() {
        let input = vec![70, 60, 50, 40, 30, 20, 10];
        let output = vec![70, 60, 60, 50, 50, 40, 30];
        test_filter!(U5, input, output);
    }

    #[test]
    fn min_max_median() {
        let input = vec![70, 50, 30, 10, 20, 40, 60];
        let mut filter: Median<_, U5> = Median::new();
        for input in input {
            filter.filter(input);
        }
        assert_eq!(filter.min(), Some(10));
        assert_eq!(filter.max(), Some(60));
        assert_eq!(filter.median(), Some(30));
    }

    fn get_input() -> Vec<f32> {
        vec![
            0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 3.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0,
            12.0, 20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 18.0, 18.0, 18.0,
            106.0, 5.0, 26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0,
            16.0, 16.0, 104.0, 11.0, 24.0, 24.0
        ]
    }

    fn get_output() -> Vec<f32> {
        vec![
            0.0, 0.0, 1.0, 1.0, 2.0, 5.0, 7.0, 5.0, 8.0, 8.0, 14.0, 9.0, 9.0, 9.0, 14.0, 9.0,
            12.0, 17.0, 17.0, 12.0, 12.0, 15.0, 15.0, 10.0, 15.0, 15.0, 15.0, 18.0, 18.0, 18.0,
            18.0, 18.0, 18.0, 18.0, 13.0, 13.0, 21.0, 21.0, 21.0, 21.0, 21.0, 21.0, 29.0, 16.0,
            16.0, 16.0, 16.0, 16.0, 16.0, 24.0
        ]
    }

    #[test]
    fn test() {
        let filter: Median<_, U5> = Median::new();
        // Sequence: https://en.wikipedia.org/wiki/Collatz_conjecture
        let input = get_input();
        let output: Vec<_> = input.iter().scan(filter, |filter, &input| {
            Some(filter.filter(input))
        }).collect();
        assert_nearly_eq!(output, get_output(), 0.001);
    }
}
