// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving median filters.

use core::fmt;

use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Filter, Reset, State as StateTrait, StateMut,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// The median filter's state.
#[derive(Clone)]
pub struct State<T, const N: usize> {
    // Buffer of list nodes:
    buffer: [ListNode<T>; N],
    // Cursor into circular buffer of data:
    cursor: usize,
    // Cursor to beginning of circular list:
    head: usize,
    // Cursor to median of circular list:
    median: usize,
    filled: usize,
}

impl<T, const N: usize> fmt::Debug for State<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("buffer", &self.buffer)
            .field("cursor", &self.cursor)
            .field("head", &self.head)
            .field("median", &self.median)
            .field("filled", &self.filled)
            .finish()
    }
}

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

/// A median filter of fixed width with linear complexity.
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
///
/// # Complexity
///
/// - **Time per sample:** O(N); one linear scan to find the insertion point in the sorted
///   linked list, plus an O(N) walk to recompute the median pointer from `head`.
/// - **Space:** O(N); fixed-size array of N `ListNode` entries embedded in-place.
#[derive(Clone, Debug)]
pub struct Median<T, const N: usize> {
    state: State<T, N>,
}

impl<T, const N: usize> Default for Median<T, N> {
    fn default() -> Self {
        assert!(N > 0, "Median: window size N must be > 0");
        let state = {
            let buffer = core::array::from_fn(|index| ListNode {
                value: None,
                previous: (index + N - 1) % N,
                next: (index + 1) % N,
            });

            let cursor = 0;
            let head = 0;
            let median = 0;
            let filled = 0;

            State {
                buffer,
                cursor,
                head,
                median,
                filled,
            }
        };
        Self { state }
    }
}

impl<T, const N: usize> StateTrait for Median<T, N> {
    type State = State<T, N>;
}

impl<T, const N: usize> StateMut for Median<T, N> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, const N: usize> HasGuts for Median<T, N> {
    type Guts = State<T, N>;
}

impl<T, const N: usize> FromGuts for Median<T, N> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, const N: usize> IntoGuts for Median<T, N> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for Median<T, N> {
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for Median<T, N> where Self: Reset {}

impl<T, const N: usize> Filter<T> for Median<T, N>
where
    T: Clone + PartialOrd,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        if self.state.filled < N {
            self.state.filled += 1;
        }

        // If the current head is about to be overwritten
        // we need to make sure to have the head point to
        // the next node after the current head:
        unsafe {
            self.move_head_forward();
        }

        // Remove the node that is about to be overwritten
        // from the linked list:
        unsafe {
            self.remove_node();
        }

        // Search for the insertion index in the linked list
        // in regards to `value` as the insertion index.
        unsafe {
            self.insert_value(&input);
        }

        // Update head to newly inserted node if
        // cursor's value <= head's value or head is empty:
        unsafe {
            self.update_head(&input);
        }

        unsafe {
            self.recompute_median_from_head();
        }

        // Increment and wrap data in pointer:
        unsafe {
            self.increment_cursor();
        }

        // Read node value from buffer at `self.medium`:
        unsafe { self.median_unchecked() }
    }
}

impl<T, const N: usize> Median<T, N>
where
    T: Clone,
{
    /// Returns the window size of the filter.
    pub fn len(&self) -> usize {
        self.state.buffer.len()
    }

    /// Returns `true` if the filter's buffer is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.state.buffer.is_empty()
    }

    /// Returns the filter buffer's current median value,
    /// or `None` if the filter has not yet received any values.
    pub fn median(&self) -> Option<T> {
        let index = self.state.median;
        self.state.buffer[index].value.clone()
    }

    /// Returns the filter buffer's current minimum value,
    /// or `None` if the filter has not yet received any values.
    pub fn min(&self) -> Option<T> {
        let index = self.state.head;
        self.state.buffer[index].value.clone()
    }

    /// Returns the filter buffer's current maximum value,
    /// or `None` if the filter has not yet received any values.
    pub fn max(&self) -> Option<T> {
        // head points to the minimum in the sorted linked list;
        // its predecessor in the circular list is therefore the maximum.
        let index = self.state.buffer[self.state.head].previous;
        self.state.buffer[index].value.clone()
    }

    /// Iterates the current window values.
    /// Returns the populated values in the current window.
    pub fn window_iter(&self) -> impl Iterator<Item = &T> {
        self.state
            .buffer
            .iter()
            .filter_map(|node| node.value.as_ref())
    }
}

impl<T, const N: usize> Median<T, N>
where
    T: Clone + PartialOrd,
{
    #[inline]
    fn should_insert(&self, value: &T, current: usize, index: usize) -> bool {
        #[allow(clippy::option_if_let_else)]
        if let Some(ref v) = self.state.buffer[current].value {
            (index + 1 == self.len()) || (v >= value)
        } else {
            true
        }
    }

    #[inline]
    unsafe fn move_head_forward(&mut self) {
        if self.state.cursor == self.state.head {
            self.state.head = self.state.buffer[self.state.head].next;
        }
    }

    #[inline]
    unsafe fn remove_node(&mut self) {
        let (predecessor, successor) = {
            let node = &self.state.buffer[self.state.cursor];
            (node.previous, node.next)
        };
        self.state.buffer[predecessor].next = successor;
        self.state.buffer[self.state.cursor] = ListNode {
            previous: usize::MAX,
            value: None,
            next: usize::MAX,
        };
        self.state.buffer[successor].previous = predecessor;
    }

    #[inline]
    unsafe fn insert_value(&mut self, value: &T) {
        let mut current = self.state.head;
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

            current = self.state.buffer[current].next;
        }
    }

    #[inline]
    unsafe fn insert(&mut self, value: &T, current: usize) {
        let successor = current;
        let predecessor = self.state.buffer[current].previous;
        debug_assert!(self.state.buffer.len() == 1 || current != self.state.cursor);
        self.state.buffer[predecessor].next = self.state.cursor;
        self.state.buffer[self.state.cursor] = ListNode {
            previous: predecessor,
            value: Some(value.clone()),
            next: successor,
        };
        self.state.buffer[successor].previous = self.state.cursor;
    }

    #[inline]
    unsafe fn update_head(&mut self, value: &T) {
        #[allow(clippy::option_if_let_else)]
        let should_update_head = if let Some(ref head) = self.state.buffer[self.state.head].value {
            value <= head
        } else {
            true
        };

        if should_update_head {
            self.state.head = self.state.cursor;
        }
    }

    #[inline]
    unsafe fn recompute_median_from_head(&mut self) {
        let target = (self.state.filled.saturating_sub(1)) / 2;
        self.state.median = self.state.head;
        for _ in 0..target {
            self.state.median = self.state.buffer[self.state.median].next;
        }
    }

    #[inline]
    unsafe fn increment_cursor(&mut self) {
        self.state.cursor = (self.state.cursor + 1) % (self.len());
    }

    #[inline]
    unsafe fn median_unchecked(&mut self) -> T {
        self.median()
            .expect("median buffer must be non-empty after the first filter() call")
    }
}

#[cfg(test)]
mod tests;
