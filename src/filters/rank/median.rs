// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Moving median filters.

use core::fmt;

use crate::storage::AsSlice;
use crate::traits::{
    guts::{FromGuts, HasGuts, IntoGuts},
    Filter, Reset, State as StateTrait, StateMut,
};

#[cfg(feature = "derive")]
use crate::traits::ResetMut;

/// Implementation detail.
/// (Once we have value generics we will hopefully be able to un-leak it.)
#[doc(hidden)]
#[derive(Clone, PartialEq, Eq)]
pub struct ListNode<T> {
    /// Value stored in the node, or `None` if the slot is vacant.
    value: Option<T>,
    /// Index of the predecessor node in the sorted linked list.
    previous: usize,
    /// Index of the successor node in the sorted linked list.
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

/// The median filter's state.
///
/// `B` is the buffer storage container and must implement [`AsSlice<ListNode<T>>`].
/// Use [`MedianArray`] for stack-allocated, const-generic storage or [`MedianVec`]
/// (with the `alloc` feature) for heap-allocated, runtime-sized storage.
#[derive(Clone)]
pub struct State<T, B> {
    /// Buffer of list nodes.
    buffer: B,
    /// Cursor into the circular buffer of data.
    cursor: usize,
    /// Cursor to the beginning of the sorted circular list.
    head: usize,
    /// Cursor to the median of the sorted circular list.
    median: usize,
    /// Number of slots filled so far (saturates at window length).
    filled: usize,
    _phantom: core::marker::PhantomData<T>,
}

impl<T, B> fmt::Debug for State<T, B>
where
    T: fmt::Debug,
    B: AsSlice<ListNode<T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State")
            .field("buffer", &self.buffer.as_slice())
            .field("cursor", &self.cursor)
            .field("head", &self.head)
            .field("median", &self.median)
            .field("filled", &self.filled)
            .finish()
    }
}

/// A moving median filter backed by a generic flat storage container.
///
/// `T` is the value type; `B` is the node-buffer storage and must implement
/// [`AsSlice<ListNode<T>>`]. Use the [`MedianArray`] type alias for fixed-size,
/// stack-allocated storage or [`MedianVec`] for heap-allocated, runtime-sized storage.
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
#[derive(Clone)]
pub struct Median<T, B> {
    state: State<T, B>,
}

impl<T, B> fmt::Debug for Median<T, B>
where
    T: fmt::Debug,
    B: AsSlice<ListNode<T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Median")
            .field("state", &self.state)
            .finish()
    }
}

/// A [`Median`] filter backed by a fixed-size array `[ListNode<T>; N]`.
///
/// Provides stack-allocated, `no_std`-friendly storage. Use [`MedianVec`]
/// when the window size is only known at runtime.
pub type MedianArray<T, const N: usize> = Median<T, [ListNode<T>; N]>;

/// A [`Median`] filter backed by a heap-allocated `Vec<ListNode<T>>`.
///
/// Requires the `alloc` feature. Use [`MedianArray`] for `no_std` contexts
/// where the window size is known at compile time.
#[cfg(feature = "alloc")]
pub type MedianVec<T> = Median<T, alloc::vec::Vec<ListNode<T>>>;

/// A [`Median`] filter that borrows a `[ListNode<T>]` slice for its node buffer.
///
/// This alias allows sharing a caller-owned node-buffer slice without taking
/// ownership of it. Construct via [`Median::from_parts`], passing a
/// correctly-linked `&mut [ListNode<T>]` slice (use [`MedianVec::new_buffer`]
/// to build one).
pub type MedianRefMut<'a, T> = Median<T, &'a mut [ListNode<T>]>;

impl<T, const N: usize> Default for MedianArray<T, N> {
    fn default() -> Self {
        assert!(N > 0, "Median: window size N must be > 0");
        let buffer = core::array::from_fn(|index| ListNode {
            value: None,
            previous: (index + N - 1) % N,
            next: (index + 1) % N,
        });

        Self {
            state: State {
                buffer,
                cursor: 0,
                head: 0,
                median: 0,
                filled: 0,
                _phantom: core::marker::PhantomData,
            },
        }
    }
}

#[cfg(feature = "alloc")]
impl<T> MedianVec<T> {
    /// Initialises an empty median filter.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "Median: window size N must be > 0");
        let buffer = (0..n)
            .map(|index| ListNode {
                value: None,
                previous: (index + n - 1) % n,
                next: (index + 1) % n,
            })
            .collect();

        Self {
            state: State {
                buffer,
                cursor: 0,
                head: 0,
                median: 0,
                filled: 0,
                _phantom: core::marker::PhantomData,
            },
        }
    }
}

impl<T, B> StateTrait for Median<T, B> {
    type State = State<T, B>;
}

impl<T, B> StateMut for Median<T, B> {
    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }
}

impl<T, B> HasGuts for Median<T, B> {
    type Guts = State<T, B>;
}

impl<T, B> FromGuts for Median<T, B> {
    fn from_guts(guts: Self::Guts) -> Self {
        let state = guts;
        Self { state }
    }
}

impl<T, B> IntoGuts for Median<T, B> {
    fn into_guts(self) -> Self::Guts {
        self.state
    }
}

impl<T, const N: usize> Reset for MedianArray<T, N> {
    fn reset(self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "derive")]
impl<T, const N: usize> ResetMut for MedianArray<T, N> where Self: Reset {}

impl<T, B> Filter<T> for Median<T, B>
where
    T: Clone + PartialOrd,
    B: AsSlice<ListNode<T>>,
{
    type Output = T;

    fn filter(&mut self, input: T) -> Self::Output {
        let n = self.state.buffer.as_slice().len();
        if self.state.filled < n {
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

        // Read node value from buffer at `self.median`:
        unsafe { self.median_unchecked() }
    }
}

impl<T, B> Median<T, B>
where
    T: Clone,
    B: AsSlice<ListNode<T>>,
{
    /// Returns the window size of the filter.
    pub fn len(&self) -> usize {
        self.state.buffer.as_slice().len()
    }

    /// Returns `true` if the filter's buffer is empty, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.state.buffer.as_slice().is_empty()
    }

    /// Returns the filter buffer's current median value,
    /// or `None` if the filter has not yet received any values.
    pub fn median(&self) -> Option<T> {
        let index = self.state.median;
        self.state.buffer.as_slice()[index].value.clone()
    }

    /// Returns the filter buffer's current minimum value,
    /// or `None` if the filter has not yet received any values.
    pub fn min(&self) -> Option<T> {
        let index = self.state.head;
        self.state.buffer.as_slice()[index].value.clone()
    }

    /// Returns the filter buffer's current maximum value,
    /// or `None` if the filter has not yet received any values.
    pub fn max(&self) -> Option<T> {
        // head points to the minimum in the sorted linked list;
        // its predecessor in the circular list is therefore the maximum.
        let buf = self.state.buffer.as_slice();
        let index = buf[self.state.head].previous;
        buf[index].value.clone()
    }

    /// Iterates the current window values.
    /// Returns the populated values in the current window.
    pub fn window_iter(&self) -> impl Iterator<Item = &T> {
        self.state
            .buffer
            .as_slice()
            .iter()
            .filter_map(|node| node.value.as_ref())
    }

    /// Creates a [`Median`] from a pre-initialised node buffer.
    ///
    /// This constructor is intended for storage containers whose size is not
    /// known at compile time (e.g. [`MedianVec`]).
    ///
    /// The buffer is taken as-is with their current contents. Each node's `previous`
    /// and `next` indices must form a valid circular linked list
    /// (i.e. `node[node[i].next].previous == i` for all `i`, with all indices in bounds).
    ///
    /// # Expected storage state
    ///
    /// For the idiomatic initial state, the buffer should contain correctly
    /// linked nodes with all values set to `None`. Use
    /// [`MedianVec::new_buffer`] to construct such a buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is empty.
    pub fn from_parts(buffer: B) -> Self {
        assert!(
            !buffer.as_slice().is_empty(),
            "Median: window size N must be > 0"
        );

        debug_assert!(
            {
                let slice = buffer.as_slice();
                let n = slice.len();
                (0..n).all(|i| {
                    let next = slice[i].next;
                    let prev = slice[i].previous;
                    next < n && prev < n && slice[next].previous == i && slice[prev].next == i
                })
            },
            "Median: buffer nodes must form a valid circular linked list (next < n, prev < n, node[node[i].next].previous == i, node[node[i].prev].next == i for all i)"
        );

        Self {
            state: State {
                buffer,
                cursor: 0,
                head: 0,
                median: 0,
                filled: 0,
                _phantom: core::marker::PhantomData,
            },
        }
    }
}

impl<T, B> Median<T, B>
where
    T: Clone + PartialOrd,
    B: AsSlice<ListNode<T>>,
{
    #[inline]
    fn should_insert(&self, value: &T, current: usize, index: usize) -> bool {
        #[allow(clippy::option_if_let_else)]
        if let Some(ref v) = self.state.buffer.as_slice()[current].value {
            (index + 1 == self.len()) || (v >= value)
        } else {
            true
        }
    }

    #[inline]
    unsafe fn move_head_forward(&mut self) {
        if self.state.cursor == self.state.head {
            self.state.head = self.state.buffer.as_slice()[self.state.head].next;
        }
    }

    #[inline]
    unsafe fn remove_node(&mut self) {
        let (predecessor, successor) = {
            let node = &self.state.buffer.as_slice()[self.state.cursor];
            (node.previous, node.next)
        };
        self.state.buffer.as_mut_slice()[predecessor].next = successor;
        self.state.buffer.as_mut_slice()[self.state.cursor] = ListNode {
            previous: usize::MAX,
            value: None,
            next: usize::MAX,
        };
        self.state.buffer.as_mut_slice()[successor].previous = predecessor;
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

            current = self.state.buffer.as_slice()[current].next;
        }
    }

    #[inline]
    unsafe fn insert(&mut self, value: &T, current: usize) {
        let successor = current;
        let predecessor = self.state.buffer.as_slice()[current].previous;
        debug_assert!(self.state.buffer.as_slice().len() == 1 || current != self.state.cursor);
        self.state.buffer.as_mut_slice()[predecessor].next = self.state.cursor;
        self.state.buffer.as_mut_slice()[self.state.cursor] = ListNode {
            previous: predecessor,
            value: Some(value.clone()),
            next: successor,
        };
        self.state.buffer.as_mut_slice()[successor].previous = self.state.cursor;
    }

    #[inline]
    unsafe fn update_head(&mut self, value: &T) {
        #[allow(clippy::option_if_let_else)]
        let should_update_head =
            if let Some(ref head) = self.state.buffer.as_slice()[self.state.head].value {
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
            self.state.median = self.state.buffer.as_slice()[self.state.median].next;
        }
    }

    #[inline]
    unsafe fn increment_cursor(&mut self) {
        self.state.cursor = (self.state.cursor + 1) % self.len();
    }

    #[inline]
    unsafe fn median_unchecked(&mut self) -> T {
        self.median()
            .expect("median buffer must be non-empty after the first filter() call")
    }
}

#[cfg(test)]
mod tests;
