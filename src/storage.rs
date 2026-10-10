// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Storage backend abstractions for DSP types.
//!
//! DSP types in this crate (filters, sinks, sources) hold their internal
//! buffers behind generic storage parameters rather than hard-coding a
//! particular container. This module defines the two storage contracts
//! used throughout the crate:
//!
//! - [`AsSlice`] for flat, fixed-length storage (e.g. coefficient tables,
//!   histogram bins), backed by `[T; N]` or (with the `alloc` feature)
//!   `alloc::vec::Vec<T>`.
//! - [`RingBuffer`] for FIFO tap/delay-line storage, backed by
//!   [`circular_buffer::FixedCircularBuffer`], (with the `alloc` feature)
//!   [`circular_buffer::HeapCircularBuffer`], or a borrowed
//!   `&mut `[`circular_buffer::CircularBuffer`].
//!
//! Prefer the `*Array` type aliases exposed by individual filters/sinks for
//! `no_std`, zero-allocation use, and the `*Vec` aliases when runtime-sized
//! storage is required.

use core::ops::{Add, Mul};

#[cfg(feature = "alloc")]
use circular_buffer::HeapCircularBuffer;
use circular_buffer::{CircularBuffer, FixedCircularBuffer};
use num_traits::{Num, Zero};

/// A contiguous, flat storage backend.
///
/// Implemented explicitly for `[T; N]` and (with `alloc`) `Vec<T>`. Downstream
/// types wrapping either can either implement `AsSlice` directly or delegate.
pub trait AsSlice<T> {
    /// Returns a shared slice view over the storage.
    fn as_slice(&self) -> &[T];

    /// Returns a mutable slice view over the storage.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Returns the number of elements in the storage.
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns `true` if the storage holds no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const N: usize> AsSlice<T> for [T; N] {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[cfg(feature = "alloc")]
impl<T> AsSlice<T> for alloc::vec::Vec<T> {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> AsSlice<T> for &mut [T] {
    fn as_slice(&self) -> &[T] {
        self
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

/// A fixed-capacity double-ended ring buffer.
///
/// Implemented for the ring-buffer types provided by the `circular-buffer`
/// crate: [`FixedCircularBuffer`] (const-generic capacity, `no_std`-friendly),
/// [`HeapCircularBuffer`] (heap-allocated, requires the `alloc` feature), and
/// a borrowed `&mut `[`CircularBuffer`] (for sharing a caller-owned ring
/// without taking ownership of it). This is the storage contract used by DSP
/// types that hold a sliding window of the most recent `N` samples (e.g.
/// convolution taps, delay lines, monotonic deques).
///
/// The full double-ended interface (`front`, `back`, `pop_front`, `pop_back`,
/// `iter_mut`) is included so that algorithms such as monotonic-deque
/// rank filters can operate through this trait without resorting to
/// `Deref`/`DerefMut` coercions, which are reserved for smart-pointer
/// relationships.
pub trait RingBuffer<T> {
    /// Appends `value` to the back of the buffer, evicting and returning the
    /// oldest element if the buffer was already at capacity.
    fn push_back(&mut self, value: T) -> Option<T>;

    /// Removes and returns the element at the front (oldest), or `None` if empty.
    fn pop_front(&mut self) -> Option<T>;

    /// Removes and returns the element at the back (newest), or `None` if empty.
    fn pop_back(&mut self) -> Option<T>;

    /// Returns a shared reference to the front (oldest) element, or `None` if empty.
    fn front(&self) -> Option<&T>;

    /// Returns a shared reference to the back (newest) element, or `None` if empty.
    fn back(&self) -> Option<&T>;

    /// Returns an iterator over the elements, from oldest to newest.
    ///
    /// The explicit `'a` lifetime and `T: 'a` bound are required here (but
    /// not on inherent methods) because RPITIT on a trait method does not
    /// get the implied bounds that an inherent-impl method's hidden
    /// `impl Trait` return type would otherwise pick up; without them,
    /// rustc cannot prove the returned iterator does not outlive `T`.
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;

    /// Returns a mutable iterator over the elements, from oldest to newest.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        T: 'a;

    /// Returns a pair of slices over the elements, from oldest to newest.
    ///
    /// The first slice holds the older run and the second the newer run; the
    /// second is empty when the contents are contiguous. Their concatenation
    /// is in the same order [`iter`](Self::iter) yields, so folding the first
    /// and then the second into one accumulator is equivalent to folding
    /// `iter()`, while giving each slice as a counted loop that can be
    /// unrolled and vectorized.
    fn as_slices(&self) -> (&[T], &[T]);

    /// Returns the number of elements currently in the buffer.
    fn len(&self) -> usize;

    /// Returns the maximum number of elements the buffer can hold.
    fn capacity(&self) -> usize;

    /// Returns `true` if the buffer holds `capacity()` elements.
    fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    /// Returns `true` if the buffer holds no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes all elements from the buffer.
    fn clear(&mut self);

    /// Replaces the entire contents with `capacity()` values produced by `f`.
    ///
    /// Equivalent to [`clear`](Self::clear) followed by pushing `capacity()`
    /// values. Note that the result is *full*, not empty: after this call
    /// [`is_full`](Self::is_full) is `true` and [`is_empty`](Self::is_empty)
    /// is `false`. `fill_with(T::zero)` establishes the zero-padded cold-start
    /// state for a numeric delay line without requiring `T: Clone`.
    fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> T;
}

impl<T, const N: usize> RingBuffer<T> for FixedCircularBuffer<T, N> {
    fn push_back(&mut self, value: T) -> Option<T> {
        (**self).push_back(value)
    }

    fn pop_front(&mut self) -> Option<T> {
        (**self).pop_front()
    }

    fn pop_back(&mut self) -> Option<T> {
        (**self).pop_back()
    }

    fn front(&self) -> Option<&T> {
        (**self).front()
    }

    fn back(&self) -> Option<&T> {
        (**self).back()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        (**self).iter()
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        T: 'a,
    {
        (**self).iter_mut()
    }

    fn as_slices(&self) -> (&[T], &[T]) {
        (**self).as_slices()
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn capacity(&self) -> usize {
        (**self).capacity()
    }

    fn is_full(&self) -> bool {
        (**self).is_full()
    }

    fn clear(&mut self) {
        (**self).clear();
    }

    fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> T,
    {
        (**self).fill_with(f);
    }
}

#[cfg(feature = "alloc")]
impl<T> RingBuffer<T> for HeapCircularBuffer<T> {
    fn push_back(&mut self, value: T) -> Option<T> {
        (**self).push_back(value)
    }

    fn pop_front(&mut self) -> Option<T> {
        (**self).pop_front()
    }

    fn pop_back(&mut self) -> Option<T> {
        (**self).pop_back()
    }

    fn front(&self) -> Option<&T> {
        (**self).front()
    }

    fn back(&self) -> Option<&T> {
        (**self).back()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        (**self).iter()
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        T: 'a,
    {
        (**self).iter_mut()
    }

    fn as_slices(&self) -> (&[T], &[T]) {
        (**self).as_slices()
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn capacity(&self) -> usize {
        (**self).capacity()
    }

    fn is_full(&self) -> bool {
        (**self).is_full()
    }

    fn clear(&mut self) {
        (**self).clear();
    }

    fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> T,
    {
        (**self).fill_with(f);
    }
}

impl<T> RingBuffer<T> for &mut CircularBuffer<T> {
    fn push_back(&mut self, value: T) -> Option<T> {
        (**self).push_back(value)
    }

    fn pop_front(&mut self) -> Option<T> {
        (**self).pop_front()
    }

    fn pop_back(&mut self) -> Option<T> {
        (**self).pop_back()
    }

    fn front(&self) -> Option<&T> {
        (**self).front()
    }

    fn back(&self) -> Option<&T> {
        (**self).back()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        (**self).iter()
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        T: 'a,
    {
        (**self).iter_mut()
    }

    fn as_slices(&self) -> (&[T], &[T]) {
        (**self).as_slices()
    }

    fn len(&self) -> usize {
        (**self).len()
    }

    fn capacity(&self) -> usize {
        (**self).capacity()
    }

    fn is_full(&self) -> bool {
        (**self).is_full()
    }

    fn clear(&mut self) {
        (**self).clear();
    }

    fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> T,
    {
        (**self).fill_with(f);
    }
}

pub(crate) fn zero_filled_fixed_ring<T: Num, const N: usize>() -> FixedCircularBuffer<T, N> {
    let mut buf = FixedCircularBuffer::new();
    RingBuffer::fill_with(&mut buf, T::zero);
    buf
}

// Accumulates `sample * coefficient` oldest-to-newest into a single running
// sum. The coefficients are reversed, so `h[N - 1]` pairs with the oldest
// sample and `h[0]` with the newest. The samples arrive as two runs whose
// concatenation is oldest-to-newest, so the split point in the coefficients
// is the first run's length, counted from the end: the first run pairs with
// the last `front.len()` coefficients (reversed) and the second run with the
// remaining leading ones (reversed). Like `zip`, the pairing truncates to the
// shorter of samples and coefficients. One accumulator is threaded through
// both runs because floating-point addition is not associative.
pub(crate) fn dot_product_reversed_coefficients<T, K, O>(
    samples: (&[T], &[T]),
    coefficients: &[K],
) -> O
where
    T: Clone + Mul<K, Output = O>,
    K: Clone,
    O: Zero + Add<Output = O>,
{
    let (front, back) = samples;
    let front_count = front.len().min(coefficients.len());
    let (head, tail) = coefficients.split_at(coefficients.len() - front_count);

    let sum = front
        .iter()
        .zip(tail.iter().rev())
        .fold(O::zero(), |sum, (sample, coefficient)| {
            sum + (sample.clone() * coefficient.clone())
        });

    back.iter()
        .zip(head.iter().rev())
        .fold(sum, |sum, (sample, coefficient)| {
            sum + (sample.clone() * coefficient.clone())
        })
}

// Sums the samples oldest-to-newest into a single running sum. The samples
// arrive as two runs whose concatenation is oldest-to-newest, so one
// accumulator is threaded through the first run and then the second. This
// order keeps the result bit-identical to a single fold over the whole
// window, which matters because floating-point addition is not associative.
pub(crate) fn sum_slice_pair<T>(samples: (&[T], &[T])) -> T
where
    T: Clone + Zero + Add<Output = T>,
{
    let (front, back) = samples;
    let sum = front
        .iter()
        .fold(T::zero(), |sum, sample| sum + sample.clone());

    back.iter().fold(sum, |sum, sample| sum + sample.clone())
}

#[cfg(test)]
mod tests {
    use circular_buffer::{CircularBuffer, FixedCircularBuffer};

    #[cfg(feature = "alloc")]
    use circular_buffer::HeapCircularBuffer;

    use super::{AsSlice, RingBuffer};

    // ---------------------------------------------------------------------------
    // RingBuffer contract tests
    // ---------------------------------------------------------------------------

    /// Runs the full RingBuffer contract against an arbitrary implementor.
    ///
    /// The macro body is intentionally kept as a single expression block so
    /// the three instantiations below share identical logic without duplication.
    macro_rules! ring_buffer_contract {
        ($name:ident, $make_ring:expr) => {
            mod $name {
                use super::*;

                fn make() -> impl RingBuffer<i32> {
                    $make_ring
                }

                #[test]
                fn push_past_full_len_capacity_is_full() {
                    let mut ring = make();
                    ring.push_back(1);
                    ring.push_back(2);
                    ring.push_back(3);
                    ring.push_back(4); // 4th push into cap-3 ring
                    assert_eq!(ring.len(), 3);
                    assert_eq!(ring.capacity(), 3);
                    assert!(ring.is_full());
                }

                #[test]
                fn push_past_full_evicts_oldest() {
                    let mut ring = make();
                    ring.push_back(1);
                    ring.push_back(2);
                    ring.push_back(3);
                    // 4th push must evict the oldest (1)
                    let evicted = ring.push_back(4);
                    assert_eq!(evicted, Some(1));
                }

                #[test]
                fn iter_yields_oldest_to_newest() {
                    let mut ring = make();
                    ring.push_back(1);
                    ring.push_back(2);
                    ring.push_back(3);
                    ring.push_back(4); // evicts 1
                    let items: alloc::vec::Vec<i32> = ring.iter().copied().collect();
                    assert_eq!(items, [2, 3, 4]);
                }

                #[test]
                fn clear_makes_empty() {
                    let mut ring = make();
                    ring.push_back(1);
                    ring.push_back(2);
                    ring.clear();
                    assert_eq!(ring.len(), 0);
                    assert!(ring.is_empty());
                }

                #[test]
                fn front_and_back_correct() {
                    let mut ring = make();
                    ring.push_back(10);
                    ring.push_back(20);
                    ring.push_back(30);
                    assert_eq!(ring.front(), Some(&10));
                    assert_eq!(ring.back(), Some(&30));
                }

                #[test]
                fn pop_front_removes_oldest() {
                    let mut ring = make();
                    ring.push_back(10);
                    ring.push_back(20);
                    ring.push_back(30);
                    assert_eq!(ring.pop_front(), Some(10));
                    assert_eq!(ring.len(), 2);
                    assert_eq!(ring.front(), Some(&20));
                }

                #[test]
                fn pop_back_removes_newest() {
                    let mut ring = make();
                    ring.push_back(10);
                    ring.push_back(20);
                    ring.push_back(30);
                    assert_eq!(ring.pop_back(), Some(30));
                    assert_eq!(ring.len(), 2);
                    assert_eq!(ring.back(), Some(&20));
                }

                #[test]
                fn as_slices_empty_yields_two_empty_slices() {
                    let ring = make();
                    assert_eq!(ring.as_slices(), (&[][..], &[][..]));
                    assert_eq!(ring.len(), 0);
                }

                #[test]
                fn as_slices_contiguous_leaves_second_empty() {
                    let mut ring = make();
                    ring.push_back(1);
                    ring.push_back(2);
                    assert_eq!(ring.as_slices(), (&[1, 2][..], &[][..]));
                }

                #[test]
                fn as_slices_wrapped_splits_oldest_first() {
                    let mut ring = make();
                    ring.push_back(1);
                    ring.push_back(2);
                    ring.push_back(3);
                    ring.push_back(4); // evicts 1
                    let (front, back) = ring.as_slices();
                    assert_eq!((front, back), (&[2, 3][..], &[4][..]));
                    assert_eq!(front.len() + back.len(), ring.len());
                }

                #[test]
                fn as_slices_concatenation_matches_iter() {
                    let mut ring = make();
                    ring.push_back(1);
                    ring.push_back(2);
                    ring.push_back(3);
                    ring.push_back(4);
                    let (front, back) = ring.as_slices();
                    let concatenated: alloc::vec::Vec<i32> =
                        front.iter().chain(back.iter()).copied().collect();
                    let iterated: alloc::vec::Vec<i32> = ring.iter().copied().collect();
                    assert_eq!(concatenated, iterated);
                    assert_eq!(concatenated, [2, 3, 4]);
                }
            }
        };
    }

    ring_buffer_contract!(fixed, FixedCircularBuffer::<i32, 3>::new());

    #[cfg(feature = "alloc")]
    ring_buffer_contract!(heap, HeapCircularBuffer::<i32>::with_capacity(3));

    // The borrowed-ring impl requires returning a concrete type, not `impl
    // Trait`, so we test it via a separate inline block instead of the macro.
    mod borrowed {
        use super::*;

        #[test]
        fn push_past_full_len_capacity_is_full() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(1);
            ring.push_back(2);
            ring.push_back(3);
            ring.push_back(4);
            assert_eq!(ring.len(), 3);
            assert_eq!(ring.capacity(), 3);
            assert!(ring.is_full());
        }

        #[test]
        fn push_past_full_evicts_oldest() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(1);
            ring.push_back(2);
            ring.push_back(3);
            let evicted = ring.push_back(4);
            assert_eq!(evicted, Some(1));
        }

        #[test]
        fn iter_yields_oldest_to_newest() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(1);
            ring.push_back(2);
            ring.push_back(3);
            ring.push_back(4);
            let items: alloc::vec::Vec<i32> = ring.iter().copied().collect();
            assert_eq!(items, [2, 3, 4]);
        }

        #[test]
        fn clear_makes_empty() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(1);
            ring.push_back(2);
            ring.clear();
            assert_eq!(ring.len(), 0);
            assert!(ring.is_empty());
        }

        #[test]
        fn front_and_back_correct() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(10);
            ring.push_back(20);
            ring.push_back(30);
            assert_eq!(ring.front(), Some(&10));
            assert_eq!(ring.back(), Some(&30));
        }

        #[test]
        fn pop_front_removes_oldest() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(10);
            ring.push_back(20);
            ring.push_back(30);
            assert_eq!(ring.pop_front(), Some(10));
            assert_eq!(ring.len(), 2);
            assert_eq!(ring.front(), Some(&20));
        }

        #[test]
        fn pop_back_removes_newest() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(10);
            ring.push_back(20);
            ring.push_back(30);
            assert_eq!(ring.pop_back(), Some(30));
            assert_eq!(ring.len(), 2);
            assert_eq!(ring.back(), Some(&20));
        }

        #[test]
        fn as_slices_empty_yields_two_empty_slices() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            assert_eq!(RingBuffer::as_slices(&ring), (&[][..], &[][..]));
            assert_eq!(RingBuffer::len(&ring), 0);
        }

        #[test]
        fn as_slices_contiguous_leaves_second_empty() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(1);
            ring.push_back(2);
            assert_eq!(RingBuffer::as_slices(&ring), (&[1, 2][..], &[][..]));
        }

        #[test]
        fn as_slices_wrapped_splits_oldest_first() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(1);
            ring.push_back(2);
            ring.push_back(3);
            ring.push_back(4);
            let (front, back) = RingBuffer::as_slices(&ring);
            assert_eq!((front, back), (&[2, 3][..], &[4][..]));
            assert_eq!(front.len() + back.len(), RingBuffer::len(&ring));
        }

        #[test]
        fn as_slices_concatenation_matches_iter() {
            let mut owned = FixedCircularBuffer::<i32, 3>::new();
            let ring: &mut CircularBuffer<i32> = &mut *owned;
            ring.push_back(1);
            ring.push_back(2);
            ring.push_back(3);
            ring.push_back(4);
            let (front, back) = RingBuffer::as_slices(&ring);
            let concatenated: alloc::vec::Vec<i32> =
                front.iter().chain(back.iter()).copied().collect();
            let iterated: alloc::vec::Vec<i32> = RingBuffer::iter(&ring).copied().collect();
            assert_eq!(concatenated, iterated);
            assert_eq!(concatenated, [2, 3, 4]);
        }
    }

    // ---------------------------------------------------------------------------
    // AsSlice contract tests
    // ---------------------------------------------------------------------------

    mod as_slice_array {
        use super::*;

        #[test]
        fn as_slice_returns_correct_elements() {
            let storage: [i32; 3] = [10, 20, 30];
            assert_eq!(storage.as_slice(), &[10, 20, 30]);
        }

        #[test]
        fn as_mut_slice_allows_mutation() {
            let mut storage: [i32; 3] = [1, 2, 3];
            storage.as_mut_slice()[1] = 99;
            assert_eq!(storage, [1, 99, 3]);
        }

        #[test]
        fn len_and_is_empty() {
            let storage: [i32; 3] = [1, 2, 3];
            assert_eq!(AsSlice::len(&storage), 3);
            assert!(!AsSlice::is_empty(&storage));

            let empty: [i32; 0] = [];
            assert_eq!(AsSlice::len(&empty), 0);
            assert!(AsSlice::is_empty(&empty));
        }
    }

    #[cfg(feature = "alloc")]
    mod as_slice_vec {
        use alloc::vec;

        use super::*;

        #[test]
        fn as_slice_returns_correct_elements() {
            let storage: alloc::vec::Vec<i32> = vec![10, 20, 30];
            assert_eq!(storage.as_slice(), &[10, 20, 30]);
        }

        #[test]
        fn as_mut_slice_allows_mutation() {
            let mut storage: alloc::vec::Vec<i32> = vec![1, 2, 3];
            storage.as_mut_slice()[1] = 99;
            assert_eq!(storage, [1, 99, 3]);
        }

        #[test]
        fn len_and_is_empty() {
            let storage: alloc::vec::Vec<i32> = vec![1, 2, 3];
            assert_eq!(AsSlice::len(&storage), 3);
            assert!(!AsSlice::is_empty(&storage));

            let empty: alloc::vec::Vec<i32> = alloc::vec::Vec::new();
            assert_eq!(AsSlice::len(&empty), 0);
            assert!(AsSlice::is_empty(&empty));
        }
    }

    mod dot_product_reversed_coefficients_helper {
        use super::super::dot_product_reversed_coefficients;

        #[test]
        fn empty_sample_pair_is_zero() {
            assert_eq!(
                dot_product_reversed_coefficients::<i32, i32, i32>((&[], &[]), &[1, 2, 3]),
                0
            );
            assert_eq!(
                dot_product_reversed_coefficients::<i32, i32, i32>((&[], &[]), &[]),
                0
            );
        }

        #[test]
        fn contiguous_pair_matches_reversed_zip() {
            let coefficients = [1, 2, 3];
            let actual =
                dot_product_reversed_coefficients((&[10, 20, 30][..], &[][..]), &coefficients);
            let expected = [10, 20, 30]
                .iter()
                .zip(coefficients.iter().rev())
                .fold(0, |sum, (sample, coefficient)| sum + sample * coefficient);

            assert_eq!(actual, 100);
            assert_eq!(actual, expected);
        }

        #[test]
        fn wrapped_pair_matches_reversed_zip() {
            let coefficients = [1, 2, 3];
            let actual =
                dot_product_reversed_coefficients((&[10, 20][..], &[30][..]), &coefficients);
            let expected = [10, 20, 30]
                .iter()
                .zip(coefficients.iter().rev())
                .fold(0, |sum, (sample, coefficient)| sum + sample * coefficient);

            assert_eq!(actual, 100);
            assert_eq!(actual, expected);
        }

        #[test]
        fn single_element_matches_reversed_zip() {
            let actual = dot_product_reversed_coefficients((&[3][..], &[][..]), &[7]);

            assert_eq!(actual, 21);
        }

        #[test]
        fn fewer_samples_than_coefficients_truncates_like_zip() {
            let actual = dot_product_reversed_coefficients((&[10][..], &[][..]), &[1, 2, 3, 4]);

            assert_eq!(actual, 40);
        }
    }

    mod sum_slice_pair_helper {
        use super::super::sum_slice_pair;

        #[test]
        fn empty_pair_is_zero() {
            assert_eq!(sum_slice_pair::<i32>((&[], &[])), 0);
        }

        // The explicit `fold` mirrors the helper's accumulation order.
        #[test]
        #[allow(clippy::unnecessary_fold)]
        fn contiguous_pair_matches_single_fold() {
            let expected = [1, 2, 3].iter().fold(0, |acc, x| acc + x);

            assert_eq!(sum_slice_pair((&[1, 2, 3][..], &[][..])), 6);
            assert_eq!(sum_slice_pair((&[1, 2, 3][..], &[][..])), expected);
        }

        #[test]
        #[allow(clippy::unnecessary_fold)]
        fn wrapped_pair_matches_single_fold() {
            let expected = [1, 2, 3, 4].iter().fold(0, |acc, x| acc + x);

            assert_eq!(sum_slice_pair((&[1, 2][..], &[3, 4][..])), 10);
            assert_eq!(sum_slice_pair((&[1, 2][..], &[3, 4][..])), expected);
        }
    }
}
