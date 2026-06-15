// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Signal consumers that process values without direct output.
//!
//! Sinks accept values from filters or sources and perform side effects or accumulate results
//! (writing to buffers, computing statistics, etc.). They must be paired with a
//! [`Finalize`](crate::traits::Finalize) implementation to extract the final result.

/// A sink retrieves the current signal value each time it is called, performing arbitrary actions
/// with it, such as writing values to a file or passing them to an audio-device.
/// When the final value has been passed to it calling `sink.finalize()` returns an output.
pub trait Sink<T>: Sized {
    /// Processes the input value.
    fn sink(&mut self, input: T);
}
