// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sources of a digital signal.

/// A sink retrieves the current signal value each time it is called, performing arbitrary actions
/// with it, such as writing values to a file or passing them to an audio-device.
/// When the final value has been passed to it calling `sink.finalize()` returns an output.
pub trait Finalize: Sized {
    /// The sink's output type.
    type Output;

    /// Consumes `self`, returning an accumulated output.
    fn finalize(self) -> Self::Output;
}
