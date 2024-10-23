// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sources of a digital signal.

/// A source returns the next signal value each time it is called.
/// When there is no more data, it just returns `None`.
pub trait Source: Sized {
    /// The source's output type.
    type Output;

    /// Produces the next value in the stream of values.
    fn source(&mut self) -> Option<Self::Output>;
}
