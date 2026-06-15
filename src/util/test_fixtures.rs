// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Common test fixtures.
//!
//! These fixtures are shared across multiple test suites to avoid
//! copy-paste duplication and reduce maintenance burden.

#![allow(clippy::unreadable_literal)]

use alloc::vec;
use alloc::vec::Vec;

/// A 50-element test sequence based on the Collatz conjecture.
///
/// Provides a deterministic non-trivial input signal with varied values,
/// used as a shared test vector across filter test suites.
pub(crate) fn collatz() -> Vec<f32> {
    vec![
        0.0, 1.0, 7.0, 2.0, 5.0, 8.0, 16.0, 13.0, 19.0, 6.0, 14.0, 9.0, 9.0, 17.0, 17.0, 4.0, 12.0,
        20.0, 20.0, 7.0, 7.0, 15.0, 15.0, 10.0, 23.0, 10.0, 111.0, 180.0, 108.0, 18.0, 106.0, 5.0,
        26.0, 13.0, 13.0, 21.0, 21.0, 21.0, 34.0, 8.0, 109.0, 8.0, 29.0, 16.0, 16.0, 16.0, 104.0,
        11.0, 24.0, 24.0,
    ]
}
