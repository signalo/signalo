// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Padding source that extends sequences with padding values.
//!
//! Appends repeated padding values to another source's output, useful for extending signals
//! with silence, edge values, or zero-padding.

pub mod constant;
pub mod edge;
