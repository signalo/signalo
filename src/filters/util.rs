// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Utility and wrapper filters; infrastructure, not signal transformation.
//!
//! These filters do not alter signal content in the frequency or amplitude domain.
//! Instead, they provide infrastructure services: delay lines, pass-through identity,
//! output caching, and dimensional unit adaptation. They implement
//! [`Filter`](crate::traits::Filter)`<T>` and can be composed with signal-processing
//! filters in pipelines.
//!
//! # When to use which filter
//!
//! | Filter                   | Purpose                                                    |
//! | ------------------------ | ---------------------------------------------------------- |
//! | `delay::Delay`           | Circular buffer history; enables multi-tap and FIR design |
//! | `identity::Identity`     | Transparent pass-through; placeholder in generic code      |
//! | `last::Last`             | Caches the most recent output of an inner filter           |
//! | `uom::Uom`               | Dimensional unit adapter (feature-gated: `dimensioned`)    |
//!
//! - **Delay** stores a fixed-length history of past samples in a circular buffer.
//!   It is the building block for multi-tap FIR filters, comb filters, and any
//!   computation that needs access to `input[n−k]` for `k > 0`.
//! - **Identity** returns its input unchanged. Useful as a default/placeholder in
//!   generic code where a `Filter<T>` is required but no transformation is needed.
//! - **Last** wraps another filter and caches its most recent output, allowing
//!   repeated access without re-computation. Useful when multiple downstream consumers
//!   need the same filter output.
//! - **Uom** bridges dimensional analysis with the `dimensioned` crate. When the
//!   `dimensioned` feature is enabled, it adapts filters to work with unit-annotated
//!   numeric types. Not available without the feature gate.
//!
//! # See also
//!
//! - [`super::ops`]: per-sample arithmetic operations; structural counterpart to these
//!   utility filters.
//! - [`super::fir::comb::FeedforwardComb`]: uses `Delay` internally for its feedforward
//!   delay line.

pub mod delay;
pub mod identity;
pub mod last;
pub mod uom;
pub(crate) mod window;
