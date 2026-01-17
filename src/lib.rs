// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! The 'signalo' DSP toolbox crate.
//!
//! A DSP toolbox with focus on embedded environments, providing zero-cost,
//! zero-allocation abstractions for building real-time filtering pipelines.
//!
//! # Core Concepts
//!
//! - **[`traits::Source`]**: Signal generators that produce values on demand
//! - **[`traits::Filter`]**: Signal transformers that accept and produce values
//! - **[`traits::Sink`]**: Signal consumers that process values
//! - **[`traits::Finalize`]**: Extractors that compute final results from sinks
//!
//! # Example
//!
//! ```ignore
//! use signalo::filters::mean::mean::Mean;
//! use signalo::sources::constant::Constant;
//! use signalo::sinks::mean::Mean as MeanSink;
//! use signalo::traits::{Source, Filter, Sink, Finalize};
//!
//! // Create a constant source generating 1.0
//! let source = Constant::new(1.0);
//!
//! // Create a moving average filter (window size 5)
//! let filter = Mean::new(5).unwrap();
//!
//! // Create a sink to compute final mean
//! let sink = MeanSink::new();
//!
//! // Pipeline: source -> filter -> sink
//! ```

#![warn(missing_docs)]
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::cargo,
    clippy::perf,
    clippy::style,
    clippy::correctness,
    clippy::use_self,
    clippy::unimplemented,
    clippy::todo,
    clippy::else_if_without_else,
    clippy::unneeded_field_pattern,
    clippy::unwrap_used,
    clippy::wrong_self_convention
)]
#![no_std]

#[cfg(any(test, feature = "std"))]
extern crate std;

#[cfg(any(test, feature = "alloc"))]
extern crate alloc;

/// Core trait definitions for the signal processing framework.
///
/// Defines the fundamental traits: [`traits::Source`], [`traits::Filter`], [`traits::Sink`],
/// [`traits::Finalize`], and auxiliary traits for configuration, state, and reset operations.
pub mod traits;

/// Filter implementations for signal transformation.
///
/// Contains a variety of filter types including moving averages, median filters, differentiation,
/// integration, convolution, and state observers (Kalman, Alpha-Beta).
pub mod filters;

/// Pipeline composition utilities for assembling filters into sequences.
///
/// Provides adapters and macros for connecting Sources, Filters, and Sinks in composable chains.
pub mod pipes;

/// Signal generator implementations.
///
/// Contains waveform generators, iterator adapters, constants, ramps, and higher-order sources
/// for building complex signal pipelines.
pub mod sources;

/// Signal consumer/accumulator implementations.
///
/// Contains statistics computations, collectors, integration, and reduction operations for
/// extracting results from signal pipelines.
pub mod sinks;

// Re-export core traits at crate root for convenience
pub use self::traits::{Filter, Finalize, Sink, Source};
