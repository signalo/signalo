// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Rank-order and order-statistic filters; robust to outliers.
//!
//! Instead of computing a weighted sum like linear (FIR/IIR) filters, rank-order filters
//! operate on the sorted values within a sliding window. The output is a value at a chosen
//! rank position; typically the minimum, maximum, or median. Because extreme values are
//! ignored rather than averaged in, these filters are naturally robust to outliers and
//! preserve sharp edges in signals.
//!
//! # When to use which filter
//!
//! | Filter                       | Purpose                                                |
//! | ---------------------------- | ------------------------------------------------------ |
//! | `bounds::Bounds`             | Tracks minimum and maximum simultaneously              |
//! | `exp::Exp`                   | Exponential-weighted median (recursive approximation)  |
//! | `hampel::Hampel`             | Outlier detection and replacement (Hampel identifier)  |
//! | `max::Max`                   | Sliding-window maximum                                 |
//! | `median::Median`             | Sliding-window median (canonical rank filter)          |
//! | `min::Min`                   | Sliding-window minimum                                 |
//!
//! - **Median** is the canonical rank-order filter: it rejects isolated outliers while
//!   preserving step edges. Use when you need smoothing that does not blur transitions.
//! - **Min** and **Max** are useful for envelope detection, peak tracking, and
//!   morphological signal processing (erosion/dilation).
//! - **Bounds** provides both min and max in a single filter, cheaper than running two
//!   separate `Min` and `Max` filters on the same window.
//! - **Hampel** detects and replaces outliers by comparing each sample to the local
//!   median and median absolute deviation (MAD), making it effective for cleaning
//!   impulsive noise while preserving the underlying signal structure.
//! - **Exp** is a recursive (IIR-style) approximation of the median that uses `O(1)`
//!   memory per sample rather than `O(window)`. It is lighter weight but less exact.
//!
//! # Median vs. mean: which to use?
//!
//! - Use `median` when your signal has outliers (spikes, glitches) or sharp edges you
//!   want to preserve. The median ignores isolated extreme values entirely.
//! - Use `fir::mean` when outliers are not a concern and you want the minimum-variance
//!   unbiased estimator for Gaussian noise. The mean is cheaper to compute (O(1) update
//!   per sample versus O(log window) for a typical heap-based median).
//!
//! # See also
//!
//! - [`super::fir::mean`]: sliding-window arithmetic mean; faster but outlier-sensitive.
//! - [`super::fir::mean_variance`]: sliding-window mean and variance together.
//! - [`super::classify`]: threshold-based classification, complementary to min/max envelope
//!   detection for triggering discrete events from continuous features.

pub mod bounds;
pub mod exp;
pub mod hampel;
pub mod max;
pub mod median;
pub mod min;
