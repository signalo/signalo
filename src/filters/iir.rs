// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Infinite Impulse Response (IIR) filters with feedback.
//!
//! All filters in this module implement recursive difference equations: the output at
//! each sample depends on both current/past inputs and past outputs.
//!
//! # Sign conventions
//!
//! Each filter uses its own natural coefficient convention. These are **not** interchangeable:
//!
//! | Filter         | Feedback term     | Convention                      | Stable when   |
//! | -------------- | ----------------- | ------------------------------- | ------------- |
//! | `first_order`  | `− a1 · y[n−1]`   | Subtractive (Audio EQ Cookbook) | `\|a1\| < 1`  |
//! | `dc_blocker`   | `+ r · y[n−1]`    | Additive (pole radius)          | `0 < r < 1`   |
//! | `comb`         | `+ fb · y[n−D]`   | Additive (Schroeder)            | `\|fb\| < 1`  |
//! | `allpass`      | mixed `c` form    | Schroeder single-multiply       | `\|c\| < 1`   |
//!
//! **Important:** `first_order` subtracts its feedback coefficient
//! (a stable pole at `z = p` requires `a1 = −p`). `dc_blocker`, `comb`, and `allpass`
//! add their feedback coefficient directly (`r = +p` places the pole at `z = p`, stable
//! when `0 < p < 1`). Do not reuse a coefficient value from one group in the other.
//!
//! # When to use which filter
//!
//! - `first_order`: generic first-order IIR; use when you need to set `b0`, `b1`, `a1`
//!   directly or when computing them from physics/control-system equations.
//! - `dc_blocker`: convenience wrapper for the specific `H(z) = (1−z⁻¹)/(1−r·z⁻¹)` form.
//! - `allpass`: phase manipulation without gain change; use in reverb networks and
//!   crossover/all-pass EQ chains.
//! - `comb`: resonant delay-line filter; use in reverb, flanger, and chorus effects.
//!
//! # See also
//!
//! The `biquad` module provides second-order filters using the Direct Form II Transposed
//! topology, including `biquad::Biquad` (raw coefficient form) and higher-level designs
//! such as Butterworth and Chebyshev low-pass, high-pass, and band-pass filters. Prefer
//! `biquad` over `first_order` when you need factored EQ-cookbook coefficients or greater
//! than first-order roll-off.

pub mod first_order;

pub mod biquad;

pub mod envelope;

pub mod exp;

pub mod allpass;

pub mod comb;

pub mod integrate;

pub mod loop_filter;

pub mod dc_blocker;
