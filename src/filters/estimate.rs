// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! State estimation filters; recursive estimation from noisy measurements.
//!
//! State estimators maintain an internal model of a system's state and update it
//! recursively as new measurements arrive. They combine a prediction step (based on
//! the system dynamics model) with a correction step (based on the measurement residual),
//! producing an estimate that is typically more accurate than either the raw measurement
//! or the model prediction alone.
//!
//! # When to use which filter
//!
//! | Filter                         | Purpose                                                    |
//! | ------------------------------ | ---------------------------------------------------------- |
//! | `kalman::Kalman`               | Optimal linear estimator with known noise covariances      |
//! | `alpha_beta::AlphaBeta`        | Fixed-gain observer (lighter-weight alternative)           |
//!
//! - **Kalman** is the optimal linear estimator when process and measurement noise are
//!   Gaussian with known covariance. The filter adapts its gain (Kalman gain) online
//!   based on the evolving state and measurement uncertainty. Use when you can
//!   characterize sensor noise and system dynamics quantitatively, and you need
//!   statistically optimal estimates.
//! - **AlphaBeta** (also known as the `α`-`β` filter or `g`-`h` filter) uses fixed
//!   gains instead of computing the optimal Kalman gain. It tracks position and velocity
//!   with two tuning parameters. Much simpler to implement and tune than a full Kalman
//!   filter, and sufficient for many tracking and smoothing applications where the
//!   noise statistics are constant or unknown.
//!
//! # See also
//!
//! - [`super::iir::first_order`]: simpler single-pole smoothing without an explicit system model.
//! - [`super::fir::mean`]: non-recursive smoothing; no internal state model, just a windowed
//!   average.

pub mod alpha_beta;

pub mod kalman;
