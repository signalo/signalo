// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

pub trait Stateful {
    fn reset(&mut self);
}

pub trait Phase: Sized {
    fn phase_shift(&self) -> isize;
}

/// ## Linear Filter
///
/// A system is called linear if it follows these two principles:
///
/// 1. **Superposition**:
///   Let `x1(t)`, `x2(t)` be the inputs applied to a system and `y1(t)`, `y2(t)` be the outputs.
///   For `x1(t)` the output of the system is `y1(t)` and for `x2(t)` output of the system `y2(t)`
///   then for `x1(t) + x2(t)` if the output of the system is `y1(t) + y2(t)` then system is said
///   to be obeying superposition principle.
///
/// 2. **Homogeneity**:
///   Consider for an input `x(t)` for which output of the system is `y(t)`. Then if for the input
///   `ax(t)` (where a is some constant value) output is `ay(t)` then system is said to be obeying
///   homogeneity principle. Consequence of the homogeneity (or scaling) property is that a zero
///   input to the system yields a zero output.
///
/// If the above two property are satisfied system is said to be a linear system.
pub trait LinearPhase: Phase {
    fn linear_phase_shift() -> isize;
}

/// A time shift in the input does not affect the properties of the output.
/// More specifically, if `f(x(t)) = y(t)`, then `f(x(t-T)) = y(t-T)`, where `T` is the time shift.
///
/// A system is called time-invariant if a time shift (delay or advance) in the input signal
/// causes the same time shift in the output signal. Consider for an input signal `x(t)` the
/// `response(output)` of the system is `y(t)`, then for system to be time invariant, for an input
/// `x(t-k)` `response(output)` should be `y(t-k)` (where `k` is some constant shift in time).
///
/// Time invariance is the property of a system which makes the behavior of the system
/// independent of time. This means the behavior of system does not depend on time at which
/// input is applied. For the discrete time system time invariance is called shift invariance.
pub trait TimeInvariant: Sized {

}

/// A shift invariant system is the discrete equivalent of a time-invariant system,
/// defined such that if `y(n)` is the response of the system to `x(n)`, then `y(n–k)`
/// is the response of the system to `x(n–k)`. That is, in a shift-invariant system the
/// contemporaneous response of the output variable to a given value of the input variable does
/// not depend on when the input occurs; time shifts are irrelevant in this regard.
pub trait ShiftInvariant: Sized {

}

/// A causal filter is a filter with output and internal states that depends only on the current
/// and previous input values. A system that has some dependence on input values from the
/// future (in addition to possible past or current input values) is termed an acausal system,
/// and a system that depends solely on future input values is an anticausal system.
pub trait Causal: Sized {

}

/// Discrete time views values of variables as occurring at distinct, separate "points in time",
/// or equivalently as being unchanged throughout each non-zero region of time ("time period")
/// — that is, time is viewed as a discrete variable. Thus a non-time variable jumps from one
/// value to another as time moves from one time period to the next. This view of time corresponds
/// to a digital clock that gives a fixed reading of 10:37 for a while, and then jumps to a
/// new fixed reading of 10:38, etc. In this framework, each variable of interest is measured
/// once at each time period. The number of measurements between any two time periods is finite.
/// Measurements are typically made at sequential integer values of the variable "time".
///
/// https://en.wikipedia.org/wiki/Discrete_time_and_continuous_time#Discrete_time_2
pub trait DiscreteTime: Sized {

}

/// Continuous time views variables as having a particular value for potentially only an
/// infinitesimally short amount of time. Between any two points in time there are an infinite
/// number of other points in time. The variable "time" ranges over the entire real number line,
/// or depending on the context, over some subset of it such as the non-negative reals.
/// Thus time is viewed as a continuous variable.
///
/// https://en.wikipedia.org/wiki/Discrete_time_and_continuous_time#Continuous_time_2
pub trait ContinuousTime: Sized {

}
