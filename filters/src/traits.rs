// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Traits for **classification of filters**.

/// Trait for **stateful** systems.
///
/// # Background:
///
/// Stateful systems **can react to the same input differently depending on the current state**.
pub trait Stateful {
    /// Resets the current state of `self`.
    fn reset(&mut self);
}

/// Trait for **arbitrarily phase shifting** systems.
///
/// # Background:
///
/// Phase shift is any change that occurs in the phase of one quantity,
/// or in the phase difference between two or more quantities.
///
/// This symbol: `φ` is sometimes referred to as a phase shift or phase offset
/// because it represents a "shift" from zero phase.
///
/// For infinitely long sinusoids, a change in `φ` is the same as a shift in time,
/// such as a time delay. If `x(t)` is delayed (time-shifted) by `1/4` of its cycle, it becomes:
///
/// ```plain
/// x(t - 1/4 T) = A * cos(2π * f(t - 1/4 T) + φ)
///              = A * cos(2π * f * t - π/2 + φ)
/// ```
///
/// whose "phase" is now `φ - π/2`. It has been shifted by `π/2` radians
/// (the variable `A` here just represents the amplitude of the wave).
/// <sup>[Wikipedia](https://en.wikipedia.org/wiki/Phase_(waves)#Phase_shift)</sup>
pub trait PhaseShift: Sized {
    /// Returns the current phase shift of `self`.
    fn phase_shift(&self) -> isize;
}

/// Trait for **linearly phase shifting** systems.
///
/// # Background:
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
/// <sup>[Wikipedia](https://en.wikipedia.org/wiki/Linear_system)</sup>
pub trait LinearPhaseShift: PhaseShift {
    /// Returns the constant linear phase shift of `self`.
    fn linear_phase_shift(&self) -> isize;
}

/// Trait for **time-invariant** systems.
///
/// # Background:
///
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
/// <sup>[Wikipedia](https://en.wikipedia.org/wiki/Time-invariant_system)</sup>
pub trait TimeInvariant: DiscreteTime {

}

/// Trait for **shift-invariant** filters.
///
/// # Background:
///
/// A shift invariant system is the **discrete equivalent of a time-invariant system
/// (see [`TimeInvariant`](trait.TimeInvariant.html))**, defined such that if `y(n)` is the response of the system to `x(n)`,
/// then `y(n–k)` is the response of the system to `x(n–k)`.
///
/// That is, in a shift-invariant system
/// the contemporaneous response of the output variable to a given value of the input variable
/// does not depend on when the input occurs; time shifts are irrelevant in this regard.
/// <sup>[Wikipedia](https://en.wikipedia.org/wiki/Shift-invariant_system)</sup>
pub trait ShiftInvariant: ContinuousTime {

}

/// Trait for **causal** systems.
///
/// # Background:
///
/// A causal filter is a filter with **output and internal states that depends only on the current
/// and previous input values**.
///
/// Contrarily a system that **has some dependence on input values from the future**
/// (in addition to possible past or current input values) **is termed an acausal system**,
/// and a system that **depends solely on future input values is an anticausal system**.
/// <sup>[Wikipedia](https://en.wikipedia.org/wiki/Causal_filter)</sup>
pub trait Causal: Sized {

}

/// Trait for **discrete-time** systems.
///
/// # Background:
///
/// Discrete time views values of variables as **occurring at distinct,
/// separate _points in time_** ("samples" or "time period"), or equivalently as being unchanged
/// throughout each sample — that is, **time is viewed as a discrete variable**.
///
/// Thus a non-time variable **jumps from one value to another** as **time moves from one
/// sample/time period to the next**. (This view of time corresponds to a digital clock
/// that gives a fixed reading of 10:37 for a while, and then jumps to a
/// new fixed reading of 10:38, etc.)
///
/// In this framework, each **variable of interest is measured once at each time period**.
/// The **number of measurements** between any two time periods **is finite**.
///
/// Measurements are typically made at sequential integer values of the variable "time".
/// <sup>[Wikipedia](https://en.wikipedia.org/wiki/Discrete_time_and_continuous_time#Discrete_time_2)</sup>
pub trait DiscreteTime: Sized {

}

/// Trait for **continuous-time** systems.
///
/// # Background:
///
/// Continuous time views variables as having a particular value for potentially only an
/// **infinitesimally short amount of time**.
///
/// Between any two points in time there are an **infinite number of other points in time**.
///
/// The variable "time" **ranges over the non-negative entire real number line**, or depending on the context,
/// over **some subset of it**. Thus **time is viewed as a continuous variable**.
/// <sup>[Wikipedia](https://en.wikipedia.org/wiki/Discrete_time_and_continuous_time#Continuous_time_2)</sup>
pub trait ContinuousTime: Sized {

}
