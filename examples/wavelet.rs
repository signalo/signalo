// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Analyzing a single impulse with the `(f32, 4)` Daubechies bank and synthesizing it back.
//!
//! The analysis emits a decomposition per input sample, and the matching synthesis consumes
//! one decomposition per call. The pair delays the impulse by three samples while preserving
//! its amplitude, so an impulse at index 8 of [`INPUT`] reappears at index 11 of the
//! reconstruction.
//!
//! Run with `cargo run --example wavelet`.
//!
//! [`AnalyzeArray`]: signalo::filters::wavelet::analyze::AnalyzeArray
//! [`SynthesizeArray`]: signalo::filters::wavelet::synthesize::SynthesizeArray

use signalo::filters::wavelet::analyze::AnalyzeArray;
use signalo::filters::wavelet::daubechies::Daubechies;
use signalo::filters::wavelet::synthesize::SynthesizeArray;
use signalo::traits::Filter;

/// A signal that is zero everywhere except for a single impulse at index 8.
const INPUT: [f32; 24] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
];

/// The reconstruction the `(f32, 4)` Daubechies analysis and synthesis pair emits for [`INPUT`].
///
/// The pair delays the impulse by three samples and preserves its amplitude, so the only
/// nonzero entry sits at index 11 with the input's magnitude.
const EXPECTED: [f32; 24] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
];

/// The half-open input range that the impulse's detail-band energy occupies.
const DETAIL_RANGE: core::ops::Range<usize> = 8..12;

/// Below this magnitude a detail-band sample counts as zero once it is clear of the impulse.
const ZERO_THRESHOLD: f32 = 1.0e-3;

/// The smallest acceptable peak magnitude within [`DETAIL_RANGE`].
///
/// The pair peaks at `0.14787659` there, so the bound sits below the measured peak.
const DETAIL_PEAK: f32 = 0.14;

/// The largest acceptable deviation of the reconstruction from [`EXPECTED`].
const MAX_ERROR: f32 = 1.0e-3;

fn main() {
    let mut analyze = AnalyzeArray::<f32, 4>::daubechies();
    let mut synthesize = SynthesizeArray::<f32, 4>::daubechies();

    let mut low_band = Vec::with_capacity(INPUT.len());
    let mut high_band = Vec::with_capacity(INPUT.len());
    let mut reconstructed = Vec::with_capacity(INPUT.len());

    // Each input sample yields one decomposition and one reconstructed sample, so the
    // bands and the reconstruction all have the same length as the input.
    for sample in INPUT {
        let decomposition = analyze.filter(sample);

        // Both fields are plain `f32`s, so they can be read out before the decomposition
        // itself is moved into the synthesis.
        let low = decomposition.low;
        let high = decomposition.high;
        let rebuilt = synthesize.filter(decomposition);

        low_band.push(low);
        high_band.push(high);
        reconstructed.push(rebuilt);
    }

    let max_error = reconstructed
        .iter()
        .zip(EXPECTED)
        .fold(0.0_f32, |error, (rebuilt, expected)| {
            error.max((rebuilt - expected).abs())
        });

    let detail_peak = high_band[DETAIL_RANGE.clone()]
        .iter()
        .fold(0.0_f32, |peak, detail| peak.max(detail.abs()));

    // The window prints the delay the pair introduces around the impulse's position.
    println!(
        "max_error: {max_error:.3e}, low[8..16]: {:?}, high[8..16]: {:?}, reconstructed[8..16]: {:?}",
        &low_band[8..16],
        &high_band[8..16],
        &reconstructed[8..16],
    );

    assert_eq!(low_band.len(), INPUT.len());
    assert_eq!(high_band.len(), INPUT.len());
    assert_eq!(reconstructed.len(), INPUT.len());
    assert!(
        max_error < MAX_ERROR,
        "reconstruction error {max_error} exceeds tolerance"
    );
    assert_eq!(
        reconstructed
            .iter()
            .position(|value| value.abs() > MAX_ERROR),
        Some(11),
        "the reconstruction does not carry a single impulse at the delayed position"
    );
    assert!(
        detail_peak >= DETAIL_PEAK,
        "the detail band peaks at {detail_peak}, below the expected {DETAIL_PEAK}"
    );
    assert!(
        high_band
            .iter()
            .enumerate()
            .all(|(index, detail)| DETAIL_RANGE.contains(&index) || detail.abs() < ZERO_THRESHOLD),
        "the detail band carries energy outside the impulse's window"
    );
}
