// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use signalo::filters::iir::exp::TimedMean;
use signalo::filters::iir::integrate::Trapezoidal;
use signalo::filters::iir::peak_hold::TimedPeakHold;
use signalo::sources::noise::Noise;
use signalo::traits::{Filter, Source};
use signalo::Timed;

/// Scales a `Noise` source's raw `u32` output to `[-half_width, +half_width]`.
fn noise_to_range(raw: u32, half_width: f64) -> f64 {
    (f64::from(raw) / f64::from(u32::MAX) - 0.5) * 2.0 * half_width
}

fn ground_rate(t: f64) -> f64 {
    100.0 + 50.0 * (-((t - 30.0).powi(2)) / (2.0 * 8.0 * 8.0)).exp()
}

fn main() {
    let mut jitter_noise = Noise::new(11);
    let mut measurement_noise = Noise::new(22);

    let mut mean_filter = TimedMean::from_time_constant(5.0);
    let mut integrator = Trapezoidal::<f64>::default();
    let mut ground_truth_integrator = Trapezoidal::<f64>::default();
    let mut peak_hold = TimedPeakHold::from_time_constant(10.0);

    let mut sample_index = 0;
    let mut t = 0.0_f64;

    let mut max_watermark = 0.0_f64;
    let mut final_watermark = 0.0_f64;
    let mut cumulative_requests = 0.0_f64;
    let mut ground_truth_cumulative = 0.0_f64;

    loop {
        let dt = if sample_index == 0 {
            1.0
        } else {
            1.0 + noise_to_range(
                jitter_noise
                    .source()
                    .expect("failed to source jitter noise"),
                0.4,
            )
        };

        if sample_index > 0 {
            t += dt;
        }

        if t > 90.0 {
            break;
        }

        let rate_truth = ground_rate(t);
        let measured_rate = rate_truth
            + noise_to_range(
                measurement_noise
                    .source()
                    .expect("failed to source measurement noise"),
                5.0,
            );

        let smoothed_rate = mean_filter.filter(Timed::new(measured_rate, dt));
        cumulative_requests = integrator.filter(Timed::new(smoothed_rate, dt));
        ground_truth_cumulative = ground_truth_integrator.filter(Timed::new(rate_truth, dt));

        let watermark = peak_hold.filter(Timed::new(smoothed_rate, dt));

        if watermark > max_watermark {
            max_watermark = watermark;
        }

        if sample_index % 10 == 0 {
            println!(
                "t = {:.3}, rate_truth = {:.3}, measured_rate = {:.3}, smoothed_rate = {:.3}, cumulative_requests = {:.3}, watermark = {:.3}",
                t, rate_truth, measured_rate, smoothed_rate, cumulative_requests, watermark
            );
        }

        final_watermark = watermark;
        sample_index += 1;
    }

    let reconstruction_error_pct =
        (cumulative_requests - ground_truth_cumulative).abs() / ground_truth_cumulative * 100.0;

    println!("Reconstruction error: {:.3}%", reconstruction_error_pct);
    println!("Peak watermark reached: {:.3} (150.00)", max_watermark);
    println!("Final watermark: {:.3} (100.00)", final_watermark);

    assert!(
        reconstruction_error_pct < 10.0,
        "cumulative-count reconstruction error must stay under 10%"
    );
    assert!(
        max_watermark > 100.0 && max_watermark < 150.0,
        "peak watermark must sit strictly between baseline and the true peak"
    );
    assert!(
        (final_watermark - 100.0).abs() < (max_watermark - 100.0).abs(),
        "watermark must have decayed toward baseline by the end of the run"
    );
}
