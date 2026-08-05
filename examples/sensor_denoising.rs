use signalo::filters::fir::mean::MeanArray;
use signalo::sources::noise::Noise;
use signalo::traits::{Filter, Source};

/// Scales a `Noise` source's raw `u32` output to `[-half_width, +half_width]`.
fn noise_to_range(raw: u32, half_width: f64) -> f64 {
    (f64::from(raw) / f64::from(u32::MAX) - 0.5) * 2.0 * half_width
}

/// Returns the slowly-varying ground-truth temperature reading in °C.
fn ground_truth(i: usize) -> f64 {
    20.0 + 5.0 * (2.0 * core::f64::consts::PI * i as f64 / 50.0).sin()
}

fn main() {
    let mut noise = Noise::new(42);
    let mut filter: MeanArray<f64, 8> = MeanArray::default();

    let mut raw_sq_err_sum = 0.0;
    let mut smoothed_sq_err_sum = 0.0;

    for i in 0..200 {
        let truth = ground_truth(i);
        let raw_noise = noise.source().unwrap();
        let noise_val = noise_to_range(raw_noise, 2.0);
        let raw = truth + noise_val;
        let smoothed = filter.filter(raw);

        raw_sq_err_sum += (raw - truth).powi(2);

        // Compensate for the moving average filter's group delay of 3.5 samples
        // to compare the smoothed value against the phase-aligned ground truth.
        let smoothed_truth = if i >= 4 { ground_truth(i - 4) } else { truth };

        smoothed_sq_err_sum += (smoothed - smoothed_truth).powi(2);

        if i % 20 == 0 {
            println!(
                "Sample {}: ground-truth = {}, raw = {}, smoothed = {}",
                i, truth, raw, smoothed
            );
        }
    }

    let raw_rms_error = (raw_sq_err_sum / 200.0).sqrt();
    let smoothed_rms_error = (smoothed_sq_err_sum / 200.0).sqrt();

    println!("Raw RMS error: {}", raw_rms_error);
    println!("Smoothed RMS error: {}", smoothed_rms_error);

    assert!(
        smoothed_rms_error < raw_rms_error,
        "smoothing must reduce RMS error"
    );
}
