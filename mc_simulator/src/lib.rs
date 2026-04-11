/// mc_simulator — Rust Monte Carlo trajectory sampler for dropout risk projection.
///
/// Build:
///   cd mc_simulator && maturin develop --release
///
/// Python usage:
///   from mc_simulator import simulate_trajectories
///   projected = simulate_trajectories(
///       current_features=[2.0, 45.0, 12, 0.6, 55.0, 0.7, 0.65, 7, 1, 6, 2, 1],
///       transition_deltas=delta_pool,   # list of list[f64]
///       bounds_lo=[0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 1, 0, 0],
///       bounds_hi=[7.0, 180.0, 9999, 1.0, 100.0, 1.0, 1.0, 60, 20, 52, 20, 50],
///       n_simulations=10_000,
///       n_steps=3,
///       seed=42,
///   )
///   # Returns List[List[f64]] — n_simulations × n_features projected endpoint features

use pyo3::prelude::*;
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

/// Simulate N forward trajectories from a given feature vector.
///
/// Each trajectory:
///   1. Randomly samples a delta row from `transition_deltas` at each step.
///   2. Adds the delta to the current features.
///   3. Clips features to `[bounds_lo, bounds_hi]`.
///
/// Parameters
/// ----------
/// current_features   : Vec<f64>        — starting feature vector (length F)
/// transition_deltas  : Vec<Vec<f64>>   — pool of empirical per-step deltas (N_pool × F)
/// bounds_lo          : Vec<f64>        — lower clip bounds (length F)
/// bounds_hi          : Vec<f64>        — upper clip bounds (length F)
/// n_simulations      : usize           — number of trajectories to simulate
/// n_steps            : usize           — number of time-steps per trajectory
/// seed               : u64             — RNG seed for reproducibility
///
/// Returns
/// -------
/// Vec<Vec<f64>>   — projected endpoint features, shape (n_simulations, F)
#[pyfunction]
fn simulate_trajectories(
    current_features:  Vec<f64>,
    transition_deltas: Vec<Vec<f64>>,
    bounds_lo:         Vec<f64>,
    bounds_hi:         Vec<f64>,
    n_simulations:     usize,
    n_steps:           usize,
    seed:              u64,
) -> PyResult<Vec<Vec<f64>>> {
    let n_features  = current_features.len();
    let n_pool      = transition_deltas.len();

    if n_pool == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "transition_deltas must not be empty",
        ));
    }
    if bounds_lo.len() != n_features || bounds_hi.len() != n_features {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bounds_lo and bounds_hi must have the same length as current_features",
        ));
    }

    // Pre-validate all delta rows
    for (i, row) in transition_deltas.iter().enumerate() {
        if row.len() != n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "transition_deltas[{}] has {} elements, expected {}",
                i, row.len(), n_features
            )));
        }
    }

    let results: Vec<Vec<f64>> = (0..n_simulations)
        .into_par_iter()
        .map(|sim_idx| {
            // Each simulation gets a deterministic per-thread RNG derived from seed
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(sim_idx as u64));
            let mut features = current_features.clone();

            for _ in 0..n_steps {
                let delta_idx = rng.gen_range(0..n_pool);
                let delta     = &transition_deltas[delta_idx];

                for f in 0..n_features {
                    features[f] += delta[f];
                    // Clip to bounds
                    if features[f] < bounds_lo[f] { features[f] = bounds_lo[f]; }
                    if features[f] > bounds_hi[f] { features[f] = bounds_hi[f]; }
                }
            }
            features
        })
        .collect();

    Ok(results)
}

/// Python module definition
#[pymodule]
fn mc_simulator(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_trajectories, m)?)?;
    Ok(())
}
