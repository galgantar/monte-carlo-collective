# Monte Carlo Collective: 3D Queens Problem

This project implements Monte Carlo Markov Chain (MCMC) methods to solve the 3D Queens problem (eg placing $N^2$ queens on $N \times N \times N$ board) using simulated annealing with various beta scheduling strategies.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All experiments are configured via `config.yaml`. Key parameters:

- `experiment_type`: Type of experiment to run
- `common.n_steps`: Number of MCMC steps per chain
- `common.n_runs`: Number of independent runs
- `common.initialization`: `"random"`, `"latin"`, or `"klarner"`
- `common.mcmc_type`: `"board"` or `"full_3d"`
- `common.betta_scheduling.type`: `"constant"`, `"linear_annealing"`, `"exponential_annealing"`, `"logarithmic_annealing"`, or `"sinusoidal_annealing"`
- `common.betta_scheduling.beta_start` / `beta_end`: Temperature schedule parameters

## Experiment Types

### `single_N`

Run experiments for a single board size N. Compares different beta schedules if multiple are specified.

**Required parameters:**
- `single_N.N`: Board size

### `measure_min_energy_vs_N`

Measure minimal energy reached as a function of board size N across multiple initialization modes.

**Required parameters:**
- `measure_min_energy_vs_N.Ns`: List of board sizes to test
- `measure_min_energy_vs_N.init_modes`: List of initialization modes to compare

### `beta_start_end_pairs`

Compare different beta_start/beta_end pairs with a fixed annealing schedule type.

**Required parameters:**
- `beta_start_end_pairs.N`: Board size
- `beta_start_end_pairs.beta_start_ends`: List of [beta_start, beta_end] pairs
- `beta_start_end_pairs.annealing_type`: Type of annealing schedule

### `compare_beta_end`

Compare beta_end values for two different board sizes side by side.

**Required parameters:**
- `compare_beta_end.Ns`: List of exactly two board sizes
- `compare_beta_end.beta_start_ends`: List of [beta_start, beta_end] pairs
- `compare_beta_end.annealing_type`: Type of annealing schedule

## Running experiments

```bash
python experiments.py
```

Results are saved to `figures/` and `results/` folders.
