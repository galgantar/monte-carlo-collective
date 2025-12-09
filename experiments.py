import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

from mcmc import State3DQueens
from mcmc_board import State3DQueensBoard

import logging
from datetime import datetime

def setup_logging():
    log_dir = "outputs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"run_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging initialized → {logfile}")

def constant_beta(beta):
    def schedule(step):
        return beta
    return schedule


def linear_annealing_beta(beta_start, beta_end, n_steps):
    def schedule(step):
        if n_steps <= 1:
            return beta_end
        frac = step / (n_steps - 1)
        return beta_start + frac * (beta_end - beta_start)
    return schedule

def exponential_annealing_beta(beta_start, beta_end, n_steps):
    if n_steps <= 1:
        def schedule(_):
            return beta_end
        return schedule

    log_ratio = np.log(beta_end / beta_start)

    def schedule(step):
        step = np.clip(step, 0, n_steps - 1)
        t = step / (n_steps - 1)
        return beta_start * np.exp(log_ratio * t)

    return schedule

def logarithmic_annealing_beta(beta_start, beta_end, n_steps):
    """
    Logarithmic annealing: beta grows quickly early, then slows down.
    β(step) = β_start + (β_end - β_start) * log(1 + step) / log(1 + n_steps)
    """
    if n_steps <= 1:
        def schedule(_):
            return beta_end
        return schedule

    log_norm = np.log(1 + n_steps)

    def schedule(step):
        step = np.clip(step, 0, n_steps)
        return beta_start + (beta_end - beta_start) * (np.log(1 + step) / log_norm)

    return schedule

def build_schedule_from_params(sched_type, n_steps, beta_const=None, beta_start=None, beta_end=None):
    """
    Build a beta schedule from parameters. This function is picklable and can be used
    in multiprocessing contexts.
    """
    if sched_type == "constant":
        if beta_const is None:
            raise ValueError("beta_const required for constant schedule")
        return constant_beta(beta_const)
    elif sched_type == "linear_annealing":
        if beta_start is None or beta_end is None:
            raise ValueError("beta_start and beta_end required for linear_annealing schedule")
        return linear_annealing_beta(beta_start, beta_end, n_steps)
    elif sched_type == "exponential_annealing":
        if beta_start is None or beta_end is None:
            raise ValueError("beta_start and beta_end required for exponential_annealing schedule")
        return exponential_annealing_beta(beta_start, beta_end, n_steps)
    elif sched_type == "logarithmic_annealing":
        if beta_start is None or beta_end is None:
            raise ValueError("beta_start and beta_end required for logarithmic_annealing schedule")
        return logarithmic_annealing_beta(beta_start, beta_end, n_steps)
    else:
        raise ValueError(f"Unknown betta_scheduling type: {sched_type}")


def build_schedule_from_common(common_cfg, n_steps):
    """
    Build a beta schedule (and base_seed) from common['betta_scheduling'].
    Returns a single schedule, base_seed, description, and schedule_params.
    """
    sched_cfg = common_cfg["betta_scheduling"]
    sched_type = sched_cfg["type"]
    base_seed = sched_cfg.get("base_seed", 0)

    if sched_type == "constant":
        beta_const = sched_cfg["beta_const"]
        schedule_params = {"type": "constant", "beta_const": beta_const}
        desc = f"constant beta={beta_const}"
    elif sched_type == "linear_annealing":
        beta_start = sched_cfg["beta_start"]
        beta_end = sched_cfg["beta_end"]
        schedule_params = {"type": "linear_annealing", "beta_start": beta_start, "beta_end": beta_end}
        desc = f"linear beta: {beta_start}→{beta_end}"
    elif sched_type == "exponential_annealing":
        beta_start = sched_cfg["beta_start"]
        beta_end = sched_cfg["beta_end"]
        schedule_params = {"type": "exponential_annealing", "beta_start": beta_start, "beta_end": beta_end}
        desc = f"exp beta: {beta_start}→{beta_end}"
    elif sched_type == "logarithmic_annealing":
        beta_start = sched_cfg["beta_start"]
        beta_end = sched_cfg["beta_end"]
        schedule_params = {"type": "logarithmic_annealing", "beta_start": beta_start, "beta_end": beta_end}
        desc = f"log beta: {beta_start}→{beta_end}"
    else:
        raise ValueError(f"Unknown betta_scheduling type: {sched_type}")

    beta_schedule = build_schedule_from_params(
        sched_type=sched_type,
        n_steps=n_steps,
        beta_const=schedule_params.get("beta_const"),
        beta_start=schedule_params.get("beta_start"),
        beta_end=schedule_params.get("beta_end"),
    )

    return beta_schedule, base_seed, desc, schedule_params


def build_schedules_from_types(sched_types, sched_cfg, n_steps):
    schedules = []
    base_seed = sched_cfg['base_seed']
    beta_start = sched_cfg['beta_start']
    beta_end = sched_cfg['beta_end']
    beta_const = sched_cfg['beta_const']
    
    for sched_type in sched_types:
        if sched_type == "constant":
            schedule_params = {"type": "constant", "beta_const": beta_const}
            desc = f"constant beta={beta_const}"
            label = f"Constant β={beta_const}"
        elif sched_type == "linear_annealing":
            schedule_params = {"type": "linear_annealing", "beta_start": beta_start, "beta_end": beta_end}
            desc = f"linear beta: {beta_start}→{beta_end}"
            label = f"Linear {beta_start}→{beta_end}"
        elif sched_type == "exponential_annealing":
            schedule_params = {"type": "exponential_annealing", "beta_start": beta_start, "beta_end": beta_end}
            desc = f"exp beta: {beta_start}→{beta_end}"
            label = f"Exponential {beta_start}→{beta_end}"
        elif sched_type == "logarithmic_annealing":
            schedule_params = {"type": "logarithmic_annealing", "beta_start": beta_start, "beta_end": beta_end}
            desc = f"log beta: {beta_start}→{beta_end}"
            label = f"Logarithmic {beta_start}→{beta_end}"
        else:
            raise ValueError(f"Unknown betta_scheduling type: {sched_type}")

        beta_schedule = build_schedule_from_params(
            sched_type=sched_type,
            n_steps=n_steps,
            beta_const=schedule_params.get("beta_const"),
            beta_start=schedule_params.get("beta_start"),
            beta_end=schedule_params.get("beta_end"),
        )

        schedules.append((beta_schedule, base_seed, desc, label, schedule_params))
    
    return schedules


def metropolis_mcmc(N, n_steps, init_mode, beta_schedule, verbose=True, seed=None, Q=None, run_idx=None):
    if seed is not None:
        np.random.seed(seed)

    state = State3DQueens(N, Q=Q, init_mode = init_mode)
    current_energy = state.energy(recompute=True)

    best_state = state.copy()
    best_energy = current_energy

    accepted = 0
    energy_history = [current_energy]
    
    accepted_steps = []
    rejected_steps = []

    if verbose and n_steps > 0:
        next_report = max(1, n_steps // 10)

    prefix = f"[chain #{run_idx}]" if run_idx is not None else "[chain]"

    for step in range(n_steps):
        beta_t = beta_schedule(step)

        q_idx = np.random.randint(0, state.Q)

        old_conflicts = state.conflicts_for_queen(q_idx)

        N_ = state.N
        while True:
            i_new = np.random.randint(0, N_)
            j_new = np.random.randint(0, N_)
            k_new = np.random.randint(0, N_)
            if (int(i_new), int(j_new), int(k_new)) not in state.occ_set:
                break

        new_conflicts = state.conflicts_for_queen(q_idx, pos=(i_new, j_new, k_new))

        delta_E = new_conflicts - old_conflicts
        proposed_energy = current_energy + delta_E

        accept_prob = min(1.0, np.exp(-beta_t * delta_E))
        was_accepted = np.random.random() < accept_prob

        if was_accepted:
            accepted_steps.append(step)
        else:
            rejected_steps.append(step)

        if was_accepted:
            state.propose_move(q_idx, (i_new, j_new, k_new))
            current_energy = proposed_energy
            state._energy = current_energy
            accepted += 1

            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy
        else:
            state._energy = current_energy

        energy_history.append(current_energy)

        if verbose and (step + 1) % next_report == 0:
            frac = (step + 1) / n_steps
            logging.info(
                f"{prefix} {step + 1}/{n_steps} steps "
                f"({frac*100:.1f}%) | E={current_energy}, best={best_energy}, "
                f"beta_t={beta_t:.3f}"
            )

    if verbose and n_steps > 0:
        logging.info(f"{prefix} Final energy: {current_energy}")
        logging.info(f"{prefix} Best energy:   {best_energy}")
        logging.info(f"{prefix} Acceptance rate: {accepted / n_steps:.3f}")

    energy_arr = np.array(energy_history)
    steps_to_best = int(np.argmin(energy_arr))

    return {
        "final_state": state,
        "final_energy": current_energy,
        "best_state": best_state,
        "best_energy": best_energy,
        "energy_history": energy_history,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "steps_to_best": steps_to_best,
    }


def metropolis_mcmc_board(N, n_steps, init_mode, beta_schedule, verbose=True, seed=None, run_idx=None):
    """MCMC for board-constrained version (one queen per (i,j) pair)."""
    if seed is not None:
        np.random.seed(seed)

    state = State3DQueensBoard(N, init_mode=init_mode)
    current_energy = state.energy(recompute=True)

    best_state = state.copy()
    best_energy = current_energy

    accepted = 0
    energy_history = [current_energy]
    
    accepted_steps = []
    rejected_steps = []

    if verbose and n_steps > 0:
        next_report = max(1, n_steps // 10)

    prefix = f"[chain #{run_idx}]" if run_idx is not None else "[chain]"

    for step in range(n_steps):
        beta_t = beta_schedule(step)

        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        old_k = state.heights[i, j]

        old_conflicts = state.conflicts_for_position(i, j, old_k)

        new_k = np.random.randint(0, N)
        while new_k == old_k:
            new_k = np.random.randint(0, N)

        new_conflicts = state.conflicts_for_position(i, j, new_k)

        delta_E = new_conflicts - old_conflicts
        proposed_energy = current_energy + delta_E

        accept_prob = min(1.0, np.exp(-beta_t * delta_E))
        was_accepted = np.random.random() < accept_prob

        if was_accepted:
            accepted_steps.append(step)
        else:
            rejected_steps.append(step)

        if was_accepted:
            state.propose_move(i, j, new_k)
            current_energy = proposed_energy
            state._energy = current_energy
            accepted += 1

            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy
        else:
            pass

        energy_history.append(current_energy)

        if verbose and (step + 1) % next_report == 0:
            frac = (step + 1) / n_steps
            logging.info(
                f"{prefix} {step + 1}/{n_steps} steps "
                f"({frac*100:.1f}%) | E={current_energy}, best={best_energy}, "
                f"beta_t={beta_t:.3f}"
            )

    if verbose and n_steps > 0:
        logging.info(f"{prefix} Final energy: {current_energy}")
        logging.info(f"{prefix} Best energy:   {best_energy}")
        logging.info(f"{prefix} Acceptance rate: {accepted / n_steps:.3f}")

    energy_arr = np.array(energy_history)
    steps_to_best = int(np.argmin(energy_arr))

    return {
        "final_state": state,
        "final_energy": current_energy,
        "best_state": best_state,
        "best_energy": best_energy,
        "energy_history": energy_history,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "steps_to_best": steps_to_best,
    }


def run_single_chain(N, n_steps, init_mode, beta_schedule, seed=None, verbose=False, run_idx=None):
    return metropolis_mcmc(
        N=N,
        n_steps=n_steps,
        init_mode = init_mode,
        beta_schedule=beta_schedule,
        verbose=verbose,
        seed=seed,
        run_idx=run_idx,
    )


def run_single_chain_board(N, n_steps, init_mode, beta_schedule, seed=None, verbose=False, run_idx=None):
    return metropolis_mcmc_board(
        N=N,
        n_steps=n_steps,
        init_mode=init_mode,
        beta_schedule=beta_schedule,
        verbose=verbose,
        seed=seed,
        run_idx=run_idx,
    )


def run_single_chain_multithread(args):
    """Wrapper function for parallel execution of a single chain (full_3d)."""
    (N, n_steps, init_mode, schedule_params, seed, verbose, run_idx) = args
    beta_schedule = build_schedule_from_params(
        sched_type=schedule_params["type"],
        n_steps=n_steps,
        beta_const=schedule_params.get("beta_const"),
        beta_start=schedule_params.get("beta_start"),
        beta_end=schedule_params.get("beta_end"),
    )
    start_time = time.time()
    res = run_single_chain(
        N=N,
        n_steps=n_steps,
        init_mode=init_mode,
        beta_schedule=beta_schedule,
        seed=seed,
        verbose=verbose,
        run_idx=run_idx,
    )
    end_time = time.time()
    duration = end_time - start_time
    return {
        "run_idx": run_idx,
        "energy_history": res["energy_history"],
        "best_energy": res["best_energy"],
        "duration": duration,
        "accepted_steps": res["accepted_steps"],
        "rejected_steps": res["rejected_steps"],
        "steps_to_best": res["steps_to_best"],
    }


def run_single_chain_board_multithread(args):
    """Wrapper function for parallel execution of a single chain (board)."""
    (N, n_steps, init_mode, schedule_params, seed, verbose, run_idx) = args
    beta_schedule = build_schedule_from_params(
        sched_type=schedule_params["type"],
        n_steps=n_steps,
        beta_const=schedule_params.get("beta_const"),
        beta_start=schedule_params.get("beta_start"),
        beta_end=schedule_params.get("beta_end"),
    )
    start_time = time.time()
    res = run_single_chain_board(
        N=N,
        n_steps=n_steps,
        init_mode=init_mode,
        beta_schedule=beta_schedule,
        seed=seed,
        verbose=verbose,
        run_idx=run_idx,
    )
    end_time = time.time()
    duration = end_time - start_time
    return {
        "run_idx": run_idx,
        "energy_history": res["energy_history"],
        "best_energy": res["best_energy"],
        "duration": duration,
        "accepted_steps": res["accepted_steps"],
        "rejected_steps": res["rejected_steps"],
        "steps_to_best": res["steps_to_best"],
    }


def run_experiment(N, n_steps, init_mode, beta_schedule, n_runs, base_seed=0, verbose=False, n_workers=None, schedule_params=None, mcmc_type="full_3d"):
    """
    Run multiple MCMC chains in parallel or sequentially.
    
    Args:
        n_workers: Number of parallel workers. If None, uses CPU count for parallel execution.
        schedule_params: Dict with schedule parameters for parallel execution. Required when n_runs > 1.
            Must contain:
            - "type": "constant", "linear_annealing", or "exponential_annealing"
            - "beta_const": for constant schedules
            - "beta_start", "beta_end": for annealing schedules
        schedule_params is required for parallel execution because schedule functions are closures
        that can't be pickled for multiprocessing.
        mcmc_type: "full_3d" (uses mcmc.py) or "board" (uses mcmc_board.py)
    """
    all_histories = []
    best_energies = []
    run_times = []
    all_accepted_steps = []
    all_rejected_steps = []
    all_steps_to_best = []
    
    if mcmc_type == "board":
        chain_runner = run_single_chain_board
        multithread_runner = run_single_chain_board_multithread
    else:
        chain_runner = run_single_chain
        multithread_runner = run_single_chain_multithread
    
    if n_runs > 1:
        if schedule_params is None:
            raise ValueError(f"schedule_params is required for parallel execution when n_runs > 1")
        run_args = [
            (N, n_steps, init_mode, schedule_params, base_seed + r, verbose, r)
            for r in range(n_runs)
        ]
        
        if verbose:
            logging.info(f"\n>>> Running {n_runs} chains in parallel ({mcmc_type})...")
        
        start_total = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_run = {
                executor.submit(multithread_runner, args): r
                for r, args in enumerate(run_args)
            }
            
            results = [None] * n_runs
            completed = 0
            
            for future in as_completed(future_to_run):
                run_idx = future_to_run[future]
                try:
                    result = future.result()
                    result["run_idx"] = run_idx
                    results[run_idx] = result
                    completed += 1
                    
                    if verbose:
                        logging.info(
                            f"=== Completed run {completed}/{n_runs} "
                            f"({completed*100/n_runs:.1f}% of experiment) | "
                            f"time this run: {result['duration']:.2f} s ==="
                        )
                except Exception as exc:
                    logging.info(f"Run {run_idx} generated an exception: {exc}")
                    raise
        
        end_total = time.time()
        total_time = end_total - start_total
        
        results.sort(key=lambda x: x["run_idx"])
        
        for result in results:
            all_histories.append(result["energy_history"])
            best_energies.append(result["best_energy"])
            run_times.append(result["duration"])
            all_accepted_steps.append(result["accepted_steps"])
            all_rejected_steps.append(result["rejected_steps"])
            all_steps_to_best.append(result["steps_to_best"])
        
        if verbose and n_runs > 0:
            mean_time = np.mean(run_times)
            logging.info(f"\n>>> Mean time per run: {mean_time:.2f} s")
            logging.info(f">>> Total time for {n_runs} runs: {total_time:.2f} s")
    else:
        for r in range(n_runs):
            if verbose:
                logging.info(f"\n=== Run {r+1}/{n_runs} ===")

            start_time = time.time()
            res = chain_runner(
                N=N,
                n_steps=n_steps,
                init_mode=init_mode,
                beta_schedule=beta_schedule,
                seed=base_seed + r,
                verbose=verbose,
                run_idx=r,
            )
            end_time = time.time()

            duration = end_time - start_time
            run_times.append(duration)

            all_histories.append(res["energy_history"])
            best_energies.append(res["best_energy"])
            all_accepted_steps.append(res["accepted_steps"])
            all_rejected_steps.append(res["rejected_steps"])
            all_steps_to_best.append(res["steps_to_best"])

            if verbose:
                frac_runs = (r + 1) / n_runs
                logging.info(
                    f"=== Completed run {r+1}/{n_runs} "
                    f"({frac_runs*100:.1f}% of experiment) | "
                    f"time this run: {duration:.2f} s ==="
                )

        if verbose and n_runs > 0:
            mean_time = np.mean(run_times)
            total_time = np.sum(run_times)
            logging.info(f"\n>>> Mean time per run: {mean_time:.2f} s")
            logging.info(f">>> Total time for {n_runs} runs: {total_time:.2f} s")

    return all_histories, best_energies, run_times, all_accepted_steps, all_rejected_steps, all_steps_to_best


def plot_energy_histories(all_histories, title, out_path=None, schedule_labels=None):
    if isinstance(all_histories, dict):
        schedule_labels = schedule_labels or list(all_histories.keys())
        histories_dict = all_histories
    else:
        schedule_labels = schedule_labels or ["Schedule"]
        histories_dict = {schedule_labels[0]: all_histories}
    
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, label in enumerate(schedule_labels):
        histories = histories_dict[label]
        n_runs = len(histories)
        n_steps_plus1 = len(histories[0])
        
        energies = np.array(histories)
        mean_energy = energies.mean(axis=0)
        std_energy = energies.std(axis=0)
        color = colors[idx % len(colors)]

        df.to_csv(f"results/{label}.csv", index=False)
        
        steps = np.arange(n_steps_plus1)

        # ---- Save CSV for this schedule ----
        os.makedirs("results", exist_ok=True)

        df = pd.DataFrame({
            "step": steps,
            "mean_energy": mean_energy,
            "std_energy": std_energy
        })
        
        plt.plot(
            steps,
            mean_energy,
            linewidth=2.5,
            label=label,
            color=color,
        )
        
        plt.fill_between(
            steps,
            mean_energy - std_energy,
            mean_energy + std_energy,
            alpha=0.25,
            color=color,
        )

    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Energy", fontsize=20)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12, framealpha=0.9, loc='best')
    
    plt.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_acceptance_rates_binned(all_accepted_steps_list, all_rejected_steps_list, n_steps, n_bins=100, title=None, out_path=None, schedule_labels=None):
    """
    Plot acceptance rates (ratio of accepted to total moves) in bins.
    
    Args:
        all_accepted_steps_list: List of lists, each containing step indices where moves were accepted
        all_rejected_steps_list: List of lists, each containing step indices where moves were rejected
        n_steps: Total number of steps
        n_bins: Number of bins to use for binning steps
        title: Plot title
        out_path: Path to save the plot
        schedule_labels: Labels for each schedule (should match energy history plot)
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    bin_edges = np.linspace(0, n_steps, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_edges[-1] = n_steps
    
    for idx, (accepted_steps_runs, rejected_steps_runs) in enumerate(zip(all_accepted_steps_list, all_rejected_steps_list)):
        all_accepted = []
        all_rejected = []
        
        for accepted, rejected in zip(accepted_steps_runs, rejected_steps_runs):
            all_accepted.extend(accepted)
            all_rejected.extend(rejected)
        
        all_accepted = np.array(all_accepted)
        all_rejected = np.array(all_rejected)
        
        acceptance_rates = []
        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            
            if i == len(bin_edges) - 2:
                accepted_in_bin = np.sum((all_accepted >= bin_start) & (all_accepted <= bin_end))
                rejected_in_bin = np.sum((all_rejected >= bin_start) & (all_rejected <= bin_end))
            else:
                accepted_in_bin = np.sum((all_accepted >= bin_start) & (all_accepted < bin_end))
                rejected_in_bin = np.sum((all_rejected >= bin_start) & (all_rejected < bin_end))
            
            total_in_bin = accepted_in_bin + rejected_in_bin
            
            if total_in_bin > 0:
                rate = accepted_in_bin / total_in_bin
            else:
                rate = np.nan
            
            acceptance_rates.append(rate)
        
        acceptance_rates = np.array(acceptance_rates)
        
        if schedule_labels:
            label = schedule_labels[idx]
        else:
            label = f"Schedule {idx+1}"
        
        # ---- Save CSV for this schedule ----
        os.makedirs("results", exist_ok=True)

        df = pd.DataFrame({
            "bin_center": bin_centers,
            "acceptance_rate": acceptance_rates
        })

        df.to_csv(f"results/acceptance_rates_{label}.csv", index=False)
        
        color = colors[idx % len(colors)]
        
        valid_mask = ~np.isnan(acceptance_rates)
        plt.plot(
            bin_centers[valid_mask],
            acceptance_rates[valid_mask],
            linewidth=2.5,
            label=label,
            color=color,
        )
    
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Acceptance Rate", fontsize=20)
    if title:
        plt.title(title, fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12, framealpha=0.9, loc='best')
    
    plt.tight_layout()
    
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def run_beta_start_end_pairs(
    N,
    n_steps,
    beta_start_ends,
    annealing_type="linear_annealing",
    init_mode="random",
    n_runs=5,
    base_seed=0,
    verbose=True,
    plot=True,
    out_path=None,
    out_path_acceptance=None,
    mcmc_type="full_3d",
):
    """
    Run experiments for multiple beta_start/beta_end pairs with fixed annealing schedule.
    
    Args:
        N: Board size
        n_steps: Number of MCMC steps
        beta_start_ends: List of [beta_start, beta_end] pairs, e.g., [[0.1, 1.0], [2.0, 5.0]]
        annealing_type: Type of annealing schedule ("linear_annealing", "logarithmic_annealing", or "exponential_annealing")
        init_mode: Initialization mode
        n_runs: Number of runs per pair
        base_seed: Base seed for random number generation
        verbose: Whether to logging.info progress
        plot: Whether to plot results
        out_path: Path to save the energy history plot
        out_path_acceptance: Path to save the acceptance rate plot
    """
    all_histories_dict = {}
    all_best_energies_dict = {}
    all_accepted_steps_dict = {}
    all_rejected_steps_dict = {}
    
    for idx, (beta_start, beta_end) in enumerate(beta_start_ends):
        if verbose:
            logging.info(f"\n{'='*60}")
            logging.info(f"Pair {idx+1}/{len(beta_start_ends)}: beta_start={beta_start}, beta_end={beta_end}")
            logging.info(f"{'='*60}")
        
        schedule_params = {
            "type": annealing_type,
            "beta_start": beta_start,
            "beta_end": beta_end,
        }
        
        beta_schedule = build_schedule_from_params(
            sched_type=annealing_type,
            n_steps=n_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        
        pair_seed = base_seed + idx * 1000
        
        all_histories, best_energies, run_times, accepted_steps, rejected_steps, steps_to_best = run_experiment(
            N=N,
            n_steps=n_steps,
            init_mode=init_mode,
            beta_schedule=beta_schedule,
            n_runs=n_runs,
            base_seed=pair_seed,
            verbose=verbose,
            schedule_params=schedule_params,
            mcmc_type=mcmc_type,
        )
        
        label = f"β: {beta_start}→{beta_end}"
        all_histories_dict[label] = all_histories
        all_best_energies_dict[label] = best_energies
        all_accepted_steps_dict[label] = accepted_steps
        all_rejected_steps_dict[label] = rejected_steps
        
        if verbose:
            mean_time = np.mean(run_times)
            mean_best = np.mean(best_energies)
            std_best = np.std(best_energies)
            mean_steps = np.mean(steps_to_best)
            std_steps = np.std(steps_to_best)
            logging.info(f"\n[{label}] Mean time per run: {mean_time:.2f} s")
            logging.info(f"[{label}] Best energies: {best_energies}")
            logging.info(f"[{label}] Mean best energy: {mean_best:.2f} ± {std_best:.2f}")
            logging.info(f"[{label}] Steps to best: {steps_to_best}")
            logging.info(f"[{label}] Mean steps to best: {mean_steps:.1f} ± {std_steps:.1f}")
    
    schedule_labels = list(all_histories_dict.keys())
    
    if plot:
        title = f"Energy History for Different β Ranges (N={N}, {annealing_type}, init_mode={init_mode})"
        plot_energy_histories(
            all_histories_dict,
            title=title,
            out_path=out_path,
            schedule_labels=schedule_labels
        )
        
        if out_path_acceptance is not None:
            title_acceptance = f"Acceptance Rate for Different β Ranges (N={N}, {annealing_type}, init_mode={init_mode})"
            accepted_steps_list = [all_accepted_steps_dict[label] for label in schedule_labels]
            rejected_steps_list = [all_rejected_steps_dict[label] for label in schedule_labels]
            plot_acceptance_rates_binned(
                all_accepted_steps_list=accepted_steps_list,
                all_rejected_steps_list=rejected_steps_list,
                n_steps=n_steps,
                n_bins=100,
                title=title_acceptance,
                out_path=out_path_acceptance,
                schedule_labels=schedule_labels
            )
    
    return {
        "all_histories": all_histories_dict,
        "all_best_energies": all_best_energies_dict,
    }

def measure_min_energy_vs_N(
    Ns,
    n_steps,
    beta_schedule,
    schedule_params=None,
    init_modes=["random"],
    n_runs=5,
    base_seed=100,
    verbose=True,
    plot=True,
    out_path=None,
    mcmc_type="full_3d",
):
    if isinstance(init_modes, str):
        init_modes = [init_modes]

    results = {}

    for init_mode in init_modes:
        if verbose:
            logging.info(f"\n{'='*60}")
            logging.info(f"Running experiments with init_mode = '{init_mode}'")
            logging.info(f"{'='*60}")

        mean_min_energies = []
        std_min_energies = []
        all_min_energies = []

        mean_steps_to_best = []
        std_steps_to_best = []
        all_steps_to_best = []

        for idx, N in enumerate(Ns):
            if verbose:
                logging.info(f"\n=== Running N = {N} (init_mode={init_mode}) ===")

            init_mode_offset = sum(ord(c) for c in init_mode) % 1000
            _, best_energies, _, _, _, steps_to_best = run_experiment(
                N=N,
                n_steps=n_steps,
                init_mode=init_mode,
                beta_schedule=beta_schedule,
                n_runs=n_runs,
                base_seed=base_seed + 10 * idx + init_mode_offset,
                verbose=verbose,
                schedule_params=schedule_params,
                mcmc_type=mcmc_type,
            )

            best_energies = np.array(best_energies)
            steps_to_best = np.array(steps_to_best)

            all_min_energies.append(best_energies)

            mean_min_energies.append(best_energies.mean())
            std_min_energies.append(best_energies.std())

            all_steps_to_best.append(steps_to_best)
            mean_steps_to_best.append(steps_to_best.mean())
            std_steps_to_best.append(steps_to_best.std())

            if verbose:
                logging.info(
                    f"  → Mean min energy = {mean_min_energies[-1]:.2f} ± {std_min_energies[-1]:.2f}"
                )
                logging.info(
                    f"  → Steps to best: mean = {mean_steps_to_best[-1]:.1f} ± {std_steps_to_best[-1]:.1f}"
                )

        results[init_mode] = {
            "mean_min_energies": np.array(mean_min_energies),
            "std_min_energies": np.array(std_min_energies),
            "all_min_energies": all_min_energies,
            "mean_steps_to_best": np.array(mean_steps_to_best),
            "std_steps_to_best": np.array(std_steps_to_best),
            "all_steps_to_best": all_steps_to_best,
        }

    if plot:
        Ns_arr = np.array(Ns)
        colors = plt.cm.tab10(np.linspace(0, 1, len(init_modes)))

        plt.figure(figsize=(10, 6))

        for idx, init_mode in enumerate(init_modes):
            mean_energies = results[init_mode]["mean_min_energies"]
            std_energies = results[init_mode]["std_min_energies"]
            color = colors[idx]

            # ---- Save minimal energy results to CSV ----
            os.makedirs("results", exist_ok=True)

            df_energy = pd.DataFrame({
                "N": Ns_arr,
                init_mode + "_mean_min_energy": mean_energies,
                init_mode + "_std_min_energy": std_energies
            })

            df_energy.to_csv(f"results/min_energy_vs_N_{init_mode}.csv", index=False)

            plt.plot(
                Ns_arr,
                mean_energies,
                "o-",
                linewidth=2,
                markersize=6,
                color=color,
                label=f"{init_mode}",
            )

            plt.fill_between(
                Ns_arr,
                mean_energies - std_energies,
                mean_energies + std_energies,
                alpha=0.2,
                color=color,
            )

        plt.xlabel("Board size N", fontsize=20)
        plt.ylabel("Minimal energy reached", fontsize=20)
        plt.title("MCMC: Minimal Energy vs. Board Size N", fontsize=18, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(init_modes)))

        for idx, init_mode in enumerate(init_modes):
            mean_steps = results[init_mode]["mean_steps_to_best"]
            std_steps = results[init_mode]["std_steps_to_best"]
            color = colors[idx]

            # ---- Save convergence results to CSV ----
            os.makedirs("results", exist_ok=True)

            df_steps = pd.DataFrame({
                "N": Ns_arr,
                init_mode + "_mean_steps_to_best": mean_steps,
                init_mode + "_std_steps_to_best": std_steps
            })

            df_steps.to_csv(f"results/steps_to_best_vs_N_{init_mode}.csv", index=False)

            plt.plot(
                Ns_arr,
                mean_steps,
                "o-",
                linewidth=2,
                markersize=6,
                color=color,
                label=f"{init_mode}",
            )
            plt.fill_between(
                Ns_arr,
                mean_steps - std_steps,
                mean_steps + std_steps,
                alpha=0.2,
                color=color,
            )

        plt.xlabel("Board size N", fontsize=20)
        plt.ylabel("Steps to best energy", fontsize=20)
        plt.title("MCMC: Steps to Best Energy vs. Board Size N", fontsize=18, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        if out_path is not None:
            base, ext = os.path.splitext(out_path)
            conv_path = base + "_convergence" + (ext if ext else ".png")
            os.makedirs(os.path.dirname(conv_path), exist_ok=True)
            plt.savefig(conv_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return {
        "Ns": Ns,
        "results": results,
    }



if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    setup_logging()

    experiment_type = config["experiment_type"]
    common = config["common"]
    n_steps = common["n_steps"]
    n_runs = common["n_runs"]
    verbose = common["verbose"]
    init_mode = common["initialization"]
    common_output_path = common["output_path"]
    mcmc_type = common.get("mcmc_type", "board")

    logging.info(f"Initialization mode: {init_mode}")
    logging.info(f"Experiment type: {experiment_type}")
    logging.info(f"MCMC type: {mcmc_type}")

    if experiment_type == "single_N":
        single_cfg = config["single_N"]
        N = single_cfg["N"]

        output_path = common_output_path

        sched_cfg = common["betta_scheduling"]
        sched_type = sched_cfg["type"]
        
        if isinstance(sched_type, list):
            schedules = build_schedules_from_types(sched_type, sched_cfg, n_steps)
            
            logging.info(f"\nRunning {n_runs} runs on N={N} with {len(schedules)} schedule(s), init_mode={init_mode}")
            logging.info(f"All schedules use: beta_start={sched_cfg['beta_start']}, "
                  f"beta_end={sched_cfg['beta_end']}, base_seed={sched_cfg['base_seed']}")
            
            all_histories_dict = {}
            all_best_energies_dict = {}
            all_run_times_dict = {}
            
            for beta_schedule, base_seed, sched_desc, label, schedule_params in schedules:
                if verbose:
                    logging.info(f"\n{'='*60}")
                    logging.info(f"Schedule: {label} ({sched_desc})")
                    logging.info(f"{'='*60}")
                
                all_histories, best_energies, run_times, _, _, _ = run_experiment(
                    N=N,
                    n_steps=n_steps,
                    init_mode=init_mode,
                    beta_schedule=beta_schedule,
                    n_runs=n_runs,
                    base_seed=base_seed,
                    verbose=verbose,
                    schedule_params=schedule_params,
                    mcmc_type=mcmc_type,
                )
                
                all_histories_dict[label] = all_histories
                all_best_energies_dict[label] = best_energies
                all_run_times_dict[label] = run_times
                
                mean_time = np.mean(run_times)
                if verbose:
                    logging.info(f"\n[{label}] Mean time per run: {mean_time:.2f} s")
                    logging.info(f"[{label}] Best energies: {best_energies}")
            
            title = f"Energy History (N={N}, {len(schedules)} schedules)"
            plot_energy_histories(
                all_histories_dict,
                title=title,
                out_path=output_path,
                schedule_labels=list(all_histories_dict.keys())
            )
        else:
            beta_schedule, base_seed, sched_desc, schedule_params = build_schedule_from_common(
                common, n_steps
            )

            logging.info(
                f"\nRunning {n_runs} runs on N={N} with {sched_desc}, "
                f"init_mode={init_mode}, base_seed={base_seed}"
            )

            all_histories, best_energies, run_times, _, _ , steps_to_best= run_experiment(
                N=N,
                n_steps=n_steps,
                init_mode=init_mode,
                beta_schedule=beta_schedule,
                n_runs=n_runs,
                base_seed=base_seed,
                verbose=verbose,
                schedule_params=schedule_params,
                mcmc_type=mcmc_type,
            )

            mean_time = np.mean(run_times)
            logging.info(f"\n[Single_N] Mean time per run: {mean_time:.2f} s")
            logging.info(f"[Single_N] Best energies: {best_energies}")

            title = f"Energy History (N={N}, {sched_desc})"
            plot_energy_histories(all_histories, title=title, out_path=output_path)

    elif experiment_type == "measure_min_energy_vs_N":
        params = config["measure_min_energy_vs_N"]
        Ns = params["Ns"]

        n_steps_exp = n_steps
        output_path = common_output_path

        beta_schedule, base_seed, sched_desc, schedule_params = build_schedule_from_common(
            common, n_steps_exp
        )

        if "init_modes" in params:
            init_modes = params["init_modes"]
            if isinstance(init_modes, str):
                init_modes = [init_modes]
        else:
            init_modes = [init_mode]

        logging.info(
            "\nMeasuring minimal energy as a function of N with "
            f"{sched_desc}, init_modes={init_modes}, base_seed={base_seed}"
        )

        result_dict = measure_min_energy_vs_N(
            Ns=Ns,
            n_steps=n_steps_exp,
            beta_schedule=beta_schedule,
            schedule_params=schedule_params,
            init_modes=init_modes,
            n_runs=n_runs,
            base_seed=base_seed,
            verbose=verbose,
            plot=True,
            out_path=output_path,
            mcmc_type=mcmc_type,
        )

        logging.info("\nResults:")
        for init_mode in init_modes:
            logging.info(f"\n{init_mode}:")
            means = result_dict["results"][init_mode]["mean_min_energies"]
            stds = result_dict["results"][init_mode]["std_min_energies"]
            for N, m, s in zip(result_dict["Ns"], means, stds):
                logging.info(f"  N={N}: {m:.2f} ± {s:.2f}")

    elif experiment_type == "beta_start_end_pairs":
        params = config["beta_start_end_pairs"]
        N = params["N"]
        beta_start_ends = params["beta_start_ends"]
        annealing_type = params.get("annealing_type", "linear_annealing")
        output_path = params.get("output_path", common_output_path)
        output_path_acceptance = params.get("output_path_acceptance", None)
        
        base_seed = common["betta_scheduling"].get("base_seed", 0)
        
        logging.info(
            f"\nRunning experiments for {len(beta_start_ends)} beta_start/beta_end pairs"
            f" with {annealing_type} annealing"
        )
        logging.info(f"N={N}, n_runs={n_runs}, init_mode={init_mode}, base_seed={base_seed}")
        logging.info(f"Beta pairs: {beta_start_ends}")
        
        result_dict = run_beta_start_end_pairs(
            N=N,
            n_steps=n_steps,
            beta_start_ends=beta_start_ends,
            annealing_type=annealing_type,
            init_mode=init_mode,
            n_runs=n_runs,
            base_seed=base_seed,
            verbose=verbose,
            plot=True,
            out_path=output_path,
            out_path_acceptance=output_path_acceptance,
            mcmc_type=mcmc_type,
        )
        
        logging.info("\nResults summary:")
        for label, best_energies in result_dict["all_best_energies"].items():
            mean_best = np.mean(best_energies)
            std_best = np.std(best_energies)
            logging.info(f"  {label}: {mean_best:.2f} ± {std_best:.2f}")

    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")


