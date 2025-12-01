import os
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

from mcmc import State3DQueens


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

        if np.random.random() < accept_prob:
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
            print(
                f"{prefix} {step + 1}/{n_steps} steps "
                f"({frac*100:.1f}%) | E={current_energy}, best={best_energy}, "
                f"beta_t={beta_t:.3f}"
            )

    if verbose and n_steps > 0:
        print(f"{prefix} Final energy: {current_energy}")
        print(f"{prefix} Best energy:   {best_energy}")
        print(f"{prefix} Acceptance rate: {accepted / n_steps:.3f}")

    return {
        "final_state": state,
        "final_energy": current_energy,
        "best_state": best_state,
        "best_energy": best_energy,
        "energy_history": energy_history,
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


def run_single_chain_multithread(args):
    """Wrapper function for parallel execution of a single chain."""
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
    }


def run_experiment(N, n_steps, init_mode, beta_schedule, n_runs, base_seed=0, verbose=False, n_workers=None, schedule_params=None):
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
    """
    all_histories = []
    best_energies = []
    run_times = []
    
    if n_runs > 1:
        if schedule_params is None:
            raise ValueError(f"schedule_params is required for parallel execution when n_runs > 1")
        run_args = [
            (N, n_steps, init_mode, schedule_params, base_seed + r, verbose, r)
            for r in range(n_runs)
        ]
        
        if verbose:
            print(f"\n>>> Running {n_runs} chains in parallel...")
        
        start_total = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_run = {
                executor.submit(run_single_chain_multithread, args): r
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
                        print(
                            f"=== Completed run {completed}/{n_runs} "
                            f"({completed*100/n_runs:.1f}% of experiment) | "
                            f"time this run: {result['duration']:.2f} s ==="
                        )
                except Exception as exc:
                    print(f"Run {run_idx} generated an exception: {exc}")
                    raise
        
        end_total = time.time()
        total_time = end_total - start_total
        
        results.sort(key=lambda x: x["run_idx"])
        
        for result in results:
            all_histories.append(result["energy_history"])
            best_energies.append(result["best_energy"])
            run_times.append(result["duration"])
        
        if verbose and n_runs > 0:
            mean_time = np.mean(run_times)
            print(f"\n>>> Mean time per run: {mean_time:.2f} s")
            print(f">>> Total time for {n_runs} runs: {total_time:.2f} s")
    else:
        for r in range(n_runs):
            if verbose:
                print(f"\n=== Run {r+1}/{n_runs} ===")

            start_time = time.time()
            res = run_single_chain(
                N=N,
                n_steps=n_steps,
                init_mode = init_mode, 
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

            if verbose:
                frac_runs = (r + 1) / n_runs
                print(
                    f"=== Completed run {r+1}/{n_runs} "
                    f"({frac_runs*100:.1f}% of experiment) | "
                    f"time this run: {duration:.2f} s ==="
                )

        if verbose and n_runs > 0:
            mean_time = np.mean(run_times)
            total_time = np.sum(run_times)
            print(f"\n>>> Mean time per run: {mean_time:.2f} s")
            print(f">>> Total time for {n_runs} runs: {total_time:.2f} s")

    return all_histories, best_energies, run_times


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
        
        steps = np.arange(n_steps_plus1)
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

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10, framealpha=0.9, loc='best')
    
    plt.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

def measure_min_energy_vs_N(
    Ns,
    n_steps,
    beta_schedule,
    schedule_params=None,
    init_modes = ["random"], 
    n_runs=5,
    base_seed=100,
    verbose=True,
    plot=True,
    out_path=None,
):
    if isinstance(init_modes, str):
        init_modes = [init_modes]
    
    results = {}
    
    for init_mode in init_modes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running experiments with init_mode = '{init_mode}'")
            print(f"{'='*60}")
        
        mean_min_energies = []
        std_min_energies = []
        all_min_energies = []

        for idx, N in enumerate(Ns):
            if verbose:
                print(f"\n=== Running N = {N} (init_mode={init_mode}) ===")

            init_mode_offset = sum(ord(c) for c in init_mode) % 1000
            _, best_energies, _ = run_experiment(
                N=N,
                n_steps=n_steps,
                init_mode=init_mode,
                beta_schedule=beta_schedule,
                n_runs=n_runs,
                base_seed=base_seed + 10 * idx + init_mode_offset,
                verbose=verbose,
                schedule_params=schedule_params,
            )

            best_energies = np.array(best_energies)
            all_min_energies.append(best_energies)

            mean_min_energies.append(best_energies.mean())
            std_min_energies.append(best_energies.std())

            if verbose:
                print(f"  → Mean min energy = {mean_min_energies[-1]:.2f} ± {std_min_energies[-1]:.2f}")

        results[init_mode] = {
            "mean_min_energies": np.array(mean_min_energies),
            "std_min_energies": np.array(std_min_energies),
            "all_min_energies": all_min_energies,
        }

    if plot:
        plt.figure(figsize=(10, 6))
        Ns_arr = np.array(Ns)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(init_modes)))
        
        for idx, init_mode in enumerate(init_modes):
            mean_energies = results[init_mode]["mean_min_energies"]
            std_energies = results[init_mode]["std_min_energies"]
            color = colors[idx]
            
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

        plt.xlabel("Board size N")
        plt.ylabel("Minimal energy reached")
        plt.title("MCMC: Minimal Energy vs. Board Size N")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches="tight")
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

    experiment_type = config["experiment_type"]
    common = config["common"]
    n_steps = common["n_steps"]
    n_runs = common["n_runs"]
    verbose = common["verbose"]
    init_mode = common["initialization"]
    common_output_path = common["output_path"]

    print(f"Initialization mode: {init_mode}")
    print(f"Experiment type: {experiment_type}")

    if experiment_type == "single_N":
        single_cfg = config["single_N"]
        N = single_cfg["N"]

        output_path = common_output_path

        sched_cfg = common["betta_scheduling"]
        sched_type = sched_cfg["type"]
        
        if isinstance(sched_type, list):
            schedules = build_schedules_from_types(sched_type, sched_cfg, n_steps)
            
            print(f"\nRunning {n_runs} runs on N={N} with {len(schedules)} schedule(s), init_mode={init_mode}")
            print(f"All schedules use: beta_start={sched_cfg['beta_start']}, "
                  f"beta_end={sched_cfg['beta_end']}, base_seed={sched_cfg['base_seed']}")
            
            all_histories_dict = {}
            all_best_energies_dict = {}
            all_run_times_dict = {}
            
            for beta_schedule, base_seed, sched_desc, label, schedule_params in schedules:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Schedule: {label} ({sched_desc})")
                    print(f"{'='*60}")
                
                all_histories, best_energies, run_times = run_experiment(
                    N=N,
                    n_steps=n_steps,
                    init_mode=init_mode,
                    beta_schedule=beta_schedule,
                    n_runs=n_runs,
                    base_seed=base_seed,
                    verbose=verbose,
                    schedule_params=schedule_params,
                )
                
                all_histories_dict[label] = all_histories
                all_best_energies_dict[label] = best_energies
                all_run_times_dict[label] = run_times
                
                mean_time = np.mean(run_times)
                if verbose:
                    print(f"\n[{label}] Mean time per run: {mean_time:.2f} s")
                    print(f"[{label}] Best energies: {best_energies}")
            
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

            print(
                f"\nRunning {n_runs} runs on N={N} with {sched_desc}, "
                f"init_mode={init_mode}, base_seed={base_seed}"
            )

            all_histories, best_energies, run_times = run_experiment(
                N=N,
                n_steps=n_steps,
                init_mode=init_mode,
                beta_schedule=beta_schedule,
                n_runs=n_runs,
                base_seed=base_seed,
                verbose=verbose,
                schedule_params=schedule_params,
            )

            mean_time = np.mean(run_times)
            print(f"\n[Single_N] Mean time per run: {mean_time:.2f} s")
            print("[Single_N] Best energies:", best_energies)

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

        print(
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
        )

        print("\nResults:")
        for init_mode in init_modes:
            print(f"\n{init_mode}:")
            means = result_dict["results"][init_mode]["mean_min_energies"]
            stds = result_dict["results"][init_mode]["std_min_energies"]
            for N, m, s in zip(result_dict["Ns"], means, stds):
                print(f"  N={N}: {m:.2f} ± {s:.2f}")

    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")


