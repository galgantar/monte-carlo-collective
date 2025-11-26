import os
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml

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


def metropolis_mcmc(N, n_steps, init_mode, beta_schedule, verbose=True, seed=None, Q=None):
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
                f"[chain] {step + 1}/{n_steps} steps "
                f"({frac*100:.1f}%) | E={current_energy}, best={best_energy}, "
                f"beta_t={beta_t:.3f}"
            )

    if verbose and n_steps > 0:
        print(f"[chain] Final energy: {current_energy}")
        print(f"[chain] Best energy:   {best_energy}")
        print(f"[chain] Acceptance rate: {accepted / n_steps:.3f}")

    return {
        "final_state": state,
        "final_energy": current_energy,
        "best_state": best_state,
        "best_energy": best_energy,
        "energy_history": energy_history,
    }


def run_single_chain(N, n_steps, init_mode,beta_schedule, seed=None, verbose=False):
    return metropolis_mcmc(
        N=N,
        n_steps=n_steps,
        init_mode = init_mode,
        beta_schedule=beta_schedule,
        verbose=verbose,
        seed=seed,
        init_mode=init_mode,
    )


def run_experiment(N, n_steps, init_mode, beta_schedule, n_runs, base_seed=0, verbose=False):
    all_histories = []
    best_energies = []
    run_times = []

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
            init_mode=init_mode,
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


def plot_energy_histories(all_histories, title, out_path=None):
    n_runs = len(all_histories)
    n_steps_plus1 = len(all_histories[0])

    energies = np.array(all_histories)
    mean_energy = energies.mean(axis=0)
    std_energy = energies.std(axis=0)

    plt.figure(figsize=(10, 6))

    for r in range(n_runs):
        plt.plot(energies[r], alpha=0.3, linewidth=1)

    steps = np.arange(n_steps_plus1)
    plt.plot(mean_energy, linewidth=2.5, label="Mean energy")
    plt.fill_between(
        steps,
        mean_energy - std_energy,
        mean_energy + std_energy,
        alpha=0.2,
        label="±1 std",
    )

    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def measure_min_energy_vs_N(
    Ns,
    n_steps,
    beta_schedule,
    n_runs=5,
    base_seed=100,
    verbose=True,
    plot=True,
    out_path=None,
    init_mode="latin",
):
    mean_min_energies = []
    std_min_energies = []
    all_min_energies = []

    for idx, N in enumerate(Ns):
        if verbose:
            print(f"\n=== Running N = {N} ===")

        _, best_energies, _ = run_experiment(
            N=N,
            n_steps=n_steps,
            beta_schedule=beta_schedule,
            n_runs=n_runs,
            base_seed=base_seed + 10 * idx,
            verbose=verbose,
            init_mode=init_mode,
        )

        best_energies = np.array(best_energies)
        all_min_energies.append(best_energies)

        mean_min_energies.append(best_energies.mean())
        std_min_energies.append(best_energies.std())

        if verbose:
            print(f"  → Mean min energy = {mean_min_energies[-1]:.2f} ± {std_min_energies[-1]:.2f}")

    mean_min_energies = np.array(mean_min_energies)
    std_min_energies = np.array(std_min_energies)

    if plot:
        plt.figure(figsize=(8, 6))
        Ns_arr = np.array(Ns)

        plt.errorbar(
            Ns_arr,
            mean_min_energies,
            yerr=std_min_energies,
            fmt="o-",
            capsize=4,
            linewidth=2,
            markersize=6,
            label="Mean minimal energy (±1 std)"
        )

        plt.xlabel("Board size N")
        plt.ylabel("Minimal energy reached")
        plt.title("MCMC: Minimal Energy vs. Board Size N")
        plt.grid(True)
        plt.legend()

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return (
        Ns,
        mean_min_energies,
        std_min_energies,
        all_min_energies,
    )



if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    experiment_type = config["experiment_type"]
    common = config["common"]
    n_steps = common["n_steps"]
    n_runs = common["n_runs"]
    verbose = common["verbose"]
    init_mode = common["initialization"]

    print(init_mode)

    if experiment_type == "constant_beta":
        params = config["constant_beta"]
        N = params["N"]
        beta_const = params["beta_const"]
        base_seed = params["base_seed"]
        output_path = params["output_path"]
        
        beta_schedule_const = constant_beta(beta_const)

        print(f"Running {n_runs} runs with constant beta = {beta_const}")
        all_hist_const, best_const, times_const = run_experiment(
            N=N,
            n_steps=n_steps,
            beta_schedule=beta_schedule_const,
            n_runs=n_runs,
            base_seed=base_seed,
            init_mode=init_mode,
            verbose=verbose,
        )

        mean_time_const = np.mean(times_const)
        print(f"\n[Metropolis] Mean time per run: {mean_time_const:.2f} s")

        plot_energy_histories(
            all_hist_const,
            title=f"Energy History (Metropolis, N={N}, beta={beta_const})",
            out_path=output_path,
        )

    elif experiment_type == "linear_annealing":
        params = config["linear_annealing"]
        N = params["N"]
        beta_start = params["beta_start"]
        beta_end = params["beta_end"]
        base_seed = params["base_seed"]
        output_path = params["output_path"]
        
        beta_schedule_linear = linear_annealing_beta(beta_start, beta_end, n_steps)

        print(
            f"\nRunning {n_runs} runs with linear annealing "
            f"(beta from {beta_start} to {beta_end})"
        )

        all_hist_la, best_la, times_la = run_experiment(
            N=N,
            n_steps=n_steps,
            init_mode=init_mode,
            beta_schedule=beta_schedule_linear,
            n_runs=n_runs,
            base_seed=base_seed,
            init_mode=init_mode,
            verbose=verbose,
        )

        mean_time_la = np.mean(times_la)
        print(f"\n[Linear Annealing] Mean time per run: {mean_time_la:.2f} s")

        plot_energy_histories(
            all_hist_la,
            title=(
                f"Energy History (Linear Annealing, N={N}, "
                f"beta: {beta_start}→{beta_end})"
            ),
            out_path=output_path,
        )

        print("\nBest energies (Linear Annealing):", best_la)

    elif experiment_type == "exponential_annealing":
        params = config["exponential_annealing"]
        N = params["N"]
        beta_start = params["beta_start"]
        beta_end = params["beta_end"]
        base_seed = params["base_seed"]
        output_path = params["output_path"]
        
        beta_schedule_ea = exponential_annealing_beta(beta_start, beta_end, n_steps)

        print(
            f"\nRunning {n_runs} runs with exponential annealing "
            f"(beta from {beta_start} to {beta_end})"
        )

        all_hist_ea, best_ea, times_ea = run_experiment(
            N=N,
            n_steps=n_steps,
            beta_schedule=beta_schedule_ea,
            n_runs=n_runs,
            base_seed=base_seed,
            init_mode=init_mode,
            verbose=verbose,
        )

        mean_time_ea = np.mean(times_ea)
        print(f"\n[Exponential Annealing] Mean time per run: {mean_time_ea:.2f} s")

        plot_energy_histories(
            all_hist_ea,
            title=(
                f"Energy History (Exponential Annealing, N={N}, "
                f"beta: {beta_start}→{beta_end})"
            ),
            out_path=output_path,
        )

    elif experiment_type == "measure_min_energy_vs_N":
        params = config["measure_min_energy_vs_N"]
        Ns = params["Ns"]
        n_steps_exp = params.get("n_steps", n_steps)
        beta_schedule_type = params["beta_schedule_type"]
        base_seed = params["base_seed"]
        output_path = params["output_path"]
        
        if beta_schedule_type == "constant_beta":
            beta_const = params["beta_const"]
            beta_schedule = constant_beta(beta_const)
        elif beta_schedule_type == "linear_annealing":
            beta_start = params["beta_start"]
            beta_end = params["beta_end"]
            beta_schedule = linear_annealing_beta(beta_start, beta_end, n_steps_exp)
        elif beta_schedule_type == "exponential_annealing":
            beta_start = params["beta_start"]
            beta_end = params["beta_end"]
            beta_schedule = exponential_annealing_beta(beta_start, beta_end, n_steps_exp)
        else:
            raise ValueError(f"Unknown beta_schedule_type: {beta_schedule_type}")

        print("\nMeasuring minimal energy as a function of N...")
        Ns_out, means, stds, all_data = measure_min_energy_vs_N(
            Ns=Ns,
            n_steps=n_steps_exp,
            beta_schedule=beta_schedule,
            n_runs=n_runs,
            base_seed=base_seed,
            verbose=verbose,
            plot=True,
            out_path=output_path,
            init_mode=init_mode,
        )

        print("\nResults:")
        for N, m, s in zip(Ns_out, means, stds):
            print(f"N={N}: {m:.2f} ± {s:.2f}")
