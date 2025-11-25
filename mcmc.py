import os
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml

import math

class State3DQueens:
    def __init__(self, N, Q=None, positions=None, init_mode="latin"):
        """
        init_mode:
            - "latin": k = (i + j) mod N (deterministic Latin-square style)
            - "klarner": k = (3i + 5j) mod N (requires gcd(N, 210) == 1 for non-attacking theorem)
            - "random": fully random distinct 3D positions
        """
        self.N = N
        if Q is None:
            Q = N * N
        self.Q = Q

        if positions is None:
            if init_mode in ("latin", "klarner"):
                if Q != N * N:
                    raise ValueError(
                        f"{init_mode} initialization assumes Q = N^2, "
                        f"got Q={Q}, N^2={N*N}."
                    )

                # Create full (i,j) grid
                i_grid, j_grid = np.indices((N, N))  # shape (N,N)

                if init_mode == "latin":
                    # Latin square: k = (i + j) mod N
                    k_grid = (i_grid + j_grid) % N
                else:  # "klarner"
                    # Klarner construction: k = (3i + 5j) mod N
                    # For gcd(N, 210) == 1 this is a known non-attacking configuration
                    if math.gcd(N, 210) != 1:
                        print(
                            f"[warning] Klarner construction has no guarantee for N={N} "
                            f"(gcd(N, 210) = {math.gcd(N,210)}). Using it anyway."
                        )
                    k_grid = (3 * i_grid + 5 * j_grid) % N

                # Flatten into (Q, 3)
                i_coords = i_grid.ravel()
                j_coords = j_grid.ravel()
                k_coords = k_grid.ravel()
                self.queens = np.stack([i_coords, j_coords, k_coords], axis=1)

            elif init_mode == "random":
                total_cells = N ** 3
                if Q > total_cells:
                    raise ValueError("Q cannot exceed N^3.")

                flat_indices = np.random.choice(total_cells, size=Q, replace=False)
                k_coords = flat_indices % N
                j_coords = (flat_indices // N) % N
                i_coords = (flat_indices // (N * N))
                self.queens = np.stack([i_coords, j_coords, k_coords], axis=1)

            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")
        else:
            positions = np.asarray(positions, dtype=int)
            if positions.shape[1] != 3:
                raise ValueError("positions must be of shape (Q, 3).")
            self.queens = positions
            self.Q = positions.shape[0]

        # Occupancy set
        self.occ_set = set()
        for (i, j, k) in self.queens:
            pos = (int(i), int(j), int(k))
            if pos in self.occ_set:
                raise ValueError("Two queens occupy the same (i,j,k) cell.")
            self.occ_set.add(pos)

        self._energy = None


    def copy(self):
        new_state = State3DQueens(self.N, Q=self.Q, positions=self.queens.copy())
        new_state._energy = self._energy
        return new_state


    def energy(self, recompute=False):
        if self._energy is None or recompute:
            self._energy = self._compute_energy()
        return self._energy

    def _compute_energy(self):
        Q = self.Q
        if Q < 2:
            return 0

        positions = self.queens
        # Note: i, j, k have shape (Q,)
        i = positions[:, 0]
        j = positions[:, 1]
        k = positions[:, 2]

        # Note: di, dj, dk have shape (Q, Q)
        di = np.abs(i[:, None] - i[None, :])
        dj = np.abs(j[:, None] - j[None, :])
        dk = np.abs(k[:, None] - k[None, :])

        # Note: same_ij, same_ik, same_jk have shape (Q, Q)
        same_ij = (i[:, None] == i[None, :]) & (j[:, None] == j[None, :])
        same_ik = (i[:, None] == i[None, :]) & (k[:, None] == k[None, :])
        same_jk = (j[:, None] == j[None, :]) & (k[:, None] == k[None, :])
        
        # Note: plane_k_diag, plane_j_diag, plane_i_diag have shape (Q, Q)
        plane_k_diag = (k[:, None] == k[None, :]) & (di == dj)
        plane_j_diag = (j[:, None] == j[None, :]) & (di == dk)
        plane_i_diag = (i[:, None] == i[None, :]) & (dj == dk)
        
        space_diag = (di == dj) & (dj == dk)

        attacked = (
            same_ij
            | same_ik
            | same_jk
            | plane_k_diag
            | plane_j_diag
            | plane_i_diag
            | space_diag
        )

        upper_triangle = np.triu(attacked, k=1)
        return int(upper_triangle.sum())

    def propose_move(self, q_idx, new_pos):
        i_old, j_old, k_old = self.queens[q_idx]
        i_new, j_new, k_new = new_pos

        old_pos = (int(i_old), int(j_old), int(k_old))
        new_pos_t = (int(i_new), int(j_new), int(k_new))

        self.occ_set.remove(old_pos)
        self.occ_set.add(new_pos_t)

        self.queens[q_idx] = [i_new, j_new, k_new]

        return old_pos

    def revert_move(self, q_idx, old_pos):
        i_current, j_current, k_current = self.queens[q_idx]
        current_pos = (int(i_current), int(j_current), int(k_current))

        self.occ_set.remove(current_pos)
        self.occ_set.add(old_pos)

        i_old, j_old, k_old = old_pos
        self.queens[q_idx] = [i_old, j_old, k_old]

    def conflicts_for_queen(self, q_idx, pos=None):
        if pos is None:
            i, j, k = self.queens[q_idx]
        else:
            i, j, k = pos

        mask = np.ones(self.Q, dtype=bool)
        mask[q_idx] = False
        others = self.queens[mask]
        if others.shape[0] == 0:
            return 0

        i2 = others[:, 0]
        j2 = others[:, 1]
        k2 = others[:, 2]

        di = np.abs(i2 - i)
        dj = np.abs(j2 - j)
        dk = np.abs(k2 - k)

        same_ij = (i2 == i) & (j2 == j)

        same_ik = (i2 == i) & (k2 == k)
        same_jk = (j2 == j) & (k2 == k)

        plane_k_diag = (k2 == k) & (di == dj)
        plane_j_diag = (j2 == j) & (di == dk)
        plane_i_diag = (i2 == i) & (dj == dk)

        space_diag = (di == dj) & (dj == dk)

        attacked = (
            same_ij
            | same_ik
            | same_jk
            | plane_k_diag
            | plane_j_diag
            | plane_i_diag
            | space_diag
        )

        return int(attacked.sum())
    

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


def metropolis_mcmc(N, n_steps, beta_schedule, verbose=True, seed=None, Q=None):
    if seed is not None:
        np.random.seed(seed)

    state = State3DQueens(N, Q=Q, init_mode = "latin")
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


def run_single_chain(N, n_steps, beta_schedule, seed=None, verbose=False):
    return metropolis_mcmc(
        N=N,
        n_steps=n_steps,
        beta_schedule=beta_schedule,
        verbose=verbose,
        seed=seed,
    )


def run_experiment(N, n_steps, beta_schedule, n_runs, base_seed=0, verbose=False):
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
            beta_schedule=beta_schedule,
            seed=base_seed + r,
            verbose=verbose,
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
            beta_schedule=beta_schedule_linear,
            n_runs=n_runs,
            base_seed=base_seed,
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
        )

        print("\nResults:")
        for N, m, s in zip(Ns_out, means, stds):
            print(f"N={N}: {m:.2f} ± {s:.2f}")
