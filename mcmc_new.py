import os
import time
import numpy as np
import matplotlib.pyplot as plt
import time   

# ------------------------------
#  Geometry / attack checks
# ------------------------------

def is_attacking(i1, j1, k1, i2, j2, k2):
    # Same (i,j) vertical line (along k)
    if i1 == i2 and j1 == j2:
        return True

    # Same "row" in (i,j,k) axes
    if i1 == i2 and k1 == k2:
        return True

    if j1 == j2 and k1 == k2:
        return True

    # Same horizontal slice: (i,j) diagonal in plane k = const
    if k1 == k2 and abs(i1 - i2) == abs(j1 - j2):
        return True

    # Same vertical slice: (i,k) diagonal in plane j = const
    if j1 == j2 and abs(i1 - i2) == abs(k1 - k2):
        return True

    # Same vertical slice: (j,k) diagonal in plane i = const
    if i1 == i2 and abs(j1 - j2) == abs(k1 - k2):
        return True

    # Full 3D diagonal
    if abs(i1 - i2) == abs(j1 - j2) == abs(k1 - k2):
        return True

    return False


# ------------------------------
#  State representation
# ------------------------------

class State3DQueens:
    """
    General 3D queens configuration:

    - Board is N x N x N.
    - We keep Q queens (by default Q = N^2).
    - Queens can be anywhere in the 3D grid, with at most one queen per (i,j,k).
    - Multiple queens can share the same (i,j) with different k.
    - We store only:
        * self.queens: (Q, 3) array of coordinates
        * self.occ_set: set of occupied (i,j,k)
      No full N^3 grid.
    """

    def __init__(self, N, Q=None, positions=None):
        self.N = N
        if Q is None:
            Q = N * N  # same number of queens as before
        self.Q = Q

        if positions is None:
            # Sample Q distinct cells from the N^3 possible positions
            total_cells = N ** 3
            if Q > total_cells:
                raise ValueError("Q cannot exceed N^3.")

            flat_indices = np.random.choice(total_cells, size=Q, replace=False)
            k_coords = flat_indices % N
            j_coords = (flat_indices // N) % N
            i_coords = flat_indices // (N * N)
            self.queens = np.stack([i_coords, j_coords, k_coords], axis=1)
        else:
            positions = np.asarray(positions, dtype=int)
            if positions.shape[1] != 3:
                raise ValueError("positions must be of shape (Q, 3).")
            self.queens = positions
            self.Q = positions.shape[0]

        # Occupancy set: track which (i,j,k) are used
        self.occ_set = set()
        for (i, j, k) in self.queens:
            pos = (int(i), int(j), int(k))
            if pos in self.occ_set:
                raise ValueError("Two queens occupy the same (i,j,k) cell.")
            self.occ_set.add(pos)

        self._energy = None

    def copy(self):
        new_state = State3DQueens(self.N, Q=self.Q, positions=self.queens.copy())
        # Copy cached energy as well
        new_state._energy = self._energy
        return new_state

    # --------- Energy computation ---------

    def energy(self, recompute=False):
        if self._energy is None or recompute:
            self._energy = self._compute_energy()
        return self._energy

    def _compute_energy(self):
        """
        Full O(Q^2) energy computation by pairwise checking.
        """
        Q = self.Q
        if Q < 2:
            return 0

        positions = self.queens
        count = 0
        for q1 in range(Q):
            i1, j1, k1 = positions[q1]
            for q2 in range(q1 + 1, Q):
                i2, j2, k2 = positions[q2]
                if is_attacking(i1, j1, k1, i2, j2, k2):
                    count += 1
        return count

    # --------- Local move operations ---------

    def propose_move(self, q_idx, new_pos):
        """
        Move queen q_idx to new_pos = (i_new, j_new, k_new).
        Returns old_pos for possible revert (not used in current sampler).
        """
        i_old, j_old, k_old = self.queens[q_idx]
        i_new, j_new, k_new = new_pos

        old_pos = (int(i_old), int(j_old), int(k_old))
        new_pos_t = (int(i_new), int(j_new), int(k_new))

        # Update occupancy set
        self.occ_set.remove(old_pos)
        self.occ_set.add(new_pos_t)

        # Update queen position
        self.queens[q_idx] = [i_new, j_new, k_new]

        return old_pos

    def revert_move(self, q_idx, old_pos):
        """
        Revert queen q_idx to old_pos = (i_old, j_old, k_old).
        Not used in current sampler, but implemented for completeness.
        """
        i_current, j_current, k_current = self.queens[q_idx]
        current_pos = (int(i_current), int(j_current), int(k_current))

        # Remove current pos from occ_set, add old_pos
        self.occ_set.remove(current_pos)
        self.occ_set.add(old_pos)

        i_old, j_old, k_old = old_pos
        self.queens[q_idx] = [i_old, j_old, k_old]

    # --------- Local conflict counting ---------

    def conflicts_for_queen(self, q_idx, pos=None):
        """
        Count how many other queens are attacked by queen q_idx.

        If pos is None, use its current position.
        If pos is given (i,j,k), treat that as a hypothetical position
        WITHOUT modifying the state.
        """
        if pos is None:
            i, j, k = self.queens[q_idx]
        else:
            i, j, k = pos

        # All other queens
        mask = np.ones(self.Q, dtype=bool)
        mask[q_idx] = False
        others = self.queens[mask]  # shape (Q-1, 3)
        if others.shape[0] == 0:
            return 0

        i2 = others[:, 0]
        j2 = others[:, 1]
        k2 = others[:, 2]

        di = np.abs(i2 - i)
        dj = np.abs(j2 - j)
        dk = np.abs(k2 - k)

        # Same (i,j) vertical line (along k)
        same_ij = (i2 == i) & (j2 == j)

        # Same "row" in (i,j,k) axes
        same_ik = (i2 == i) & (k2 == k)
        same_jk = (j2 == j) & (k2 == k)

        # 2D diagonals in slices
        plane_k_diag = (k2 == k) & (di == dj)
        plane_j_diag = (j2 == j) & (di == dk)
        plane_i_diag = (i2 == i) & (dj == dk)

        # Full 3D diagonal
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
    

# ------------------------------
#  Beta schedules (Metropolis vs SA)
# ------------------------------

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


# ------------------------------
#  Core sampler using local ΔE
# ------------------------------

def metropolis_mcmc(N, n_steps, beta_schedule, verbose=True, seed=None, Q=None):
    """
    Metropolis / SA sampler with local ΔE updates on a general 3D state.

    - Board: N x N x N
    - Queens: Q (default N^2) distinct positions (i,j,k), no constraint per (i,j).
    """
    if seed is not None:
        np.random.seed(seed)

    state = State3DQueens(N, Q=Q)
    current_energy = state.energy(recompute=True)

    best_state = state.copy()
    best_energy = current_energy

    accepted = 0
    energy_history = [current_energy]

    # For progress tracking
    if verbose and n_steps > 0:
        next_report = max(1, n_steps // 10)  # report roughly every 10%

    for step in range(n_steps):
        beta_t = beta_schedule(step)

        # Pick a random queen index to move
        q_idx = np.random.randint(0, state.Q)

        # Conflicts at old position
        old_conflicts = state.conflicts_for_queen(q_idx)

        # Propose a random new empty (i,j,k)
        N_ = state.N
        while True:
            i_new = np.random.randint(0, N_)
            j_new = np.random.randint(0, N_)
            k_new = np.random.randint(0, N_)
            if (int(i_new), int(j_new), int(k_new)) not in state.occ_set:
                break

        # Compute conflicts at new position (hypothetical)
        new_conflicts = state.conflicts_for_queen(q_idx, pos=(i_new, j_new, k_new))

        # Each conflict corresponds to one pair involving this queen,
        # so the global energy changes by new_conflicts - old_conflicts.
        delta_E = new_conflicts - old_conflicts
        proposed_energy = current_energy + delta_E

        accept_prob = min(1.0, np.exp(-beta_t * delta_E))

        if np.random.random() < accept_prob:
            # Accept: actually move the queen
            state.propose_move(q_idx, (i_new, j_new, k_new))
            current_energy = proposed_energy
            state._energy = current_energy
            accepted += 1

            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy
        else:
            # Reject: do nothing (we never changed the state)
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


# ------------------------------
#  Experiment layer
# ------------------------------

def run_single_chain(N, n_steps, beta_schedule, seed=None, verbose=False):
    return metropolis_mcmc(
        N=N,
        n_steps=n_steps,
        beta_schedule=beta_schedule,
        verbose=verbose,
        seed=seed,
    )


def run_experiment(N, n_steps, beta_schedule, n_runs, base_seed=0, verbose=False):
    """
    Run multiple independent chains and gather stats.

    Returns
    -------
    all_histories : list of lists (energies per run)
    best_energies : list of best energies per run
    run_times     : list of wall-clock durations (seconds) per run
    """
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
    """
    Plot all individual runs + mean energy over time.
    """
    n_runs = len(all_histories)
    n_steps_plus1 = len(all_histories[0])  # all same length

    energies = np.array(all_histories)  # shape (n_runs, n_steps+1)
    mean_energy = energies.mean(axis=0)
    std_energy = energies.std(axis=0)

    plt.figure(figsize=(10, 6))

    # Light lines for each run
    for r in range(n_runs):
        plt.plot(energies[r], alpha=0.3, linewidth=1)

    # Mean with shading
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
    """
    Measure minimal energy reached by MCMC as a function of board size N.
    """

    mean_min_energies = []
    std_min_energies = []
    all_min_energies = []

    for idx, N in enumerate(Ns):
        if verbose:
            print(f"\n=== Running N = {N} ===")

        # Run the experiment for this N
        _, best_energies, _ = run_experiment(
            N=N,
            n_steps=n_steps,
            beta_schedule=beta_schedule,
            n_runs=n_runs,
            base_seed=base_seed + 10 * idx,
            verbose=verbose,
        )

        # best_energies is a list of minimal energies from each run
        best_energies = np.array(best_energies)
        all_min_energies.append(best_energies)

        mean_min_energies.append(best_energies.mean())
        std_min_energies.append(best_energies.std())

        if verbose:
            print(f"  → Mean min energy = {mean_min_energies[-1]:.2f} ± {std_min_energies[-1]:.2f}")

    mean_min_energies = np.array(mean_min_energies)
    std_min_energies = np.array(std_min_energies)

    # -------------------------------------------------------------
    # Optional plot
    # -------------------------------------------------------------
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


# ------------------------------
#  Main: choose schedule & run
# ------------------------------

if __name__ == "__main__":
    N = 5
    n_steps = 200000
    n_runs = 5

    # Example 1: plain Metropolis (constant beta)
    beta_const = 1.0
    beta_schedule_const = constant_beta(beta_const)

    print(f"Running {n_runs} runs with constant beta = {beta_const}")
    all_hist_const, best_const, times_const = run_experiment(
        N=N,
        n_steps=n_steps,
        beta_schedule=beta_schedule_const,
        n_runs=n_runs,
        base_seed=42,
        verbose=True,
    )

    mean_time_const = np.mean(times_const)
    print(f"\n[Metropolis] Mean time per run: {mean_time_const:.2f} s")

    plot_energy_histories(
        all_hist_const,
        title=f"Energy History (Metropolis, N={N}, beta={beta_const})",
        out_path="figures/energy_history_metropolis.png",
    )

    # Example 2: simulated annealing with linear schedule
    beta_start = 0.1   # high temperature (weak penalty on uphill moves)
    beta_end = 5.0     # low temperature
    beta_schedule_sa = linear_annealing_beta(beta_start, beta_end, n_steps)

    print(
        f"\nRunning {n_runs} runs with simulated annealing "
        f"(beta from {beta_start} to {beta_end})"
    )

    all_hist_sa, best_sa, times_sa = run_experiment(
        N=N,
        n_steps=n_steps,
        beta_schedule=beta_schedule_sa,
        n_runs=n_runs,
        base_seed=123,
        verbose=True,
    )

    mean_time_sa = np.mean(times_sa)
    print(f"\n[Simulated Annealing] Mean time per run: {mean_time_sa:.2f} s")

    plot_energy_histories(
        all_hist_sa,
        title=(
            f"Energy History (Simulated Annealing, N={N}, "
            f"beta: {beta_start}→{beta_end})"
        ),
        out_path="figures/energy_history_sa.png",
    )

    print("\nBest energies (Metropolis):", best_const)
    print("Best energies (Simulated Annealing):", best_sa)

    # Measuring minimal energy vs N
    Ns = [3, 4, 5, 6, 7, 8]

    print("\nMeasuring minimal energy as a function of N...")
    Ns_out, means, stds, all_data = measure_min_energy_vs_N(
        Ns=Ns,
        n_steps=100000,
        beta_schedule=beta_schedule_const,   # or beta_schedule_sa
        n_runs=5,
        base_seed=500,
        verbose=True,
        plot=True,
        out_path="figures/min_energy_vs_N.png",
    )

    print("\nResults:")
    for N, m, s in zip(Ns_out, means, stds):
        print(f"N={N}: {m:.2f} ± {s:.2f}")
