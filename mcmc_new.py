import os
import numpy as np
import matplotlib.pyplot as plt
import time   

# ------------------------------
#  Geometry / attack checks
# ------------------------------

def is_attacking(i1, j1, k1, i2, j2, k2):
    # Same (i,j) position — impossible in your representation, but safe to keep
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
    def __init__(self, N, state=None):
        """
        State: N x N array.
        For each (i, j), state[i, j] = k in {0, ..., N-1},
        meaning there is exactly one queen at (i, j, k).
        """
        self.N = N
        if state is None:
            self.state = np.random.randint(0, N, size=(N, N))
        else:
            self.state = state

        # Precompute coordinate grids for vectorization
        self.I_grid, self.J_grid = np.indices((N, N))

        self._energy = None

    def copy(self):
        return State3DQueens(self.N, state=self.state.copy())

    def energy(self, recompute=False):
        if self._energy is None or recompute:
            self._energy = self._compute_energy()
        return self._energy

    def _compute_energy(self):
        """
        Full O(N^4) energy computation, only used for initialization / debugging.
        """
        N = self.N
        positions = [(i, j, self.state[i, j]) for i in range(N) for j in range(N)]
        Q = len(positions)

        if Q < 2:
            return 0

        count = 0
        for q1 in range(Q):
            i1, j1, k1 = positions[q1]
            for q2 in range(q1 + 1, Q):
                i2, j2, k2 = positions[q2]
                if is_attacking(i1, j1, k1, i2, j2, k2):
                    count += 1

        return count

    def propose_move(self, i, j, new_k):
        """
        Apply move (i, j) -> new_k, return old_k.
        """
        old_k = self.state[i, j]
        self.state[i, j] = new_k
        return old_k

    def revert_move(self, i, j, old_k):
        """
        Revert move at (i, j) to old_k.
        """
        self.state[i, j] = old_k

    def conflicts_for_queen(self, i, j, k):
        """
        Vectorized count of how many other queens are attacked by a queen
        at (i, j, k). Complexity: O(N^2) in NumPy.
        """
        k_grid = self.state
        I = self.I_grid
        J = self.J_grid

        # Exclude the queen itself
        not_self = (I != i) | (J != j)

        # Same rows/lines
        same_ik = (I == i) & (k_grid == k)
        same_jk = (J == j) & (k_grid == k)

        # 2D diagonals
        plane_k_diag = (k_grid == k) & (np.abs(I - i) == np.abs(J - j))
        plane_j_diag = (J == j) & (np.abs(I - i) == np.abs(k_grid - k))
        plane_i_diag = (I == i) & (np.abs(J - j) == np.abs(k_grid - k))

        # 3D space diagonal
        di = np.abs(I - i)
        dj = np.abs(J - j)
        dk = np.abs(k_grid - k)
        space_diag = (di == dj) & (dj == dk)

        attacked = (same_ik | same_jk |
                    plane_k_diag | plane_j_diag | plane_i_diag |
                    space_diag)

        attacked &= not_self

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

def metropolis_mcmc(N, n_steps, beta_schedule, verbose=True, seed=None):
    """
    Metropolis / SA sampler with local, vectorized ΔE update.
    """
    if seed is not None:
        np.random.seed(seed)

    state = State3DQueens(N)
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

        # Pick random (i, j)
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        old_k = state.state[i, j]

        # Compute conflicts at old position
        old_conflicts = state.conflicts_for_queen(i, j, old_k)

        # Propose new k != old_k
        new_k = np.random.randint(0, N - 1)
        if new_k >= old_k:
            new_k += 1

        # Apply move
        state.state[i, j] = new_k

        # Conflicts at new position
        new_conflicts = state.conflicts_for_queen(i, j, new_k)

        # Each conflict corresponds to one pair,
        # so the global energy changes by new_conflicts - old_conflicts
        delta_E = new_conflicts - old_conflicts
        proposed_energy = current_energy + delta_E

        accept_prob = min(1.0, np.exp(-beta_t * delta_E))

        if np.random.random() < accept_prob:
            # Accept
            current_energy = proposed_energy
            state._energy = current_energy
            accepted += 1

            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy
        else:
            # Reject: revert
            state.state[i, j] = old_k
            state._energy = current_energy

        energy_history.append(current_energy)

        if verbose and (step + 1) % next_report == 0:
            frac = (step + 1) / n_steps
            print(
                f"[chain] {step + 1}/{n_steps} steps "
                f"({frac*100:.1f}%) | E={current_energy}, best={best_energy}, "
                f"beta_t={beta_t:.3f}"
            )

    if verbose:
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
            verbose=verbose,   # step-level progress comes from here
        )
        end_time = time.time()

        duration = end_time - start_time
        run_times.append(duration)

        all_histories.append(res["energy_history"])
        best_energies.append(res["best_energy"])

        if verbose:
            frac_runs = (r + 1) / n_runs
            print(f"=== Completed run {r+1}/{n_runs} "
                  f"({frac_runs*100:.1f}% of experiment) | "
                  f"time this run: {duration:.2f} s ===")

    # Print mean time if verbose
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
    n_steps = 100000
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
        verbose=True,   # 👈 get progress + timing
    )

    mean_time_const = np.mean(times_const)
    print(f"\n[Metropolis] Mean time per run: {mean_time_const:.2f} s")

    plot_energy_histories(
        all_hist_const,
        title=f"Energy History (Metropolis, N={N}, beta={beta_const})",
        out_path="figures/energy_history_metropolis.png",
    )

    # Example 2: simulated annealing with linear schedule
    beta_start = 0.01   # high temperature (weak penalty on uphill moves)
    beta_end = 10.0     # low temperature
    beta_schedule_sa = linear_annealing_beta(beta_start, beta_end, n_steps)

    print(f"\nRunning {n_runs} runs with simulated annealing "
          f"(beta from {beta_start} to {beta_end})")

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
        title=f"Energy History (Simulated Annealing, N={N}, "
              f"beta: {beta_start}→{beta_end})",
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
