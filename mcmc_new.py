import os
import numpy as np
import matplotlib.pyplot as plt


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
        self._energy = None

    def copy(self):
        return State3DQueens(self.N, state=self.state.copy())

    def energy(self, recompute=False):
        if self._energy is None or recompute:
            self._energy = self._compute_energy()
        return self._energy

    def _compute_energy(self):
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
        Temporarily apply move (i, j) -> new_k, return old_k.
        """
        old_k = self.state[i, j]
        self.state[i, j] = new_k
        return old_k

    def revert_move(self, i, j, old_k):
        self.state[i, j] = old_k


# ------------------------------
#  Beta schedules (Metropolis vs SA)
# ------------------------------

def constant_beta(beta):
    """Return a schedule beta_t = beta (plain Metropolis)."""
    def schedule(step):
        return beta
    return schedule


def linear_annealing_beta(beta_start, beta_end, n_steps):
    """
    Linear schedule from beta_start to beta_end over n_steps.
    beta_start small -> high temperature at beginning.
    """
    def schedule(step):
        if n_steps <= 1:
            return beta_end
        frac = step / (n_steps - 1)
        return beta_start + frac * (beta_end - beta_start)
    return schedule


def exponential_annealing_beta(beta_start, beta_end, n_steps):
    """
    Exponential schedule between beta_start and beta_end.
    """
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError("beta_start and beta_end must be > 0 for exponential schedule.")

    ratio = beta_end / beta_start

    def schedule(step):
        if n_steps <= 1:
            return beta_end
        frac = step / (n_steps - 1)
        return beta_start * (ratio ** frac)

    return schedule


# ------------------------------
#  Core sampler (no plotting)
# ------------------------------

def metropolis_mcmc(N, n_steps, beta_schedule, verbose=True, seed=None):
    """
    Metropolis / Simulated Annealing sampler.

    Parameters
    ----------
    N : int
        Board size.
    n_steps : int
        Number of MCMC steps.
    beta_schedule : callable
        Function beta_schedule(step) returning beta_t at given step.
        - constant_beta(beta) => plain Metropolis
        - *_annealing_beta(...) => simulated annealing
    verbose : bool
        Whether to print progress.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        final_state, final_energy, best_state, best_energy, energy_history
    """
    if seed is not None:
        np.random.seed(seed)

    state = State3DQueens(N)
    current_energy = state.energy(recompute=True)

    best_state = state.copy()
    best_energy = current_energy

    accepted = 0
    energy_history = [current_energy]

    for step in range(n_steps):
        beta_t = beta_schedule(step)

        # Pick random (i, j)
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        old_k = state.state[i, j]

        # Propose new k != old_k
        new_k = np.random.randint(0, N - 1)
        if new_k >= old_k:
            new_k += 1

        # Apply move in-place
        state.propose_move(i, j, new_k)

        proposed_energy = state.energy(recompute=True)
        delta_E = proposed_energy - current_energy

        accept_prob = min(1.0, np.exp(-beta_t * delta_E))

        if np.random.random() < accept_prob:
            # Accept
            current_energy = proposed_energy
            accepted += 1

            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy
        else:
            # Reject: revert
            state.revert_move(i, j, old_k)
            state._energy = current_energy  # restore cached energy

        energy_history.append(current_energy)

        if verbose and (step + 1) % 1000 == 0:
            print(
                f"Step {step + 1}/{n_steps}: "
                f"energy = {current_energy}, best = {best_energy}, "
                f"beta_t = {beta_t:.4f}"
            )

    if verbose:
        print(f"Final energy: {current_energy}")
        print(f"Best energy: {best_energy}")
        print(f"Acceptance rate: {accepted / n_steps:.3f}")

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
    """
    all_histories = []
    best_energies = []

    for r in range(n_runs):
        if verbose:
            print(f"\n=== Run {r+1}/{n_runs} ===")
        res = run_single_chain(
            N=N,
            n_steps=n_steps,
            beta_schedule=beta_schedule,
            seed=base_seed + r,
            verbose=verbose,
        )
        all_histories.append(res["energy_history"])
        best_energies.append(res["best_energy"])

    return all_histories, best_energies


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


# ------------------------------
#  Main: choose schedule & run
# ------------------------------

if __name__ == "__main__":
    N = 7
    n_steps = 100000
    n_runs = 5

    # Example 1: plain Metropolis (constant beta)
    beta_const = 1.0
    beta_schedule_const = constant_beta(beta_const)

    print(f"Running {n_runs} runs with constant beta = {beta_const}")
    all_hist_const, best_const = run_experiment(
        N=N,
        n_steps=n_steps,
        beta_schedule=beta_schedule_const,
        n_runs=n_runs,
        base_seed=42,
        verbose=False,
    )

    plot_energy_histories(
        all_hist_const,
        title=f"Energy History (Metropolis, N={N}, beta={beta_const})",
        out_path="figures/energy_history_metropolis.png",
    )

    # Example 2: simulated annealing with linear schedule
    beta_start = 0.1   # high temperature (weak penalty on uphill moves)
    beta_end = 5.0     # low temperature
    beta_schedule_sa = linear_annealing_beta(beta_start, beta_end, n_steps)

    print(f"\nRunning {n_runs} runs with simulated annealing "
          f"(beta from {beta_start} to {beta_end})")

    all_hist_sa, best_sa = run_experiment(
        N=N,
        n_steps=n_steps,
        beta_schedule=beta_schedule_sa,
        n_runs=n_runs,
        base_seed=123,
        verbose=False,
    )

    plot_energy_histories(
        all_hist_sa,
        title=f"Energy History (Simulated Annealing, N={N}, "
              f"beta: {beta_start}→{beta_end})",
        out_path="figures/energy_history_sa.png",
    )

    print("\nBest energies (Metropolis):", best_const)
    print("Best energies (Simulated Annealing):", best_sa)
