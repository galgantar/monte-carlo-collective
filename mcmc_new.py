import numpy as np
import matplotlib.pyplot as plt
import os


def initialize_state(N):
    """
    State: N x N array.
    For each (i, j), state[i, j] = k in {0, ..., N-1},
    meaning there is exactly one queen at (i, j, k).
    """
    # Random k for each (i, j)
    state = np.random.randint(0, N, size=(N, N))
    return state


def energy(state):
    """
    Count the number of pairs of queens that attack each other.

    Input:
        state: (N, N) array of heights k.
               Queen positions are (i, j, k_ij).

    Output:
        count: integer number of attacking pairs.
    """
    N = state.shape[0]
    # Total queens = N^2
    count = 0

    # Build a list of all queen positions (i, j, k)
    positions = []
    for i in range(N):
        for j in range(N):
            k = state[i, j]
            positions.append((i, j, k))

    Q = len(positions)
    if Q < 2:
        return 0

    # Check all pairs
    for q1 in range(Q):
        i1, j1, k1 = positions[q1]
        for q2 in range(q1 + 1, Q):
            i2, j2, k2 = positions[q2]

            # Sharing coordinates
            if i1 == i2 and j1 == j2: 
                count += 1
                continue
            if i1 == i2 and k1 == k2: 
                count += 1
                continue
            if j1 == j2 and k1 == k2: 
                count += 1
                continue

            # 2D diagonals in planes k = const
            if k1 == k2 and abs(i1 - i2) == abs(j1 - j2):
                count += 1
                continue

            # 2D diagonals in planes j = const
            if j1 == j2 and abs(i1 - i2) == abs(k1 - k2):
                count += 1
                continue

            # 2D diagonals in planes i = const
            if i1 == i2 and abs(j1 - j2) == abs(k1 - k2):
                count += 1
                continue

            # 3D space diagonals
            di = abs(i1 - i2)
            dj = abs(j1 - j2)
            dk = abs(k1 - k2)
            if di == dj == dk:
                count += 1

    return count


def metropolis_mcmc(N, beta, n_steps, verbose=True):
    """
    Metropolis sampler on the new state space:

    - State: N x N array of heights.
    - Proposal: pick random (i, j), change height to a different k'.
    """
    state = initialize_state(N)
    current_energy = energy(state)

    best_state = state.copy()
    best_energy = current_energy

    if verbose:
        print(f"Initial energy: {current_energy}")

    accepted = 0
    energy_history = [current_energy]

    for step in range(n_steps):
        # Pick a random (i, j)
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        old_k = state[i, j]

        # Propose a new k' != old_k
        new_k = np.random.randint(0, N - 1)
        if new_k >= old_k:
            new_k += 1

        # Build proposed state
        proposed_state = state.copy()
        proposed_state[i, j] = new_k

        proposed_energy = energy(proposed_state)
        delta_E = proposed_energy - current_energy

        accept_prob = min(1.0, np.exp(-beta * delta_E))

        if np.random.random() < accept_prob:
            state = proposed_state
            current_energy = proposed_energy
            accepted += 1

            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy

        energy_history.append(current_energy)

        if verbose and (step + 1) % 1000 == 0:
            print(
                f"Step {step + 1}/{n_steps}: "
                f"energy = {current_energy}, best = {best_energy}"
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


if __name__ == "__main__":
    np.random.seed(42)
    N = 7
    beta = 1.0
    n_steps = 10000

    print(f"N = {N}, beta = {beta}, n_steps = {n_steps}")

    results = metropolis_mcmc(N, beta, n_steps, verbose=True)

    print(f"Final energy: {results['final_energy']}")
    print(f"Best energy: {results['best_energy']}")

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(results["energy_history"])
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title(f"Energy History (N={N}, beta={beta})")
    plt.grid(True)
    plt.savefig("figures/energy_history.png")
    plt.close()
