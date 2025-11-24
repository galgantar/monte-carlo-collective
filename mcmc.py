import numpy as np
import matplotlib.pyplot as plt
import os


def initialize_state(N):
    all_positions = [(i, j, k) for i in range(N) for j in range(N) for k in range(N)]
    selected = np.random.choice(len(all_positions), size=N*N, replace=False)
    state = np.array([all_positions[idx] for idx in selected])
    return state


def is_attacking(i1, j1, k1, i2, j2, k2):
    if i1 == i2 and j1 == j2:
        return True
    if i1 == i2 and k1 == k2:
        return True
    if j1 == j2 and k1 == k2:
        return True
    if k1 == k2 and abs(i1 - i2) == abs(j1 - j2):
        return True
    if j1 == j2 and abs(i1 - i2) == abs(k1 - k2):
        return True
    if i1 == i2 and abs(j1 - j2) == abs(k1 - k2):
        return True
    if abs(i1 - i2) == abs(j1 - j2) == abs(k1 - k2):
        return True
    return False


def energy(state):
    N = len(state)
    count = 0
    
    for q1 in range(N):
        for q2 in range(q1 + 1, N):
            i1, j1, k1 = state[q1]
            i2, j2, k2 = state[q2]
            
            if is_attacking(i1, j1, k1, i2, j2, k2):
                count += 1
    return count


def conflicts_for_queen_vectorized(state, queen_idx, i, j, k):
    if len(state) == 0:
        return 0
    
    state_array = np.array(state)
    I = state_array[:, 0]
    J = state_array[:, 1]
    K = state_array[:, 2]
    
    not_self = np.arange(len(state)) != queen_idx
    
    same_i = (I == i)
    same_j = (J == j)
    same_k = (K == k)
    
    di = np.abs(I - i)
    dj = np.abs(J - j)
    dk = np.abs(K - k)
    
    same_ij = same_i & same_j
    same_ik = same_i & same_k
    same_jk = same_j & same_k
    
    plane_k_diag = same_k & (di == dj) & (di > 0)
    plane_j_diag = same_j & (di == dk) & (di > 0)
    plane_i_diag = same_i & (dj == dk) & (dj > 0)
    space_diag = (di == dj) & (dj == dk) & (di > 0)
    
    attacked = same_ij | same_ik | same_jk | plane_k_diag | plane_j_diag | plane_i_diag | space_diag
    attacked = attacked & not_self
    
    return int(attacked.sum())


def energy_delta(state, queen_idx, old_pos, new_pos):
    i_old, j_old, k_old = old_pos
    i_new, j_new, k_new = new_pos
    
    old_attacks = conflicts_for_queen_vectorized(state, queen_idx, i_old, j_old, k_old)
    new_attacks = conflicts_for_queen_vectorized(state, queen_idx, i_new, j_new, k_new)
    
    delta = new_attacks - old_attacks
    return delta


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


def metropolis_mcmc(N, beta_schedule, n_steps, verbose=True):
    state = initialize_state(N)
    current_energy = energy(state)
    
    best_state = state.copy()
    best_energy = current_energy
    
    if verbose:
        print(f"Initial energy: {current_energy}")
    
    occupied_set = {tuple(pos) for pos in state}
    
    energy_history = [current_energy]
    
    for step in range(n_steps):
        beta_t = beta_schedule(step)
        
        queen_idx = np.random.randint(0, N*N)
        old_pos = tuple(state[queen_idx])

        while True:
            i_new = np.random.randint(0, N)
            j_new = np.random.randint(0, N)
            k_new = np.random.randint(0, N)
            new_pos = (i_new, j_new, k_new)
            
            if new_pos not in occupied_set:
                break
        
        delta_E = energy_delta(state, queen_idx, old_pos, new_pos)
        proposed_energy = current_energy + delta_E
        
        accept_prob = min(1.0, np.exp(-beta_t * delta_E))
        
        if np.random.random() < accept_prob:
            occupied_set.remove(old_pos)
            occupied_set.add(new_pos)
            state[queen_idx] = np.array([i_new, j_new, k_new])
            current_energy = proposed_energy
            
            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy

        energy_history.append(current_energy)
        
        if verbose and (step + 1) % 1000 == 0:
            print(f"Step {step + 1}/{n_steps}: energy = {current_energy}, best = {best_energy}, beta_t = {beta_t:.3f}")
    
    return {
        'final_state': state,
        'final_energy': current_energy,
        'best_state': best_state,
        'best_energy': best_energy,
        'energy_history': energy_history
    }


if __name__ == "__main__":
    np.random.seed(42)
    N = 6
    n_steps = 100000
    
    beta_start = 0.01
    beta_end = 10.0 
    beta_schedule = linear_annealing_beta(beta_start, beta_end, n_steps)
    print(f"N = {N}, Simulated Annealing: beta = {beta_start} -> {beta_end}, n_steps = {n_steps}")
    
    results = metropolis_mcmc(N, beta_schedule, n_steps, verbose=True)
    
    print(f"Final energy: {results['final_energy']}")
    print(f"Best energy: {results['best_energy']}")
    print(f"Sanity: {energy(results['best_state'])}")
    
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['energy_history'])
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title(f'Energy History (N={N}, SA: beta {beta_start}->{beta_end})')
    plt.grid(True)
    plt.savefig('figures/energy_history_sa.png')
    plt.close()
