import numpy as np
import matplotlib.pyplot as plt
import os


def initialize_state(N):
    all_positions = [(i, j, k) for i in range(N) for j in range(N) for k in range(N)]
    selected = np.random.choice(len(all_positions), size=N*N, replace=False)
    state = np.array([all_positions[idx] for idx in selected])
    return state


def is_attacking(i1, j1, k1, i2, j2, k2):
    if i1 == i2 or j1 == j2 or k1 == k2:
        return True
    
    if k1 == k2 and abs(i1 - i2) == abs(j1 - j2):
        return True
    
    if j1 == j2 and abs(i1 - i2) == abs(k1 - k2):
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


def energy_delta(state, queen_idx, old_pos, new_pos):
    i_old, j_old, k_old = old_pos
    i_new, j_new, k_new = new_pos
    old_attacks = 0
    new_attacks = 0
    
    for q in range(len(state)):
        if q == queen_idx:
            continue
        
        i, j, k = state[q]

        if is_attacking(i, j, k, i_old, j_old, k_old):
            old_attacks += 1
        if is_attacking(i, j, k, i_new, j_new, k_new):
            new_attacks += 1
    
    delta = new_attacks - old_attacks
    return delta


def metropolis_mcmc(N, beta, n_steps, verbose=True):
    state = initialize_state(N)
    current_energy = energy(state)
    
    best_state = state.copy()
    best_energy = current_energy
    
    if verbose:
        print(f"Initial energy: {current_energy}")
    
    occupied_set = {tuple(pos) for pos in state}
    
    energy_history = [current_energy]
    
    for step in range(n_steps):
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
        
        accept_prob = min(1.0, np.exp(-beta * delta_E))
        
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
            print(f"Step {step + 1}/{n_steps}: energy = {current_energy}, best = {best_energy}")
    
    return {
        'final_state': state,
        'final_energy': current_energy,
        'best_state': best_state,
        'best_energy': best_energy,
        'energy_history': energy_history
    }


if __name__ == "__main__":
    np.random.seed(42)
    N = 50
    beta = 10.0
    n_steps = 100_000
    
    print(f"N = {N}, beta = {beta}, n_steps = {n_steps}")
    
    results = metropolis_mcmc(N, beta, n_steps, verbose=True)
    
    print(f"Final energy: {results['final_energy']}")
    print(f"Best energy: {results['best_energy']}")
    
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['energy_history'])
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title(f'Energy History (N={N}, beta={beta})')
    plt.grid(True)
    plt.savefig('figures/energy_history.png')
    plt.close()
