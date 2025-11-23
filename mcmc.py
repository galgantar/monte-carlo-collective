import numpy as np
import matplotlib.pyplot as plt
import os


def initialize_state(N):
    all_positions = [(i, j, k) for i in range(N) for j in range(N) for k in range(N)]
    selected = np.random.choice(len(all_positions), size=N*N, replace=False)
    state = np.array([all_positions[idx] for idx in selected])
    return state


def energy(state):
    N = len(state)
    if N < 2:
        return 0
    
    count = 0
    
    for q1 in range(N):
        for q2 in range(q1 + 1, N):
            i1, j1, k1 = state[q1]
            i2, j2, k2 = state[q2]
            
            if i1 == i2 or j1 == j2 or k1 == k2:
                count += 1
                continue
            
            if k1 == k2 and abs(i1 - i2) == abs(j1 - j2):
                count += 1
                continue
            
            if j1 == j2 and abs(i1 - i2) == abs(k1 - k2):
                count += 1
                continue
            
            if i1 == i2 and abs(j1 - j2) == abs(k1 - k2):
                count += 1
                continue
            
            di = abs(i1 - i2)
            dj = abs(j1 - j2)
            dk = abs(k1 - k2)
            if di == dj == dk:
                count += 1
    
    return count


def metropolis_mcmc(N, beta, n_steps, verbose=True):
    state = initialize_state(N)
    current_energy = energy(state)
    
    best_state = state.copy()
    best_energy = current_energy
    
    if verbose:
        print(f"Initial energy: {current_energy}")
    
    all_positions = np.array([(i, j, k) for i in range(N) for j in range(N) for k in range(N)])
    
    accepted = 0
    energy_history = [current_energy]
    
    for step in range(n_steps):
        queen_idx = np.random.randint(0, N*N)
        old_pos = state[queen_idx]
        
        occupied_set = {tuple(pos) for pos in state}
        empty_positions = [pos for pos in all_positions if tuple(pos) not in occupied_set]
        
        if len(empty_positions) == 0:
            continue
        
        new_pos_idx = np.random.randint(0, len(empty_positions))
        new_pos = empty_positions[new_pos_idx]
        
        proposed_state = state.copy()
        proposed_state[queen_idx] = new_pos
        
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
            print(f"Step {step + 1}/{n_steps}: energy = {current_energy}, best = {best_energy}")
    
    if verbose:
        print(f"Final energy: {current_energy}")
        print(f"Best energy: {best_energy}")
    
    return {
        'final_state': state,
        'final_energy': current_energy,
        'best_state': best_state,
        'best_energy': best_energy,
        'energy_history': energy_history
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
    
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['energy_history'])
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title(f'Energy History (N={N}, beta={beta})')
    plt.grid(True)
    plt.savefig('figures/energy_history.png')
    plt.close()
