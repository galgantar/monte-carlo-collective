from experiments import linear_annealing_beta
from experiments import run_single_chain_board_multithread
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import numpy as np
from mcmc_board import State3DQueensBoard



def linear_annealing_beta(beta_start, beta_end, n_steps):
    def schedule(step):
        if n_steps <= 1:
            return beta_end
        frac = step / (n_steps - 1)
        return beta_start + frac * (beta_end - beta_start)
    return schedule


def metropolis_mcmc_board(N, n_steps, init_mode, beta_schedule, verbose=True, seed=None, run_idx=None, early_stop_patience=None):
    """MCMC for board-constrained version (one queen per (i,j) pair)."""
    if early_stop_patience in (None, 'None', 'null'):
        early_stop_patience = None
    
    if seed is not None:
        np.random.seed(seed)

    state = State3DQueensBoard(N, init_mode=init_mode)
    current_energy = state.energy(recompute=True)

    best_state = state.copy()
    best_energy = current_energy

    accepted = 0
    energy_history = [current_energy]
    
    accepted_steps = []
    rejected_steps = []

    no_improvement_steps = 0

    if verbose and n_steps > 0:
        next_report = max(1, n_steps // 10)


    for step in range(n_steps):
        beta_t = beta_schedule(step)

        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        old_k = state.heights[i, j]

        old_conflicts = state.conflicts_for_position(i, j, old_k)

        new_k = np.random.randint(0, N)
        while new_k == old_k:
            new_k = np.random.randint(0, N)

        new_conflicts = state.conflicts_for_position(i, j, new_k)

        delta_E = new_conflicts - old_conflicts
        proposed_energy = current_energy + delta_E

        accept_prob = min(1.0, np.exp(-beta_t * delta_E))
        was_accepted = np.random.random() < accept_prob

        if was_accepted:
            accepted_steps.append(step)
        else:
            rejected_steps.append(step)

        if was_accepted:
            state.propose_move(i, j, new_k)
            current_energy = proposed_energy
            state._energy = current_energy
            accepted += 1

            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1
        else:
            no_improvement_steps += 1

        if early_stop_patience is not None and no_improvement_steps >= early_stop_patience:
            if verbose:
                print(f"Early stopping: Final energy = {current_energy}")
                print(f"Early stopping: Best energy = {best_energy}")
            break

        energy_history.append(current_energy)

        if verbose and (step + 1) % next_report == 0:
            print(f"Step {step + 1}: Current energy = {current_energy}")

    energy_arr = np.array(energy_history)
    steps_to_best = int(np.argmin(energy_arr))

    return {
        "final_state": state,
        "final_energy": current_energy,
        "best_state": best_state,
        "best_energy": best_energy,
        "energy_history": energy_history,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "steps_to_best": steps_to_best,
    }


def run_single_chain_board_multithread(args):
    """Wrapper function for parallel execution of a single chain (board)."""
    (N, n_steps, init_mode, seed, verbose, run_idx, early_stop_patience, beta_start, beta_end) = args
    beta_schedule = linear_annealing_beta(beta_start, beta_end, n_steps)
    start_time = time.time()
    res = metropolis_mcmc_board(
        N=N,
        n_steps=n_steps,
        init_mode=init_mode,
        beta_schedule=beta_schedule,
        seed=seed,
        verbose=verbose,
        run_idx=run_idx,
        early_stop_patience=early_stop_patience,
    )
    end_time = time.time()
    duration = end_time - start_time
    return {
        "run_idx": run_idx,
        "best_state": res["best_state"],
        "energy_history": res["energy_history"],
        "best_energy": res["best_energy"],
        "duration": duration,
        "accepted_steps": res["accepted_steps"],
        "rejected_steps": res["rejected_steps"],
        "steps_to_best": res["steps_to_best"],
    }



def main():
    N = 5
    n_runs = 10
    n_steps = 100000
    init_mode = "random"
    run_idx = 0
    early_stop_patience = None
    beta_start = 1.0
    beta_end = 3.0

    base_seed = 42

    run_args = [
        (N, n_steps, init_mode, base_seed + r, True, r, early_stop_patience, beta_start, beta_end)
        for r in range(n_runs)
    ]

    with ProcessPoolExecutor(max_workers=n_runs) as executor:
        future_to_run = {
            executor.submit(run_single_chain_board_multithread, args): r
            for r, args in enumerate(run_args)
        }

        results = [None] * n_runs
        completed = 0

        for future in as_completed(future_to_run):
            run_idx = future_to_run[future]
            result = future.result()
            result["run_idx"] = run_idx
            results[run_idx] = result
            completed += 1
            
    results.sort(key=lambda x: x["best_energy"])

    print("Best energies: ", [res["best_energy"] for res in results])

    print(results[0]["best_state"].heights)
    
    os.makedirs("competition_results", exist_ok=True)
    filename = f"best_heights_{N}_{time.strftime("%Y%m%d_%H%M")}.txt"
    with open(os.path.join("competition_results", filename), "w") as f:
        best_heights = results[0]["best_state"].heights
        for i in range(N):
            for j in range(N):
                f.write(f"{i},{j},{best_heights[i, j]}\n")


if __name__ == "__main__":
    main()
