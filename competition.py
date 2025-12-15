from experiments import linear_annealing_beta
from experiments import run_single_chain_board_multithread
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time


N = 8
n_runs = 10
n_steps = 100000
init_mode = "random"
beta_schedule = linear_annealing_beta
seed = 42
verbose = False
run_idx = 0
early_stop_patience = None
multithread_runner = run_single_chain_board_multithread
schedule_params = {
    "type": "linear_annealing",
    "beta_start": 1.0,
    "beta_end": 3.0,
}

base_seed = 42


args = (
    N,
    n_steps,
    init_mode,
    beta_schedule,
    seed,
    verbose,
    run_idx,
    early_stop_patience,
)


if __name__ == "__main__":
    run_args = [
        (N, n_steps, init_mode, schedule_params, base_seed + r, verbose, r, early_stop_patience)
        for r in range(n_runs)
    ]

    with ProcessPoolExecutor(max_workers=n_runs) as executor:
        future_to_run = {
            executor.submit(multithread_runner, args): r
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

            if verbose:
                print(f"Best energy for run {run_idx}: {result['best_energy']}")
            
    results.sort(key=lambda x: x["best_energy"])

    print("Best energies:")
    for res in results:
        print(res["best_energy"])

    print(results[0]["best_state"].heights)
    
    os.makedirs("competition_results", exist_ok=True)
    filename = f"best_heights_{N}_{time.strftime("%Y%m%d_%H%M")}.txt"
    with open(os.path.join("competition_results", filename), "w") as f:
        best_heights = results[0]["best_state"].heights
        for i in range(N):
            for j in range(N):
                f.write(f"{i},{j},{best_heights[i, j]}\n")
