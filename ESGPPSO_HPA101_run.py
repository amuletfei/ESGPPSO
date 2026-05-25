import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pso_mc_HPA101_class import PSO


def run_single_experiment(n_div, level, recount):
    PSOi = PSO(n_div=n_div, le=level)
    c, b = PSOi.PSO_ReRun(recount)
    n_div_val = PSOi.n_div
    level_val = PSOi.level
    dir_path_avg = f'./MCResult/HPA101/{n_div_val}N/{level_val}le/avgResult/'
    os.makedirs(dir_path_avg, exist_ok=True)
    np.savetxt(f'{dir_path_avg}avg_score_N{n_div_val}_L{level_val}.csv', c, delimiter=",")
    dir_path_total = f'./MCResult/HPA101/{n_div_val}N/{level_val}le/totalResult/'
    os.makedirs(dir_path_total, exist_ok=True)
    np.savetxt(f'{dir_path_total}totalResult_N{n_div_val}_L{level_val}.csv', b, delimiter=",")
    return n_div_val, level_val


def main():
    recount = 30
    n_div_list = [3, 4, 5]
    level_list = [0, 1, 2]
    max_workers = 9
    total_tasks = len(n_div_list) * len(level_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for n_div in n_div_list:
            for level in level_list:
                future = executor.submit(run_single_experiment, n_div, level, recount)
                futures[future] = (n_div, level)
        for future in as_completed(futures):
            orig_n_div, orig_level = futures[future]
            try:
                res_n_div, res_level = future.result()
                print(f"Experiment [n_div: {res_n_div}, level: {res_level}] is over.")
            except Exception as exc:
                print(f"Experiment [n_div: {orig_n_div}, level: {orig_level}] generated an exception: {exc}")

    print(">>> all results saved to ./MCResult/ ")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()