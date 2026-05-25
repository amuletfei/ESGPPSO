import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from bio_pso_mc_cec2022 import PSO


def run_single_function(nu, recount):

    PSOi = PSO(nu)
    c, b = PSOi.PSO_ReRun(recount)
    demension = PSOi.dimensions
    dir_path_avg = f'./MC2022Result/{demension}D/avgResult/'
    os.makedirs(dir_path_avg, exist_ok=True)
    np.savetxt(dir_path_avg + 'avg_score{}.csv'.format(nu), c, delimiter=",")
    dir_path_total = f'./MC2022Result/{demension}D/totalResult/'
    os.makedirs(dir_path_total, exist_ok=True)
    np.savetxt(dir_path_total + 'totalResult{}.csv'.format(nu), b, delimiter=",")
    return nu


def main():
    recount = 30
    # cec 2022 f1-f12
    func_list = [nu for nu in range(1, 13)]
    max_workers = 15

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_function, nu, recount): nu for nu in func_list}
        for future in as_completed(futures):
            nu = futures[future]
            try:
                res_nu = future.result()
                print(f"function: {res_nu:02d} is over.")
            except Exception as exc:
                print(f"function: {nu:02d} generated an exception: {exc}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()