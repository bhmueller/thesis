from multiprocessing import Pool
from multiprocessing import cpu_count
from joblib import Parallel
from delayed.delay import delayed
import numpy as np
import pandas as pd
from econsa.morris import elementary_effects
from econsa.morris import _get_uniform_base_draws
from econsa.morris import _uniform_to_standard_normal
from econsa.morris import _dependent_draws
from econsa.morris import _shift_sample
from econsa.morris import _shift_cov
from econsa.morris import _evaluate_model
from econsa.morris import _calculate_indices
from econsa.morris import _calculate_cumulative_indices

# EOQ model.
def eoq_model_ndarray(x, r=0.1):
    """EOQ Model that accepts ndarray."""
    m = x[:, 0]
    c = x[:, 1]
    s = x[:, 2]
    return np.sqrt((24 * m * s) / (r * c))

def eoq_model_df(x, r=0.1):
    """EOQ Model that handles pandas DataFrames."""
    m = x['value'][0]
    c = x['value'][1]
    s = x['value'][2]
    return np.sqrt((24 * m * s) / (r * c))

# def model_func(params):
#     return 5

func = eoq_model_df


params = pd.DataFrame(data=[5.345, 0.0135, 2.15], columns=['value'])
cov = pd.DataFrame(data=np.diag([1, 0.000001, 0.01]))

cov

params

eoq_model_ndarray(np.array([[5, 3, 2]]))

n_draws = 100
# n_cores = cpu_count()
n_cores = 1
sampling_scheme = 'sobol' 

u_a, u_b = _get_uniform_base_draws(n_draws, len(params), sampling_scheme)
z_a = _uniform_to_standard_normal(u_a)
z_b = _uniform_to_standard_normal(u_b)
mean_np = params["value"].to_numpy()
cov_np = cov.to_numpy()

dep_samples_ind_x, a_sample_ind_x = _dependent_draws(
    z_a,
    z_b,
    mean_np,
    cov_np,
    "ind",
)


dep_samples_corr_x, _ = _dependent_draws(z_a, z_b, mean_np, cov_np, "corr")


# Dive into _evaluate_model
sample = dep_samples_ind_x
def _evaluate_model(func, params, sample, n_cores):
    n_draws, n_params, _ = sample.shape
    evals = np.zeros((n_draws, n_params))


    inputs = []
    for d in range(n_draws):
        for p in range(n_params):
            par = params.copy()
            par["value"] = sample[d, p]
            inputs.append(par)



    p = Pool(processes=n_cores)

    # The below line causes malfunction.
    # evals_flat = p.map(func, inputs)

    # Alternative: use map only. No multiprocessing.
    evals_flat = map(func, inputs)
    evals_flat = np.asarray(list(evals_flat))

    # evals_flat = Parallel(n_jobs=n_cores)(delayed(func)(inp) for inp in inputs)

    evals = np.array(evals_flat).reshape(n_draws, n_params)

    return evals


evals_ind = _evaluate_model(func, params, dep_samples_ind_x, n_cores)

evals_base_ind = _evaluate_model(func, params, a_sample_ind_x, n_cores)

evals_corr = _evaluate_model(func, params, dep_samples_corr_x, n_cores)

evals_base_corr = _shift_sample(evals_base_ind, -1)

deltas = u_b - u_a

mu_ind, sigma_ind = _calculate_indices(
    evals_ind,
    evals_base_ind,
    deltas,
    params.index,
)

mu_corr, sigma_corr = _calculate_indices(
    evals_corr,
    evals_base_corr,
    deltas,
    params.index,
)

mu_ind_cum, sigma_ind_cum = _calculate_cumulative_indices(
    evals_ind,
    evals_base_ind,
    deltas,
    params.index,
)
mu_corr_cum, sigma_corr_cum = _calculate_cumulative_indices(
    evals_corr,
    evals_base_corr,
    deltas,
    params.index,
)

res = {
    "mu_ind": mu_ind,
    "mu_corr": mu_corr,
    "sigma_ind": sigma_ind,
    "sigma_corr": sigma_corr,
    "mu_ind_cum": mu_ind_cum,
    "mu_corr_cum": mu_corr_cum,
    "sigma_ind_cum": sigma_ind_cum,
    "sigma_corr_cum": sigma_corr_cum,
}
