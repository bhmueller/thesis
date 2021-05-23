from multiprocessing import Pool
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from python.auxiliary import rust_model_morris
from econsa.morris import elementary_effects
from econsa.morris import _get_uniform_base_draws
from econsa.morris import _uniform_to_standard_normal
from econsa.morris import _dependent_draws
from econsa.morris import _shift_sample
from econsa.morris import _shift_cov
from econsa.morris import _calculate_indices
from econsa.morris import _calculate_cumulative_indices

use_eoq = True


def eoq_model_morris(x, r=0.1):
    """EOQ Model that handles data as presented by elementary_effects."""
    m = x["value"][0]
    c = x["value"][1]
    s = x["value"][2]
    # m_below_zero = 0
    if m < 0:
        m = 0
        # print('m < 0')
    elif c < 0:
        raise ValueError
    elif s < 0:
        s = 0
        # print('s < 0')
    else:
        pass

    return np.sqrt((24 * m * s) / (r * c))
    # if (24 * m * s) / (r * c) <= 0:
    #     return 0
    # elif (24 * m * s) / (r * c) > 0:
    #     return np.sqrt((24 * m * s) / (r * c))
    # else:
    #     raise ValueError


def model_func(params):
    return 5


# Choose model.
if use_eoq is True:
    func = eoq_model_morris
elif use_eoq is False:
    func = model_func
else:
    raise ValueError

params = pd.DataFrame(data=[5.345, 0.0135, 2.15], columns=["value"])
cov = pd.DataFrame(data=np.diag([1, 0.000001, 0.01]))

n_draws = 100
n_cores = cpu_count()
sampling_scheme = "sobol"

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
# def _evaluate_model(func, params, sample, n_cores):
#     n_draws, n_params, _ = sample.shape
#     evals = np.zeros((n_draws, n_params))


#     inputs = []
#     for d in range(n_draws):
#         for p in range(n_params):
#             par = params.copy()
#             par["value"] = sample[d, p]
#             inputs.append(par)

#     # TypeError joblib; cannot unpack non-it fct. object.
#     inputs = np.asarray(inputs)

#     p = Pool(processes=n_cores)

#     # The below line causes malfunction.
#     # evals_flat = p.map(func, inputs)

#     # Alternative: use map only. No multiprocessing.
#     # evals_flat = map(func, inputs)
#     # evals_flat = np.asarray(list(evals_flat))

#     # This was in the original script as well, but commented out.
#     evals_flat = Parallel(n_jobs=n_cores)(delayed(func)(inp) for inp in inputs)

#     evals = np.array(evals_flat).reshape(n_draws, n_params)

#     return evals


def _evaluate_model(func, params, sample, n_cores):
    """Do all model evaluations needed for the EE indices.
    Args:
        func (function): Maps params to quantity of interest.
        params (pd.DataFrame): Model parameters. The "value" column will be replaced
            with values from the morris samples.
        sample (np.ndarray): Array of shape (n_draws, n_params, n_params).
            Morris samples in the multivariate normal space.
    Returns:
        evals (np.ndarray): Array of shape (n_draws, n_params) with model evaluations
    """
    n_draws, n_params, _ = sample.shape
    evals = np.zeros((n_draws, n_params))

    inputs = []
    for d in range(n_draws):
        for p in range(n_params):
            par = params.copy()
            par["value"] = sample[d, p]
            inputs.append(par)

    p = Pool(processes=n_cores)
    # evals_flat = p.map(func, inputs)

    evals_flat = Parallel(n_jobs=n_cores)(delayed(func)(inp) for inp in inputs)

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
