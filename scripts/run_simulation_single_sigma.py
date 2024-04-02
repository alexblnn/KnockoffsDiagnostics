import numpy as np
from hidimstat.knockoffs.data_simulation import simu_data
from utils import report_fdp_tdp_size, get_knockoffs_stats
from utils import perform_inference_given_KO
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import warnings
import sys
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # also affect subprocesses

n, p = 500, 500
rho = 0.7
snr = 7
sparsity = 0.1
draws = 2
n_jobs = 40
seed = 42
q = 0.05
n_runs = 20
true_cov = False
rng = np.random.default_rng(seed)

n_folds = 5

sigma = float(sys.argv[1])

fdrs = np.ones((2, n_runs))
accs = np.empty((2, n_folds))

for runs in tqdm(range(n_runs)):
    # mu_vec = np.ones(p) * mu
    _, _, beta, non_zero_index, _ = simu_data(
        n, p, rho=rho, snr=snr, sparsity=sparsity, no_blobs=True
    )

    X_norms = np.random.normal(size=(50, 50, 100), scale=5)
    X_norms_smoothed = gaussian_filter(X_norms, sigma=sigma)

    X_ht = X_norms_smoothed.reshape((500, 500))

    eps = np.random.normal(0, 1, n)
    prod_temp = np.dot(X_ht, beta)
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    y_ht = prod_temp + noise_mag * eps

    X_ht = StandardScaler().fit_transform(X_ht)

    ko_stats_NG, X_tildes_NG, alphas_chosen_NG, active_sets_NG = get_knockoffs_stats(
        X_ht,
        y_ht,
        centered=False,
        draws=draws,
        n_jobs=n_jobs,
        return_alpha=False,
        statistic="lasso_cv",
        gaussian=False,
        seed=seed,
    )

    if true_cov:
        ko_stats_G, X_tildes_G, alphas_chosen_G, active_sets_G = get_knockoffs_stats(
            X_ht,
            y_ht,
            centered=False,
            true_covar=Sigma,
            draws=draws,
            n_jobs=n_jobs,
            return_alpha=False,
            statistic="lasso_cv",
            gaussian=True,
            seed=seed,
        )

    else:
        ko_stats_G, X_tildes_G, alphas_chosen_G, active_sets_G = get_knockoffs_stats(
            X_ht,
            y_ht,
            centered=False,
            draws=draws,
            n_jobs=n_jobs,
            return_alpha=False,
            statistic="lasso_cv",
            gaussian=True,
            seed=seed,
        )



    if runs == n_runs - 1:
        fdrs[0][runs], accs[0] = perform_inference_given_KO(
            X_ht, ko_stats_G, X_tildes_G, q, beta, draws=draws, n_jobs=n_jobs, diag=True
        )
        fdrs[1][runs], accs[1] = perform_inference_given_KO(
            X_ht, ko_stats_NG, X_tildes_NG, q, beta, draws=draws, n_jobs=n_jobs, diag=True
        )

    else:
        fdrs[0][runs] = perform_inference_given_KO(
            X_ht, ko_stats_G, X_tildes_G, q, beta, draws=draws, n_jobs=n_jobs,
        )
        fdrs[1][runs] = perform_inference_given_KO(
            X_ht, ko_stats_NG, X_tildes_NG, q, beta, draws=draws, n_jobs=n_jobs,
        )


print(np.mean(fdrs, axis=1))

np.save(
    f"../results/fdr{q}/results_n{n}_rho{rho}_fdr{q}_snr{snr}_sparsity{sparsity}_sigma{sigma}_smooth_3D.npy",
    fdrs,
)
np.save(
    f"../results/fdr{q}/results_n{n}_rho{rho}_fdr{q}_snr{snr}_sparsity{sparsity}_sigma{sigma}_smooth_3D_c2st.npy",
    accs,
)
