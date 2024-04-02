import numpy as np
import sanssouci as sa
from tqdm import tqdm
from scipy.stats import hmean
from sklearn.linear_model import (LassoCV, LinearRegression, LassoLarsCV, Lasso,
                                  LogisticRegression, LogisticRegressionCV,
                                  ElasticNetCV, LassoLars)
from joblib import Parallel, delayed
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_memory

from hidimstat.knockoffs.gaussian_knockoff import (_estimate_distribution,
                                                   gaussian_knockoff_generation)
from hidimstat.knockoffs.knockoff_aggregation import _empirical_pval
from hidimstat.knockoffs.stat_coef_diff import _coef_diff_threshold

from sklearn.covariance import (GraphicalLassoCV, empirical_covariance,
                                ledoit_wolf)

from scipy.linalg import toeplitz
from sklearn.utils import shuffle

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from valdiags.vanillaC2ST import c2st_scores
import pandas as pd

from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
import xgboost as xgb


def quantile_aggregation(pvals, gamma=0.5, gamma_min=0.05, adaptive=False, drop_gamma=False):
    if pvals.shape[0] == 1:
        return pvals[0]
    if adaptive:
        return _adaptive_quantile_aggregation(pvals, gamma_min)
    else:
        return _fixed_quantile_aggregation(pvals, gamma, drop_gamma=drop_gamma)


def fdr_threshold(pvals, fdr=0.1, method='bhq', reshaping_function=None):
    if method == 'bhq':
        return _bhq_threshold(pvals, fdr=fdr)
    elif method == 'bhy':
        return _bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function)
    elif method == 'ebh':
        return _ebh_threshold(pvals, fdr=fdr)
    else:
        raise ValueError(
            '{} is not support FDR control method'.format(method))


def _bhq_threshold(pvals, fdr=0.1):
    """Standard Benjamini-Hochberg for controlling False discovery rate
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        return pvals_sorted[selected_index]
    else:
        return -1.0


def _ebh_threshold(evals, fdr=0.1):
    """e-BH procedure for FDR control (see Wang and Ramdas 2021)
    """
    n_features = len(evals)
    evals_sorted = -np.sort(-evals)  # sort in descending order
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if evals_sorted[i] >= n_features / (fdr * (i + 1)):
            selected_index = i
            break
    if selected_index <= n_features:
        return evals_sorted[selected_index]
    else:
        return np.infty


def _bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    """Benjamini-Hochberg-Yekutieli procedure for controlling FDR, with input
    shape function. Reference: Ramdas et al (2017)
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    # Default value for reshaping function -- defined in
    # Benjamini & Yekutieli (2001)
    if reshaping_function is None:
        temp = np.arange(n_features)
        sum_inverse = np.sum(1 / (temp + 1))
        return _bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
        if selected_index <= n_features:
            return pvals_sorted[selected_index]
        else:
            return -1.0


def _fixed_quantile_aggregation(pvals, gamma=0.5, drop_gamma=False):
    """Quantile aggregation function based on Meinshausen et al (2008)

    Parameters
    ----------
    pvals : 2D ndarray (n_bootstrap, n_test)
        p-value (adjusted)

    gamma : float
        Percentile value used for aggregation.

    Returns
    -------
    1D ndarray (n_tests, )
        Vector of aggregated p-value
    """
    if drop_gamma:
        converted_score = np.percentile(pvals, q=100*gamma, axis=0)
    
    else:
        converted_score = (1 / gamma) * (np.percentile(pvals, q=100*gamma, axis=0))

    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05):
    """adaptive version of the quantile aggregation method, Meinshausen et al.
    (2008)"""
    gammas = np.arange(gamma_min, 1.05, 0.05)
    list_Q = np.array([
        _fixed_quantile_aggregation(pvals, gamma) for gamma in gammas])

    return np.minimum(1, (1 - np.log(gamma_min)) * list_Q.min(0))


def simu_data(n, p, rho=0.25, snr=2.0, sparsity=0.06, effect=1.0, seed=None):
    """Function to simulate data follow an autoregressive structure with Toeplitz
    covariance matrix.
    Adapted from hidimstat: https://github.com/ja-che/hidimstat

    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of variables
    sparsity : float, optional
        ratio of number of variables with non-zero coefficients over total
        coefficients
    rho : float, optional
        correlation parameter
    effect : float, optional
        signal magnitude, value of non-null coefficients
    seed : None or Int, optional
        random seed for generator

    Returns
    -------
    X : ndarray, shape (n, p)
        Design matrix resulted from simulation
    y : ndarray, shape (n, )
        Response vector resulted from simulation
    beta_true : ndarray, shape (n, )
        Vector of true coefficient value
    non_zero : ndarray, shape (n, )
        Vector of non zero coefficients index

    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # Number of non-null
    k = int(sparsity * p)

    # Generate the variables from a multivariate normal distribution
    mu = np.zeros(p)
    Sigma = toeplitz(rho ** np.arange(0, p))  # covariance matrix of X
    # X = np.dot(np.random.normal(size=(n, p)), cholesky(Sigma))
    X = rng.multivariate_normal(mu, Sigma, size=(n))
    # Generate the response from a linear model
    blob_indexes = np.linspace(0, p - 6, int(k/5), dtype=int)
    non_zero = np.array([np.arange(i, i+5) for i in blob_indexes])
    # non_zero = rng.choice(p, k, replace=False)
    beta_true = np.zeros(p)
    beta_true[non_zero] = effect
    eps = rng.standard_normal(size=n)
    prod_temp = np.dot(X, beta_true)
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    y = prod_temp + noise_mag * eps

    return X, y, beta_true, non_zero, Sigma


def get_null_pis(B, p):
    """
    Sample from the joint distribution of pi statistics
    under the null.

    Parameters
    ----------

    B: int
        Number of samples
    p: int
        Number of variables

    Returns
    -------
    pi0 : array-like of shape (B, p)
        Each row contains a vector of
        p \pi statistics
    
    """
    pi0 = np.zeros((B, p))
    for b in range(B):
        signs = (np.random.binomial(1, 0.5, size=p) * 2) - 1
        Z = 0
        for j in range(p):
            if signs[j] < 0:
                pi0[b][j] = 1
                Z += 1
            else:
                pi0[b][j] = (1 + Z) / p
    pi0 = np.sort(pi0, axis=1)
    return pi0

def get_pi_template(B, p):
    """
    Build a template by sampling from the joint distribution 
    of pi statistics under the null.

    Parameters
    ----------

    B: int
        Number of samples
    p: int
        Number of variables

    Returns
    -------
    template : array-like of shape (B, p)
        Sorted set of candidate threshold families
    
    """
    pi0 = get_null_pis(B, p)
    template = np.sort(pi0, axis=0) # extract quantile curves
    return template


def _estimate_distribution(X, shrink=True, cov_estimator='ledoit_wolf', n_jobs=1):
    """
    Adapted from hidimstat: https://github.com/ja-che/hidimstat
    """
    alphas = [1e-3, 1e-2, 1e-1, 1]

    mu = X.mean(axis=0)
    Sigma = empirical_covariance(X)

    if shrink or not _is_posdef(Sigma):

        if cov_estimator == 'ledoit_wolf':
            Sigma_shrink = ledoit_wolf(X, assume_centered=True)[0]

        elif cov_estimator == 'graph_lasso':
            model = GraphicalLassoCV(alphas=alphas, n_jobs=n_jobs)
            Sigma_shrink = model.fit(X).covariance_

        else:
            raise ValueError('{} is not a valid covariance estimated method'
                             .format(cov_estimator))

        return mu, Sigma_shrink

    return mu, Sigma


def _is_posdef(X, tol=1e-14):
    """Check a matrix is positive definite by calculating eigenvalue of the
    matrix. Adapted from hidimstat: https://github.com/ja-che/hidimstat

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples x n_features)
        Matrix to check

    tol : float, optional
        minimum threshold for eigenvalue

    Returns
    -------
    True or False
    """
    eig_value = np.linalg.eigvalsh(X)
    return np.all(eig_value > tol)


def _get_single_clf_ko(X, j, method="lasso"):
    """
    Fit a single classifier to predict the j-th variable from all others.

    Args:
        X : input data
        j (int): variable index
        method (str, optional): Classifier used. Defaults to "lasso".

    Returns:
        pred: Predicted values for variable j from all others.
    """
    
    n, p = X.shape
    idc = np.array([i for i in np.arange(0, p) if i != j])

    if method == "lasso":
        lambda_max = np.max(np.abs(np.dot(X[:, idc].T, X[:, j]))) / (2 * (p - 1))
        alpha = (lambda_max / 100)
        clf = Lasso(alpha)
    
    if method == "logreg_cv":
        clf = LogisticRegressionCV(cv=5, max_iter=int(10e4), n_jobs=-1)

    if method == "xgb":
        clf = xgb.XGBRegressor(n_jobs=-1)

    clf.fit(X[:, idc], X[:, j])
    pred = clf.predict(X[:, idc])
    return pred


def _adjust_marginal(v, ref, discrete=False):
    """
    Make v follow the marginal of ref.
    """
    if discrete:
        sorter = np.argsort(v)
        sorter_ = np.argsort(sorter)
        return np.sort(ref)[sorter_]
    
    else:
        G = ECDF(ref)
        F = ECDF(v)

        unif_ = F(v)

        # G_inv = np.argsort(G(ref))
        G_inv = monotone_fn_inverter(G, ref)

        return G_inv(unif_)


def _get_samples_ko(X, pred, j, discrete=False, adjust_marg=True, seed=None):
    """

    Generate a Knockoff for variable j.

    Args:
        X : input data
        pred (array): Predicted Xj using all other variables
        j (int): variable index
        discrete (bool, optional): Indicates discrete or continuous data. Defaults to False.
        adjust_marg (bool, optional): Whether to adjust marginals or not. Defaults to True.
        seed (int, optional): seed. Defaults to None.

    Returns:
        sample: Knockoff for variable j.
    """
    np.random.seed(seed)
    n, p = X.shape

    residuals = X[:, j] - pred
    indices_ = np.arange(residuals.shape[0])
    np.random.shuffle(indices_)

    sample = pred + residuals[indices_]

    if adjust_marg:
        sample = _adjust_marginal(sample, X[:, j], discrete=discrete)

    return sample[np.newaxis].T


def conditional_sequential_gen_ko(X, preds, n_jobs=1, discrete=False, adjust_marg=True, seed=None):
    """
    Generate Knockoffs for all variables in X.

    Args:
        X : input data
        preds (array): Predicted values for all variables
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        discrete (bool, optional): Indicates discrete or continuous data. Defaults to False.
        adjust_marg (bool, optional): Whether to adjust marginals or not. Defaults to True.
        seed (int, optional): seed. Defaults to None.
    Returns:
        samples: Knockoffs for all variables in X.
    """
    
    rng = check_random_state(seed)
    n, p = X.shape

    samples = np.hstack(Parallel(n_jobs=n_jobs)(delayed(
        _get_samples_ko)(X, preds[j], j, discrete=discrete, adjust_marg=adjust_marg) for j in tqdm(range(p))))
    
    return samples


def get_knockoffs_stats(X, labels, draws=100, n_jobs=1,
                        centered=True, shrink=True,
                        offset=1, construct_method='equi',
                        return_alpha=True, alpha_chosen=None,
                        true_covar=None,
                        statistic='lasso_cv', memory=None, gaussian=True,
                        method_ko_gen='lasso', discrete=False,
                        cov_estimator='graph_lasso', 
                        adjust_marginals=False,
                        adjust_marg_NG=True,
                        seed=None):
    """
    Compute Knockoffs p-values as described in [1].

    Parameters
    ----------
    X : array-like of shape (n,p)
        numpy array of size [n,p], containing n observations of p variables
        (hypotheses)
    labels : array-like of shape (n,)
        numpy array of size [n], containing n values in {0, 1}, each of them
        specifying the column indices of the first and the second sample.
    B : int
        number of knockoffs draws to be performed (default=100)

    Returns
    -------
    pval0 : array-like of shape (B, p)
        A numpy array of size [B,p] with each row containing a vector of
        p knockoff p-values
    References
    ----------
    .. [1] Nguyen, T.B., Chevalier, J.A., Thirion, B. and Arlot, S.,
           2020, November. Aggregation of multiple knockoffs.
           In International Conference on Machine Learning
           (pp. 7283-7293). PMLR.
    """

    n, p = X.shape
    
    if centered:
        X = StandardScaler().fit_transform(X)

    rng = check_random_state(seed)
    seed_list = rng.randint(1, np.iinfo(np.int32).max, draws)
    parallel = Parallel(n_jobs)
    
    if gaussian:
        if true_covar is None:
            mu, Sigma = _estimate_distribution(
            X, shrink=shrink, cov_estimator=cov_estimator, n_jobs=n_jobs)
        else:
            mu = np.zeros(X.shape[1])
            Sigma = true_covar
            
        X_tildes = parallel(delayed(gaussian_knockoff_generation)(
            X, mu, Sigma, method=construct_method, memory=memory,
            seed=seed) for seed in seed_list)
        X_tildes = np.array(X_tildes)

        if adjust_marginals:
            for dr in range(draws):
                X_tildes[dr] = np.array(parallel(delayed(_adjust_marginal)(
                    X_tildes[dr][:, j], X[:, j]) for j in range(p))).T
    else:
        preds = np.array(Parallel(n_jobs=n_jobs)(delayed(
            _get_single_clf_ko)(X, j, method_ko_gen) for j in tqdm(range(p))))
        
        X_tildes = [conditional_sequential_gen_ko(X, preds, n_jobs=n_jobs, discrete=discrete, adjust_marg=adjust_marg_NG, seed=seed) for seed in seed_list]
        X_tildes = np.array(X_tildes)   

    mem = check_memory(memory)
    stat_coef_diff_cached = mem.cache(stat_coef_diff,
                                      ignore=['n_jobs', 'joblib_verbose'])



    if alpha_chosen is not None:
        ko_stats = np.array(parallel(delayed(stat_coef_diff_cached)(
            X, X_tildes[i], labels, alpha_chosen=alpha_chosen, method=statistic, return_alpha=False) for i in range(draws)))
        alphas_chosen = [alpha_chosen] * draws
        active_sets = [None] * draws
        return ko_stats, X_tildes, alphas_chosen, active_sets
    else:

        if return_alpha:
            result = parallel(delayed(stat_coef_diff_cached)(
                X, X_tildes[i], labels, method=statistic, return_alpha=True) for i in range(draws))

            ko_stats, alphas_chosen, active_sets = zip(*result)
            ko_stats = np.array(ko_stats)
            alphas_chosen = np.array(alphas_chosen)
            active_sets = np.array(active_sets)

            return ko_stats, X_tildes, alphas_chosen, active_sets

            
        else:
            ko_stats = np.array(parallel(delayed(stat_coef_diff_cached)(
                X, X_tildes[i], labels, method=statistic, return_alpha=False) for i in range(draws)))
            alphas_chosen = [None] * draws
            active_sets = [None] * draws
            return ko_stats, X_tildes, alphas_chosen, active_sets



def stat_coef_diff(X, X_tilde, y, alpha_chosen=None, active_set=None, method='lasso_cv', n_splits=5, n_jobs=1,
                   n_lambdas=10, n_iter=1000, group_reg=1e-3, l1_reg=1e-3,
                   joblib_verbose=0, return_coef=False, return_alpha=False, 
                   solver='liblinear', seed=0):
    """
    Adapted from hidimstat: https://github.com/ja-che/hidimstat
    Calculate test statistic by doing estimation with Cross-validation on
    concatenated design matrix [X X_tilde] to find coefficients [beta
    beta_tilde]. The test statistic is then:

                        W_j =  abs(beta_j) - abs(beta_tilde_j)

    with j = 1, ..., n_features

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        Original design matrix

    X_tilde : 2D ndarray (n_samples, n_features)
        Knockoff design matrix

    y : 1D ndarray (n_samples, )
        Response vector

    loss : str, optional
        if the response vector is continuous, the loss used should be
        'least_square', otherwise
        if the response vector is binary, it should be 'logistic'

    n_splits : int, optional
        number of cross-validation folds

    solver : str, optional
        solver used by sklearn function LogisticRegressionCV

    n_regu : int, optional
        number of regulation used in the regression problem

    return_coef : bool, optional
        return regression coefficient if set to True

    Returns
    -------
    test_score : 1D ndarray (n_features, )
        vector of test statistic

    coef: 1D ndarray (n_features * 2, )
        coefficients of the estimation problem
    """

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])
    lambda_max = np.max(np.dot(X_ko.T, y)) / (2 * n_features)
    lambdas = np.linspace(
        lambda_max*np.exp(-n_lambdas), lambda_max, n_lambdas)

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
#   
    estimator = {
        'lasso_cv': LassoLarsCV(n_jobs=n_jobs, cv=cv),
        'logistic_l1': LogisticRegressionCV(
            penalty='l1', max_iter=int(1e4),
            solver=solver, cv=cv,
            n_jobs=n_jobs, tol=1e-8),
        'logistic_l2': LogisticRegressionCV(
            penalty='l2', max_iter=int(1e4), n_jobs=n_jobs,
            verbose=joblib_verbose, cv=cv, tol=1e-8),
        'enet': ElasticNetCV(cv=cv, max_iter=int(1e4), tol=1e-6,
                             n_jobs=n_jobs, verbose=joblib_verbose),
    }

    try:
        clf = estimator[method]
    except KeyError:
        print('{} is not a valid estimator'.format(method))

    if alpha_chosen is not None and method == 'lasso_cv':
        clf = LassoLars(alpha=alpha_chosen, max_iter=active_set)

    clf.fit(X_ko, y)
    

    try:
        coef = np.ravel(clf.coef_)
    except AttributeError:
        coef = np.ravel(clf.best_estimator_.coef_)  # for GridSearchCV object

    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    if return_coef:
        return test_score, coef

    if alpha_chosen is None:
        if method == 'lasso_cv' and return_alpha:
            alpha_ = clf.alpha_
            # active_set = len(clf.active_)
            active_set = 0
            return test_score, alpha_, active_set
    
    return test_score


def aggregate_list_of_matrices(pi0_raw, gamma=0.5, gamma_min=0.05, adaptive=False, drop_gamma=False, use_hmean=False):
    """
    Provided "draws" null pi-statistics matrices pi0,
    aggregate them to retain a single pi-statistics matrix.
    """
    pi0_raw = np.array(pi0_raw)
    draws, B, p = pi0_raw.shape
    pi0_raw_ = np.reshape(pi0_raw, (B, draws, p))
    if use_hmean:
        pi0 = np.vstack([hmean(pi0_raw_[i], axis=0) for i in range(B)])
        
    else:
        pi0 = np.vstack([quantile_aggregation(
        pi0_raw_[i], gamma=gamma, gamma_min=gamma_min,
        adaptive=adaptive, drop_gamma=drop_gamma) for i in range(B)])

    return pi0


def curve_max_fdp_CT(W, k_vec, v_vec):
    """
    Python version of 'get_FDP_KJI' function from
    https://github.com/Jinzhou-Li/KnockoffSimulFDP
    

    References
    ----------
    [1] Li, J., Maathuis, M. H., & Goeman, J. J. (2022).
     Simultaneous false discovery proportion bounds 
     via knockoffs and closed testing.
     arXiv preprint arXiv:2212.12822.

    """
    if len(np.where(W == 0)[0]) > 0 or len(set(W))!=len(W) or (-np.sort(-W) != W).any(): 
        raise ValueError("The input W might have ties or zeros or not sorted!")

    m = len(k_vec)
    S_list = []
    for i_S in range(1, m+1):
        v = v_vec[i_S-1]
        negatives = W[W < 0]
        negatives = np.sort(negatives) # sort by decreasing module
        
        if len(negatives) < v:
            threshold = min(abs(W)) 
        else:
            threshold = abs(negatives[v - 1])

        S = [i for i in range(len(W)) if W[i]>=threshold]
        S_list.append(S)
    
    p = len(W)
    FDP_bound_vec = [1] * p

    number_pos = len(np.where(W > 0)[0])
    for i in range(number_pos): #careful, since we removed zeros we can't consider bounds on sets including negative W_j because we would have to include zeros...
        R = np.where(W >= W[i])[0]
        if len(R)==0:
            FDP_bound_vec[i] = 0
            continue
        
        FDP_k_temp = []
        for j in range(1, m+1):
            S_temp = S_list[j-1]
            FDP_k_temp.append(min(len(R), k_vec[j-1]-1+len(set(R)-set(S_temp))) / max(1,len(R)))
        FDP_bound_vec[i] = min(FDP_k_temp)
    

    return FDP_bound_vec


def find_largest_region_CT(ko_stats, k, v, tdp):
    """
    Find largest admissible region using 
    the Closed Testing procedure [1].

    References
    ----------
    [1] Li, J., Maathuis, M. H., & Goeman, J. J. (2022).
     Simultaneous false discovery proportion bounds 
     via knockoffs and closed testing.
     arXiv preprint arXiv:2212.12822.

    """
    min_tdp = 1 - np.array(curve_max_fdp_CT(ko_stats, k, v))
    admissible = np.where(min_tdp >= tdp)[0]
 
    if len(admissible) > 0:
        region_size = np.max(admissible)
        w_cutoff = sorted(ko_stats)[region_size - 1]
    else:
        region_size = 0
        w_cutoff = np.infty
    
    return region_size, w_cutoff


def preprocess_W_func_CT(W):
    """
    Python version of function 'preprocess_W_func' 
    from https://github.com/Jinzhou-Li/KnockoffSimulFDP.
    

    References
    ----------
    [1] Li, J., Maathuis, M. H., & Goeman, J. J. (2022).
     Simultaneous false discovery proportion bounds 
     via knockoffs and closed testing.
     arXiv preprint arXiv:2212.12822.

    """
    W_order = np.argsort(-W)
    W_sort = W[W_order]
    
    # delete zeros
    zero_index = np.where(W_sort == 0)[0]
    non_zero_index = np.where(W_sort != 0)[0]
    if len(zero_index) > 0:
        W_sort = W_sort[non_zero_index]
        W_order = W_order[non_zero_index]
    
    # break the tie if there is any without changing the order
    W_sort_abs = np.abs(W_sort)
    for i in range(len(W_sort_abs)):
        temp = W_sort_abs[i]
        if sum(x == temp for x in W_sort_abs) >= 2:
            tie_index = np.where(W_sort_abs == temp)[0]
            first_index = tie_index[0]
            last_index = tie_index[-1]
            print(first_index, last_index)
            if last_index != len(W_sort_abs) - 1:
                max_value = W_sort_abs[last_index] - W_sort_abs[last_index + 1]
                W_sort_abs[tie_index] = [x - (max_value / 2) * (i + 1) for i, x in enumerate(W_sort_abs[tie_index])]
            else:
                max_value = W_sort_abs[first_index - 1] - W_sort_abs[first_index]
                W_sort_abs[tie_index] = [x + (max_value / 2) * (i + 1) for i, x in enumerate(W_sort_abs[tie_index])]
    
    W_sort_new = [np.sign(x) * y for x, y in zip(W_sort, W_sort_abs)]

    return np.array(W_sort_new), W_order



def _empirical_eval(test_score, fdr=0.1, offset=1):

    """
    Compute Knockoff e-values as in [1].

    References
    ----------
    [1] Ren, Z., & Barber, R. F. (2022).
    Derandomized knockoffs: leveraging e-values for false
    discovery rate control. arXiv preprint arXiv:2205.15461.

    """

    evals = []
    n_features = test_score.size

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    ko_thr = _coef_diff_threshold(test_score, fdr=fdr, offset=offset)

    for i in range(n_features):
        if test_score[i] < ko_thr:
            evals.append(0)
        else:
            evals.append(
                n_features /
                (offset + np.sum(test_score <= - ko_thr))
            )

    return np.array(evals)


def perform_inference_given_KO(X_ht, ko_stats, X_tildes, q, beta, draws=2, n_folds=5, n_jobs=1, diag=False):
    """
    Performance inference with Knockoffs already computed.
    """

    p = len(ko_stats[0])
    params = {
        'n_estimators': [1, 3, 5, 10],
        'reg_lambda': [0, 0.1, 1.0, 5.0, 10.0],
        'reg_alpha': [0, 0.1, 1.0],
        'max_depth': [1, 3, 5, 10, 50],
        'colsample_bytree' : [0.3, 1],
        }
    
    pvals = np.array([_empirical_pval(ko_stats[i], 1) for i in range(draws)])

    # select features (w.r.t p-values)
    vanilla_threshold = fdr_threshold(pvals[0], fdr=q, method="bhq")
    selected_ko = np.where(pvals[0] <= vanilla_threshold)[0]
    size_vanilla = len(selected_ko)

    # compute fdr and tdr
    non_zero_index = np.where(beta != 0)[0]
    fdr_, tdr_vanilla_, selected_vanilla = report_fdp_tdp_size(
        pvals[0], size_vanilla, non_zero_index, p
    )
    print(fdr_, tdr_vanilla_)

    if diag:
        clf = GridSearchCV(XGBClassifier(), params, cv=3, verbose=1, n_jobs=n_jobs)

        features = np.concatenate([X_ht, X_tildes[0]], axis=0)  # (2*n_samples, dim)
        labels = np.concatenate(
            [np.array([0] * len(X_ht)), np.array([1] * len(X_tildes[0]))]
        ).ravel()

        features, labels = shuffle(features, labels)

        clf.fit(features, labels)

        scores_1, probas_1 = c2st_scores(X_ht, X_tildes[0], clf_class=XGBClassifier, clf_kwargs=clf.best_params_, n_folds=n_folds)
        df_result1 = pd.DataFrame(scores_1)
        accs = np.array(df_result1['accuracy'])

        print(np.mean(df_result1['accuracy']))
    
        return fdr_, accs
    
    else:
        return fdr_


def report_fdp_tdp_size(p_values, region_size, non_zero_index, n_clusters, use_evalues=False):
    """
    Use region size instead of cutoffs: useful for discrete p-values
    """
    if region_size == 0:
        return 0, 0, None

    if use_evalues:
        selected = np.argsort(p_values)[-region_size:]
    else:
        selected = np.argsort(p_values)[:region_size]
    prediction = np.array([0] * n_clusters)
    prediction[selected] = 1

    non_zero_index_ = np.array([0] * n_clusters)
    non_zero_index_[non_zero_index] = 1

    conf = confusion_matrix(non_zero_index_, prediction)
    tn, fp, fn, tp = conf.ravel()
    if fp + tp == 0:
        fdp = 0
        tdp = 0
    else:
        fdp = fp/(fp+tp)
        tdp = tp/np.sum(non_zero_index_)

    return fdp, tdp, selected
