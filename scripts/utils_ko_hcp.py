"""Contains some ancillary functions for script_hcp.py"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hidimstat.utils import  quantile_aggregation
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from hidimstat.gaussian_knockoff import gaussian_knockoff_generation


def _estimate_distribution(X, shrink=False, cov_estimator='ledoit_wolf'):
    """Copied from old hidimstat repo"""
    from sklearn.covariance import (GraphicalLassoCV, empirical_covariance,
                                ledoit_wolf)
    alphas = [1e-3, 1e-2, 1e-1, 1]

    def _is_posdef(X, tol=1e-14):
        eig_value = np.linalg.eigvalsh(X)
        return np.all(eig_value > tol)

    mu = X.mean(axis=0)
    Sigma = empirical_covariance(X)

    if shrink or not _is_posdef(Sigma):

        if cov_estimator == 'ledoit_wolf':
            Sigma_shrink = ledoit_wolf(X, assume_centered=True)[0]

        elif cov_estimator == 'graph_lasso':
            model = GraphicalLassoCV(alphas=alphas)
            Sigma_shrink = model.fit(X).covariance_

        else:
            raise ValueError('{} is not a valid covariance estimated method'
                             .format(cov_estimator))

        return mu, Sigma_shrink

    return mu, Sigma


def aggregate_list_of_matrices(pval0_raw, gamma=0.5, gamma_min=0.05, 
                               adaptive=False, drop_gamma=False, use_hmean=False,
                               use_arithmetic=False, use_geometric=False):    
    """
    Provided "draws" permuted pvalues matrices pval0, aggregate them to retain a single
    permuted aggregated p-values matrix
    """
    from scipy.stats import hmean, gmean
    pval0_raw = np.array(pval0_raw)
    draws, B, p = pval0_raw.shape
    pval0_raw_ = np.reshape(pval0_raw, (B, draws, p))
    if use_hmean:
        pval0 = np.vstack([hmean(pval0_raw_[i], axis=0) for i in range(B)])

    elif use_arithmetic:
        pval0 = np.vstack([np.mean(pval0_raw_[i], axis=0) for i in range(B)])
    
    elif use_geometric:
        pval0 = np.vstack([gmean(pval0_raw_[i], axis=0) for i in range(B)])
        
    else:
        pval0 = np.vstack([quantile_aggregation(
        pval0_raw_[i], gamma=gamma, gamma_min=gamma_min,
        adaptive=adaptive, drop_gamma=drop_gamma) for i in range(B)])

    return pval0


def get_null_pvals_new(B, p):
    pval0 = np.zeros((B, p))
    for b in range(B):
        signs = (np.random.binomial(1, 0.5, size=p) * 2) - 1
        Z = 0
        for j in range(p):
            if signs[j] < 0:
                pval0[b][j] = 1
                Z += 1
            else:
                pval0[b][j] = (1 + Z) / p
    pval0 = np.sort(pval0, axis=1)
    return pval0


def get_template_new(B, p):
    pval0 = np.zeros((B, p))
    for b in range(B):
        signs = (np.random.binomial(1, 0.5, size=p) * 2) - 1
        Z = 0
        for j in range(p):
            if signs[j] < 0:
                pval0[b][j] = 1
                Z += 1
            else:
                pval0[b][j] = (1 + Z) / p
    pval0 = np.sort(pval0, axis=1)
    pval0 = np.sort(pval0, axis=0)
    return pval0


def find_largest_region_goeman(ko_stats, k, v, tdp):
    min_tdp = 1 - np.array(curve_max_fdp_goeman(ko_stats, k, v))
    admissible = np.where(min_tdp >= tdp)[0]
 
    if len(admissible) > 0:
        region_size = np.max(admissible)
        w_cutoff = sorted(ko_stats)[region_size - 1]
    else:
        region_size = 0
        w_cutoff = np.inf
    
    return region_size, w_cutoff



def preprocess_W_func_goeman(W):
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
                W_sort_abs[tie_index] = [x - (max_value / 2) * (i + 1) 
                                         for i, x in enumerate(W_sort_abs[tie_index])]
            else:
                max_value = W_sort_abs[first_index - 1] - W_sort_abs[first_index]
                W_sort_abs[tie_index] = [x + (max_value / 2) * (i + 1)
                                         for i, x in enumerate(W_sort_abs[tie_index])]
    
    W_sort_new = [np.sign(x) * y for x, y in zip(W_sort, W_sort_abs)]
    cc = 2
    return np.array(W_sort_new), W_order


def get_knockoffs_stats(
        X, labels,
        draws=100,
        n_jobs=1,
        centered=True,
        shrink=True,
        offset=1,
        construct_method='equi',
        return_alpha=True,
        alpha_chosen=None,
        true_covar=None,
        statistic='lasso_cv',
        memory=None,
        gaussian=True,
        method_ko_gen='lasso',
        discrete=False,
        cov_estimator='graph_lasso',
        use_scip=False,
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
    from sklearn.utils.validation import check_memory
    n, p = X.shape
    
    if centered:
        X = StandardScaler().fit_transform(X)

    rng = check_random_state(seed)
    seed_list = rng.randint(1, np.iinfo(np.int32).max, draws)
    parallel = Parallel(n_jobs)
    
    if gaussian:
        if true_covar is None:
            mu, Sigma = _estimate_distribution(
            X, shrink=shrink, cov_estimator=cov_estimator)
        else:
            mu = np.zeros(X.shape[1])
            Sigma = true_covar
            
        X_tildes = parallel(delayed(gaussian_knockoff_generation)(
            X, mu, Sigma, seed=seed) for seed in seed_list)
        X_tildes = np.array([xt[0] for xt in X_tildes])

        if adjust_marginals:
            for dr in range(draws):
                X_tildes[dr] = np.array(parallel(delayed(_adjust_marginal)(
                    X_tildes[dr][:, j], X[:, j]) for j in range(p))).T
    else:

        if use_scip:
            X_ko_scip = scip(X)
            X_tildes = [X_ko_scip for seed in seed_list]
            X_tildes = np.array(X_tildes)
        
        else:
            preds = np.array(Parallel(n_jobs=n_jobs)(delayed(
                _get_single_clf_ko)(X, j, method_ko_gen) for j in tqdm(range(p))))
            

            # for seed in seed_list:
                # lp = LineProfiler()
                # lp_wrapper = lp(conditional_sequential_gen_ko)
                # X_tildes.append(lp_wrapper(X, clfs, n_jobs=n_jobs, seed=seed))
                # lp.print_stats()
            
            X_tildes = [conditional_sequential_gen_ko(
                X,
                preds,
                n_jobs=n_jobs,
                discrete=discrete,
                adjust_marg=adjust_marg_NG,
                seed=seed) for seed in seed_list]
            X_tildes = np.array(X_tildes)   

    mem = check_memory(memory)
    stat_coef_diff_cached = mem.cache(stat_coef_diff,
                                      ignore=['n_jobs', 'joblib_verbose'])



    if alpha_chosen is not None:
        ko_stats = np.array(parallel(delayed(stat_coef_diff_cached)(
            X,
            X_tildes[i],
            labels,
            alpha_chosen=alpha_chosen,
            method=statistic,
            return_alpha=False) for i in range(draws)))
        alphas_chosen = [alpha_chosen] * draws
        active_sets = [None] * draws
        return ko_stats, X_tildes, alphas_chosen, active_sets
    else:

        if return_alpha:
            result = parallel(delayed(stat_coef_diff_cached)(
                X, X_tildes[i], labels, method=statistic, return_alpha=True)
                for i in range(draws))

            ko_stats, alphas_chosen, active_sets = zip(*result)
            ko_stats = np.array(ko_stats)
            alphas_chosen = np.array(alphas_chosen)
            active_sets = np.array(active_sets)

            return ko_stats, X_tildes, alphas_chosen, active_sets
        else:
            ko_stats = np.array(parallel(delayed(stat_coef_diff_cached)(
                X, X_tildes[i], labels, method=statistic) for i in range(draws)))
            alphas_chosen = [None] * draws
            active_sets = [None] * draws
            return ko_stats, X_tildes, alphas_chosen, active_sets


def perform_inference_given_KO(
        X_ht, ko_stats, X_tildes, q, beta, draws=2, n_folds=5, n_jobs=1, diag=False):
    """
    Performance inference with Knockoffs already computed.
    """
    from nilearn.glm import fdr_threshold
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    from sklearn.utils import shuffle
    from valdiags.c2st import c2st_scores

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
    vanilla_threshold = fdr_threshold(pvals[0], alpha=q)
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

        scores_1, probas_1 = c2st_scores(
            X_ht,
            X_tildes[0],
            clf_class=XGBClassifier,
            clf_kwargs=clf.best_params_,
            n_folds=n_folds)
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
    from sklearn.metrics  import confusion_matrix
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


def curve_max_fdp_goeman(W, k_vec, v_vec):

    if len(np.where(W == 0)[0]) > 0 or len(set(W))!=len(W) or (-np.sort(-W) != W).any(): 
        raise ValueError("The input W might have ties or zeros or not sorted!")

    if len(W) == 0:
        return [1]
    m = len(k_vec)
    S_list = []
    for i_S in range(1, m+1):
        v = v_vec[i_S-1]
        negatives = W[W < 0]
        negatives = np.sort(negatives) #sort by decreasing module
        
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


def stat_coef_diff(X, X_tilde, y, method='lasso_cv', n_splits=5, n_jobs=1,
                   n_lambdas=10, n_iter=1000, group_reg=1e-3, l1_reg=1e-3,
                   joblib_verbose=0, return_coef=False, solver='liblinear',
                   seed=0):
    """Calculate test statistic by doing estimation with Cross-validation on
    concatenated design matrix [X X_tilde] to find coefficients [beta
    beta_tilda]. The test statistic is then:

                        W_j =  abs(beta_j) - abs(beta_tilda_j)

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
    from sklearn.linear_model import (LassoCV, LogisticRegressionCV)
    from sklearn.model_selection import KFold

    n_features = X.shape[1]
    X_ko = np.column_stack([X, X_tilde])
    lambda_max = np.max(np.dot(X_ko.T, y)) / (2 * n_features)
    lambdas = np.linspace(
        lambda_max*np.exp(-n_lambdas), lambda_max, n_lambdas)

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    estimator = {
        'lasso_cv': LassoCV(alphas=lambdas, n_jobs=n_jobs,
                            verbose=joblib_verbose, max_iter=int(1e4), cv=cv),
        'logistic_l1': LogisticRegressionCV(
            penalty='l1', max_iter=int(1e4),
            solver=solver, cv=cv,
            n_jobs=n_jobs, tol=1e-8),
        'logistic_l2': LogisticRegressionCV(
            penalty='l2', max_iter=int(1e4), n_jobs=n_jobs,
            verbose=joblib_verbose, cv=cv, tol=1e-8),
    }

    try:
        clf = estimator[method]
    except KeyError:
        print('{} is not a valid estimator'.format(method))

    clf.fit(X_ko, y)

    try:
        coef = np.ravel(clf.coef_)
    except AttributeError:
        coef = np.ravel(clf.best_estimator_.coef_)  # for GridSearchCV object

    test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    if return_coef:
        return test_score, coef

    return test_score


def _empirical_pval(test_score, offset=1):
    """Copied from old hidimstat"""
    pvals = []
    n_features = test_score.size

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    test_score_inv = -test_score
    for i in range(n_features):
        if test_score[i] <= 0:
            pvals.append(1)
        else:
            pvals.append(
                (offset + np.sum(test_score_inv >= test_score[i])) /
                n_features
            )

    return np.array(pvals)
