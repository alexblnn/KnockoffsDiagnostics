"""
This script fetches the HCP collection #4337 on NeuroVault and then processes it to perform knowckoff validation.
"""
import numpy as np
import os
from joblib import Parallel, delayed
from nilearn.datasets import fetch_neurovault_ids
from sklearn.linear_model import (
    LassoCV, LinearRegression, LogisticRegression, LogisticRegressionCV)
import sanssouci as sa
# from preprocess_experiments import preprocess_hcp
from knockoff_aggregation import _empirical_pval
from hidimstat.knockoffs import _empirical_knockoff_eval
from utils import quantile_aggregation
from hidimstat.utils import fdr_threshold
from utils_ko_hcp import (
    aggregate_list_of_matrices,
    get_null_pvals_new,
    get_template_new,
    find_largest_region_goeman,
    preprocess_W_func_goeman,
    get_knockoffs_stats,
    perform_inference_given_KO,
    report_fdp_tdp_size
)
from scipy.stats import hmean

MASK_IMG = "mask_img.nii.gz"
alpha = 0.1
fdr = 0.1
n_jobs = 10
B = 2000
method = 'lasso_cv'
draws = 50
seed = 42
n_clusters = 1000
k_max = int(n_clusters/50)
# n_subjects = None
snr = 5
# sparsity = 0.1
gaussian = True

nv_data = fetch_neurovault_ids(collection_ids=(4337,))


def preprocess_hcp(data_dir='/data/parietal/store/data/HCP900/',
                   n_subjects=150, experiment='RELATIONAL', no_mask=False,
                   mask_type='classic', mask_file=None, n_jobs=1, memory=None):
    """Available experiment: 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR',
    'RELATIONAL', 'SOCIAL', 'WM'

    """
    # from hcp_builder.dataset import fetch_hcp
    from sklearn.utils import Bunch
    from sklearn.preprocessing import StandardScaler
    from nilearn.image import load_img, math_img
    from nilearn.maskers import MultiNiftiMasker
    

    data = fetch_hcp(nv_data=nv_data, n_subjects=n_subjects)
    contrasts = data.contrasts.reset_index()

    if experiment in ('MOTOR_HAND', 'MOTOR_FOOT'):
        experiment_corrected = 'MOTOR'
        TASK = contrasts[contrasts['task'] == experiment_corrected]
    else:
        TASK = contrasts[contrasts['task'] == experiment]

    input_images = TASK.z_map.values
    conditions = TASK.contrast.values

    if experiment == 'GAMBLING':
        condition_mask = np.logical_or(conditions == 'PUNISH',
                                       conditions == 'REWARD')
        y = np.asarray((conditions[condition_mask] == 'PUNISH') * 2 - 1)

    elif experiment == 'RELATIONAL':
        condition_mask = np.logical_or(conditions == 'MATCH',
                                       conditions == 'REL')
        y = np.asarray((conditions[condition_mask] == 'MATCH') * 2 - 1)

    elif experiment == 'EMOTION':
        condition_mask = np.logical_or(conditions == 'FACES',
                                       conditions == 'SHAPES')
        y = np.asarray((conditions[condition_mask] == 'FACES') * 2 - 1)

    elif experiment == 'SOCIAL':
        condition_mask = np.logical_or(conditions == 'RANDOM',
                                       conditions == 'TOM')
        y = np.asarray((conditions[condition_mask] == 'RANDOM') * 2 - 1)

    elif experiment == 'LANGUAGE':
        condition_mask = np.logical_or(conditions == 'MATH',
                                       conditions == 'STORY')
        y = np.asarray((conditions[condition_mask] == 'MATH') * 2 - 1)

    elif experiment == 'MOTOR_HAND':
        # Left hand vs right hand
        condition_mask = np.logical_or(conditions == 'LH',
                                       conditions == 'RH')
        y = np.asarray((conditions[condition_mask] == 'LH') * 2 - 1)

    elif experiment == 'MOTOR_FOOT':
        # Left foot vs right foot
        condition_mask = np.logical_or(conditions == 'LF',
                                       conditions == 'RF')
        y = np.asarray((conditions[condition_mask] == 'LF') * 2 - 1)

    # Working Memory
    elif experiment == 'WM':
        # 2-back vs 0-back
        condition_mask = np.asarray([x[:3] in ['0BK', '2BK'] for x in conditions])
        y = np.asarray([x[:3] == '2BK' for x in conditions[condition_mask]]) * 2 - 1
        
    else:
        raise ValueError('Wrong type of experiment.')

    # groups = TASK.subject.values[condition_mask]

    ######################################################################
    # Masking statistical maps - X, y
    # -------------------------------
    if mask_type == 'classic':
        mask_img = load_img(data.mask)
    elif mask_type == 'specific':
        mask_img = load_img(mask_file)

    if no_mask:
        mask_img = math_img("img > -1", img=mask_img)

    else:

        masker = MultiNiftiMasker(mask_img=mask_img, n_jobs=n_jobs, verbose=1,
                                  memory=memory)
        mask = mask_img.get_fdata().astype(bool)

        X_init = masker.fit_transform(input_images)
        X_sc = StandardScaler()

        if condition_mask is None:
            X = X_sc.fit_transform(np.vstack(X_init))
        else:
            X = X_sc.fit_transform(np.vstack(X_init))[condition_mask]

    return Bunch(X=X, y=y, mask=mask, mask_img=mask_img, masker=masker)


def _make_table(nv_data):
    """Put all data in nv_data in a table with the following information:
    contrast, task, z_map, subject"""
    from pandas import DataFrame
    task = []
    contrast = []
    subject = []
    for x in nv_data.images_meta:
        task.append(x['task'])
        contrast.append(x['contrast_definition'])
        subject.append(x['name'].split('_')[0])
    return DataFrame({
        'task': task,
        'contrast': contrast,
        'subject': subject,
        'z_map': nv_data.images,
    })


def fetch_hcp(nv_data, n_subjects):
    """Extract data from the HCP collection"""
    from sklearn.utils import Bunch
    from os.path import join
    # do a table with all nv_data
    contrasts = _make_table(nv_data)

    # set subjects list
    subjects = np.unique(contrasts.subject.values)[:n_subjects]
    contrasts = contrasts[contrasts.subject.isin(subjects)]
    
    return Bunch(
        contrasts=contrasts,
        mask=MASK_IMG 
    )


parallel = Parallel(n_jobs)
learned_tpl_raw = np.array(
    parallel(delayed(get_template_new)(B, n_clusters) for draw in range(draws)))
pval0_raw = np.array(
    parallel(delayed(get_null_pvals_new)(B, n_clusters) for draw in range(draws)))
pval0_hmean = aggregate_list_of_matrices(
    pval0_raw, gamma=0.3, use_hmean=True)

learned_tpl_hmean_ = aggregate_list_of_matrices(
    learned_tpl_raw,
    gamma=0.3,
    use_hmean=True)
learned_tpl_hmean = np.sort(learned_tpl_hmean_, axis=0)
calibrated_thr_hmean = sa.calibrate_jer(
    alpha,
    learned_tpl_hmean,
    pval0_hmean,
    k_max)

experiments = [
    'MOTOR_HAND',
    'MOTOR_FOOT',
    'GAMBLING',
    'RELATIONAL',
    'EMOTION',
    'SOCIAL',
    'WM',
]  
        

def get_hcp_data(experiment, n_jobs, n_clusters=1000, preloaded=True, n_subjects=150):
    """"""
    from sklearn.feature_extraction import grid_to_graph
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from joblib import dump, load
    from scipy.sparse import coo_matrix, dia_matrix

    list_saves = [
        'X_reduced', 
        'y', 
        'cluster_labels',
        'mask',
        'mask_img',
        'ward_clustering']
    specific_names_ = ['{}_{}'.format(save, experiment) for save in list_saves]

    dir_path = 'data'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    specific_names = [os.path.join(dir_path, pt) for pt in specific_names_]

    if preloaded:

        X_reduced = np.load(specific_names[0] + '.npy')
        y = np.load(specific_names[1] + '.npy')
        cluster_labels = np.load(specific_names[2] + '.npy')
        mask = np.load(specific_names[3] + '.npy')
        mask_img = specific_names[4] + '.nii.gz'
        masker = NiftiMasker(mask_img=mask_img).fit()
        n_samples, n_voxels = np.load(
            os.path.join(abs_path, 'shape_original_{}.npy'.format(experiment)))
        ward = load(os.path.join(abs_path,'ward_clustering_MOTOR_HAND.joblib'))
        shape = mask.shape

        return X_reduced, y, cluster_labels, mask, ward

    else:
        hcp_data = preprocess_hcp(
            n_subjects=n_subjects, n_jobs=n_jobs, experiment=experiment)
        X = hcp_data.X
        y = hcp_data.y
        mask = hcp_data.mask
        masker = hcp_data.masker
        mask_img = hcp_data.mask_img
        n_samples, n_voxels = X.shape
        shape = mask.shape


        connectivity = grid_to_graph(
            n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)

        # ward = load(os.path.join(dir_path,'ward_clustering_MOTOR_HAND.joblib'))
        n_clusters = 500
        ward_path = os.path.join(dir_path,'ward_clustering_MOTOR_HAND.joblib')
        if experiment == "MOTOR_HAND":
            ward = AgglomerativeClustering(
                n_clusters=n_clusters, linkage='ward', connectivity=connectivity
            ).fit(X[:10, ].T)
            # save the model
            dump(ward, ward_path)
        else:
            # load the model
            ward = load(ward_path)
        
        # compress the data
        n_vertices = X.shape[1]
        edges = (np.arange(n_vertices), ward.labels_)
        incidence = coo_matrix((np.ones(n_vertices), edges), shape=(n_vertices, n_clusters))
        weight = dia_matrix((1. / incidence.sum(0), 0), shape=(n_clusters, n_clusters))
        incidence = incidence.dot(weight)
        X_reduced = incidence.T.dot(X.T).T

        cluster_labels = ward.labels_
        X_reduced = StandardScaler().fit_transform(X_reduced)
        print('here')
        np.save(os.path.join(dir_path, 'shape_original_{}'.format(experiment)),
                (n_samples, n_voxels))
        np.save(specific_names[0], X_reduced)
        np.save(specific_names[1], y)
        np.save(specific_names[2], cluster_labels)
        np.save(specific_names[3], mask)
        mask_img.to_filename('{}.nii.gz'.format(specific_names[4]))
    
        return X_reduced, y, cluster_labels, mask, None


def perform_inference(experiment_train, experiment_test, n_clusters, n_jobs, alpha, fdr, snr, draws):
    """
    For a pair of HCP experiments, generate semi-simulated data and perform inference
    using 5 Knockoffs-based methods.
    
    """

    k_opti = [5, 8, 15, 21, 31, 45, 63, 77, 84, 103, 116]
    v_opti = [1, 2, 5, 8, 13, 18, 25, 32, 41, 50, 61]
    
    X_reduced_train, y_train, cluster_labels_train, mask_train, ward_train = get_hcp_data(
        experiment_train, n_jobs, n_clusters=n_clusters, preloaded=False)

    lambda_max = np.max(np.dot(X_reduced_train.T, y_train)) / (2 * X_reduced_train.shape[1])
    print(lambda_max)
    clf = LogisticRegression(C=1/(lambda_max*0.1), penalty='l1', max_iter=int(1e4), n_jobs=n_jobs, solver='liblinear')
    clf.fit(X_reduced_train, y_train)

    beta_train = clf.coef_ # generate ground truth
    beta_train = np.ravel(beta_train)

    tol = np.max(np.abs(beta_train)) * 1e-2
    print(tol)

    non_zero_index = np.where(np.abs(beta_train) > tol)[0]
    zero_index = np.where(np.abs(beta_train) <= tol)[0]

    beta_train[zero_index] = 0

    print(len(non_zero_index))
    print(experiment_train, experiment_test)
    X_reduced_test, _, cluster_labels_test, mask_test, _ = get_hcp_data(
        experiment_test, n_jobs, n_clusters=n_clusters, preloaded=False)
    prod_temp = np.dot(X_reduced_test, beta_train)
    eps = np.random.normal(size=X_reduced_test.shape[0])
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))

    y_test = prod_temp + noise_mag * eps # generate new y

    ko_stats, X_tildes, alphas_chosen, active_sets = get_knockoffs_stats(
        X_reduced_test,
        y_test,
        draws=draws,
        n_jobs=n_jobs,
        return_alpha=False,
        true_covar=None,
        statistic=method,
        gaussian=gaussian,
        seed=seed)

    fdp_, acc_ = perform_inference_given_KO(
            X_reduced_test, ko_stats, X_tildes, fdr, beta_train, draws=draws, n_jobs=n_jobs, diag=True
        )
    
    print(acc_)

    pvals = np.array([_empirical_pval(ko_stats[i], 1)
                for i in range(draws)])
    evals = np.array([ _empirical_knockoff_eval(ko_stats[i], ko_threshold=(fdr / 2))
            for i in range(draws)])

    p_values_cal = quantile_aggregation(pvals, gamma=0.3)
    p_values_hmean = hmean(pvals, axis=0)
    e_values = np.mean(evals, axis=0)
    pvals_vanilla = pvals[0]
    W_goeman = preprocess_W_func_goeman(ko_stats[0])[0]

    size_hmean = sa.find_largest_region(p_values_hmean, calibrated_thr_hmean, 1 - fdr)
    fdp_hmean_, tdp_hmean_, selected_hmean = report_fdp_tdp_size(p_values_hmean, size_hmean, non_zero_index, n_clusters)
    print(fdp_hmean_, tdp_hmean_)

    ebh_threshold = fdr_threshold(e_values, fdr=fdr, method='ebh')
    size_ebh = len(np.where(e_values >= ebh_threshold)[0])
    fdp_ebh_, tdp_ebh_, selected_ebh = report_fdp_tdp_size(e_values, size_ebh, non_zero_index, n_clusters, use_evalues=True)
    print(fdp_ebh_, tdp_ebh_)

    ako_threshold = fdr_threshold(p_values_cal, fdr=fdr, method='bhq')
    size_ako = len(np.where(p_values_cal <= ako_threshold)[0])
    fdp_ako_, tdp_ako_, selected_ako = report_fdp_tdp_size(p_values_cal, size_ako, non_zero_index, n_clusters)
    print(fdp_ako_, tdp_ako_)

    # W_goeman = ko_stats[0]
    size_goeman, cutoff_goeman = find_largest_region_goeman(W_goeman, k_opti, v_opti, 1 - fdr)
    fdp_goeman_, tdp_goeman_, selected_goeman = report_fdp_tdp_size(np.array(ko_stats[0]), size_goeman, non_zero_index, n_clusters, use_evalues=True)
    print(fdp_goeman_, tdp_goeman_)

    vanilla_threshold = fdr_threshold(pvals_vanilla, fdr=fdr, method='bhq')
    size_vanilla = len(np.where(pvals_vanilla <= vanilla_threshold)[0])
    fdp_vanilla_, tdp_vanilla_, selected_vanilla = report_fdp_tdp_size(pvals_vanilla, size_vanilla, non_zero_index, n_clusters)
    print(fdp_vanilla_, tdp_vanilla_)

    return [
        fdp_hmean_,
        fdp_goeman_,
        fdp_ebh_,
        fdp_ako_,
        fdp_vanilla_,
        tdp_hmean_,
        tdp_goeman_,
        tdp_ebh_,
        tdp_ako_,
        tdp_vanilla_
    ], [size_hmean, size_goeman, size_ebh, size_ako, size_vanilla]



n_methods = 5

import itertools
all_pairs1 = list(itertools.combinations(experiments, 2))
all_pairs2 = [t[::-1] for t in all_pairs1]
all_pairs = all_pairs1 + all_pairs2

print(len(all_pairs))

nb_expes = len(all_pairs)
# bounds_res = np.zeros((nb_expes, n_methods * 2))
# sizes_res = np.zeros((nb_expes, n_methods))

for id_exp in range(nb_expes):
    # print(all_pairs[id_exp])
    experiment_train, experiment_test = all_pairs[id_exp]
    # experiment_train, experiment_test = 'MOTOR_HAND', 'RELATIONAL'
    # experiment_train, experiment_test = 'WM', 'WM'
    # experiment_train, experiment_test = 'MOTOR_FOOT', 'WM'
    bounds, sizes = perform_inference(experiment_train, experiment_test,
                            n_clusters, n_jobs, alpha, 
                            fdr, snr, draws)
    print(bounds)
    print(sizes)
    """
    np.save(f'../../figures/non_gaussian_ko/HCP_semisim/{experiment_train}/{experiment_test}/gaussian{gaussian}_fdr{fdr}_snr{snr}_regression.npy', bounds)
    np.save(f'../../figures/non_gaussian_ko/HCP_semisim/{experiment_train}/{experiment_test}/gaussian{gaussian}_fdr{fdr}_snr{snr}_sizes_regression.npy', sizes)
    """


