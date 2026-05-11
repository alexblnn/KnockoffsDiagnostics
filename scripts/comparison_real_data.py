"""
This script fetches the HCP collection #4337 on NeuroVault 
and then processes with different knockoff generation approaches.

Todo: 
* remove unused functions
* remove unused imports

"""

import numpy as np
import os
from joblib import Parallel, delayed, Memory
from nilearn.datasets import fetch_neurovault_ids
from sklearn.linear_model import (
    LassoCV, LinearRegression, LogisticRegression, LogisticRegressionCV)
from hidimstat.statistical_tools.multiple_testing import fdr_threshold
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
from hidimstat import ModelXKnockoff
from hidimstat.samplers import GaussianKnockoffs
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from nilearn.image import new_img_like
from nilearn.plotting import plot_roi, show

mem = Memory(location='/home/bthirion/tmp/', verbose=0)   


MASK_IMG = "data/mask_img.nii.gz"
alpha = 0.1
fdr = 0.2
n_jobs = 10
B = 2000
method = 'lasso_cv'
draws = 50
seed = 42
n_clusters = 1000
k_max = int(n_clusters / 50)
# n_subjects = None
snr = 5
# sparsity = 0.1
gaussian = True
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

nv_data = mem.cache(fetch_neurovault_ids)(collection_ids=(4337,))


def preprocess_hcp(data_dir='/data/parietal/store/data/HCP900/',
                   n_subjects=150, experiment='RELATIONAL', no_mask=False,
                   mask_type='classic', mask_file=None, n_jobs=1, memory=None):
    """Available experiment: 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR',
    'RELATIONAL', 'SOCIAL', 'WM'

    """
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

   
    hcp_data = mem.cache(preprocess_hcp)(
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
    
    np.save(os.path.join(dir_path, 'shape_original_{}'.format(experiment)),
            (n_samples, n_voxels))
    np.save(specific_names[0], X_reduced)
    np.save(specific_names[1], y)
    np.save(specific_names[2], cluster_labels)
    np.save(specific_names[3], mask)
    mask_img.to_filename('{}.nii.gz'.format(specific_names[4]))
    
    return X_reduced, y, cluster_labels, mask_img, ward


def make_image(selection, mask_img, ward):
        """Make an image of the selected clusters"""
        binary = np.zeros(ward.n_clusters)
        binary[selection] = 1
        voxel_select = binary[ward.labels_]
        mask =  mask_img.get_fdata()
        data_select = mask.copy()
        data_select[mask > 0] = voxel_select
        return new_img_like(mask_img, data_select)


def perform_inference(experiment_train, n_clusters, n_jobs, alpha, fdr, snr, draws):
    """
    For a pair of HCP experiments, generate semi-simulated data and perform inference
    using 5 Knockoffs-based methods.
    """
    (X_reduced_train, 
     y_train, 
     cluster_labels_train, 
     mask_train,
     ward_train) = mem.cache(get_hcp_data)(
        experiment_train, n_jobs, n_clusters=n_clusters, preloaded=False)
    
    # This is just to pick a sensible value for beta_train
    lambda_max = np.max(np.abs(X_reduced_train.T.dot(y_train)))
    clf = LogisticRegression(
        C=1/(lambda_max*0.1),
        penalty='l1',
        max_iter=int(1e4),
        n_jobs=n_jobs,
        solver='liblinear')
    clf.fit(X_reduced_train, y_train)
    beta_train = np.ravel(clf.coef_)

    # -- LedoitWolf knockoff generation, current hidimstat implementation
    """
    model_x_knockoff = ModelXKnockoff(
        ko_generator=GaussianKnockoffs(
            cov_estimator=LedoitWolf(assume_centered=True), tol=1e-15
        ),
        estimator=LogisticRegressionCV(
            solver="liblinear",
            penalty="l1",
            Cs=np.logspace(-3, 3, 10),
            random_state=0,
            tol=1e-3,
            max_iter=1000,
        ),
        random_state=0,
        preconfigure_lasso_path=False,
    )
    importance = model_x_knockoff.fit_importance(
        X_reduced_train,
        y_train,
    )
    selected = model_x_knockoff.fdr_selection(fdr=0.1)
    print(np.where(selected))
    """

    # Use of the old API
    ko_stats, X_tildes, alphas_chosen, active_sets = get_knockoffs_stats(
        X_reduced_train,
        y_train,
        draws=draws,
        n_jobs=n_jobs,
        return_alpha=False,
        true_covar=None,
        statistic=method,
        gaussian=True,
        use_scip=False,
        seed=seed)
    assert len(ko_stats) == draws
    fdp_gauss, acc_gauss, selection_gauss = perform_inference_given_KO(
        X_reduced_train,
        ko_stats,
        X_tildes,
        fdr,
        beta_train,
        n_jobs=n_jobs,
        diagnosis=True
    )
    

    # -- GraphicalLassoCV knockoff generation
    """
    model_x_knockoff = ModelXKnockoff(
        ko_generator=GaussianKnockoffs(
            cov_estimator=GraphicalLassoCV(assume_centered=True), tol=1e-15
        ),
        estimator=LogisticRegressionCV(
            solver="liblinear",
            penalty="l1",
            Cs=np.logspace(-3, 3, 10),
            random_state=0,
            tol=1e-3,
            max_iter=1000,
        ),
        random_state=0,
        preconfigure_lasso_path=False,
    )
    importance = model_x_knockoff.fit_importance(
        X_reduced_train,
        y_train,
    )
    selected = model_x_knockoff.fdr_selection(fdr=0.1)
    print(np.where(selected))
    """
    # parallel knockoff generation
    ko_stats, X_tildes, alphas_chosen, active_sets = get_knockoffs_stats(
        X_reduced_train,
        y_train,
        draws=draws,
        n_jobs=n_jobs,
        return_alpha=False,
        true_covar=None,
        statistic=method,
        gaussian=False,
        use_scip=False,
        seed=seed)
    assert len(ko_stats) == draws
    fdp_scip, acc_scip, selection_scip = perform_inference_given_KO(
        X_reduced_train,
        ko_stats,
        X_tildes,
        fdr,
        beta_train,
        n_jobs=n_jobs,
        diagnosis=True
    )

    # -- scip knockoff generation
    ko_stats, X_tildes, alphas_chosen, active_sets = get_knockoffs_stats(
        X_reduced_train,
        y_train,
        draws=draws,
        n_jobs=n_jobs,
        return_alpha=False,
        true_covar=None,
        statistic=method,
        gaussian=False,
        use_scip=True,
        seed=seed)
    fdp_parallel, acc_parallel, selection_parallel = perform_inference_given_KO(
        X_reduced_train,
        ko_stats,
        X_tildes,
        fdr,
        beta_train,
        n_jobs=n_jobs,
        diagnosis=True
    )
    img_gauss = make_image(selection_gauss, mask_train, ward_train)
    img_scip = make_image(selection_scip, mask_train, ward_train)
    img_parallel = make_image(selection_parallel, mask_train, ward_train)
    img_gauss.to_filename(results_dir + '/{}_gaussian_ko_selection.nii.gz'.format(experiment_train))
    img_scip.to_filename(results_dir + '/{}_scip_ko_selection.nii.gz'.format(experiment_train))
    img_parallel.to_filename(results_dir + '/{}_parallel_ko_selection.nii.gz'.format(experiment_train))


experiments = [
    'MOTOR_HAND',
    'MOTOR_FOOT',
    'GAMBLING',
    'RELATIONAL',
    'EMOTION',
    'SOCIAL',
    'WM',
]  

import itertools

n_experiments = len(experiments)

for id_exp in range(n_experiments):
    experiment_train = experiments[id_exp]
    fdr = 0.2
    # fdr = .1 returns empty results for all methods, which is not very interesting to compare

    perform_inference(
        experiment_train,
        n_clusters,
        n_jobs,
        alpha, 
        fdr,
        snr,
        draws)
    
    