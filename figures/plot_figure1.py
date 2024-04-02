import numpy as np
import matplotlib.pyplot as plt
import os

n, p = 500, 500
rho = 0.7
snr = 7
sparsity = 0.1
draws = 2
n_jobs = 40
seed = 42
q = 0.05
n_runs = 30
n_folds = 5

rng = np.random.default_rng(seed)

results_dir = "../results"
os.chdir(results_dir)

str_rho = f"rho{rho}"
str_snr = f"snr{snr}"

pows = [0.0, 0.5, 0.8, 1.0, 1.25]

fdrs = np.ones((len(pows), 2, n_runs))
accs = np.ones((len(pows), 2, n_folds))

compt = 0
sorter_ = []
for path in os.listdir():

    if str_rho in path and str_snr in path and "smooth_3D" in path and "c2st" not in path:
        bounds = np.load(path, allow_pickle=True)
        accs_ = np.load(path.replace('.npy', '_c2st.npy'), allow_pickle=True)

        fdrs[compt] = bounds
        accs[compt] = accs_

        compt = compt + 1

        param_current = float(path.rsplit("smooth_3D", 3)[0].rsplit('_', 10)[-2].rsplit('sigma', 2)[1])
        print(param_current)
        sorter_.append(param_current)

sorter = np.argsort(sorter_)
param_list = np.sort(sorter_)

fdrs = fdrs[sorter]
accs = accs[sorter]

accs = np.maximum(accs, 1 - accs)

err_mat = np.zeros((len(pows), 2))
for poww in range(len(pows)):
    for method in range(2):
        current = fdrs[poww, method]
        err = current.std() * np.sqrt(1/len(current) +
                                        (current - current.mean())**2 / np.sum((current - current.mean())**2))
        err_mat[poww][method] = np.quantile(err, 0.95)

conf_alpha = 0.05
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
fig.tight_layout(pad=4)

# plot fdp
axs[0].hlines(
    q, pows[0], pows[-1], color="red", linestyle="dashed", label="Target FDR"
)

discrete_error = 1 / (sparsity * p)

axs[0].fill_between(pows, q, q + discrete_error, alpha=0.1, color="red")
mean_values = np.mean(fdrs, axis=2).T
axs[0].plot(pows, mean_values[0], label='Gaussian Knockoffs', color='blue')
axs[0].plot(pows, mean_values[1], label='Non-Gaussian Knockoffs', color='green')

axs[0].fill_between(pows, mean_values[0] - err_mat[:, 0], mean_values[0] + err_mat[:, 0], alpha=0.2, color='blue')
axs[0].fill_between(pows, mean_values[1] - err_mat[:, 1], mean_values[1] + err_mat[:, 1], alpha=0.2, color='green')

axs[0].set_xlabel("Kernel width")
axs[0].set_ylabel("FDR")
axs[0].legend()

mean_values = np.mean(accs, axis=2).T
axs[1].plot(pows, mean_values[0], label='Gaussian Knockoffs', color='blue')
axs[1].plot(pows, mean_values[1], label='Non-Gaussian Knockoffs', color='green')

# Add error bars
std_values_ = np.std(accs, axis=2).T  # Calculate standard deviation
std_values = 1.96 * std_values_ / np.sqrt(n_runs)  # Get the standard deviation error bars

axs[1].fill_between(pows, mean_values[0] - std_values[0], mean_values[0] + std_values[0], alpha=0.2, color='blue')
axs[1].fill_between(pows, mean_values[1] - std_values[1], mean_values[1] + std_values[1], alpha=0.2, color='green')


axs[1].set_xlabel("Kernel width")
axs[1].set_ylabel("C2ST accuracy")

axs[1].hlines(
    0.5,
    pows[0],
    pows[-1],
    color="red",
    linestyle="dashed",
    label=r"$X=\tilde{X}$",
)

axs[0].set_ylim(0.00, 0.5)
axs[1].set_ylim(0.45, 0.85)
axs[1].legend()

plt.suptitle("Knockoffs performance with varying smoothing" + "\n" + rf"$(n={n}, p={p})$")
plt.savefig("../figure_7_reproduced.pdf")
plt.show()
