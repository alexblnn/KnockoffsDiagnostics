import os

sigma_list = [0.0, 0.5, 0.8, 1.0, 1.25]

path_to_script = ''

for sigma in sigma_list:
    os.system(path_to_script + ' run_simulation_single_sigma.py ' + str(sigma))
