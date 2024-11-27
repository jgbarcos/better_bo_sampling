import os
import pickle

from run_experiments import ExperimentResults


filepath = os.path.join('./results/forrester/sample_boltzmann_log_ei', 'opt_res_0.pkl')

with open(filepath, 'rb') as fp:         
    data = pickle.load(fp)
    print(data)
    print(ExperimentResults(**data))