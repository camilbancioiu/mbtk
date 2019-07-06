import sys
import os
from pprint import pprint
from pathlib import Path

from string import Template

# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd())
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))


from mbff.experiment.ExperimentDefinition import ExperimentDefinition
from mbff.experiment.ExperimentRun import ExperimentRun
from mbff.experiment.AlgorithmRun import AlgorithmRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
from mbff.algorithms.mb.ipcmb import algorithm_IPCMB

# DEVELOPMENT ONLY:
# Load the source Bayesian network from the BIF file, in order to pass its
# 'conditionally_independent' method to the IPC-MB algorithm as a CI test.
import mbff.utilities.functions as util
bayesian_network = util.read_bif_file(EXPERIMENTS_ROOT / 'bif_repository' / 'survey.bif')
bayesian_network.finalize()

def development_conditionally_independent(X, Y, Z):
    result = bayesian_network.conditionally_independent(X, Y, Z)
    print('CI test {} vs {} given {} is {}'.format(X, Y, Z, result))
    return result



EXPRUN_REPO = EXPERIMENTS_ROOT / 'exprun_repository'

IPCMBDevelopmentDefinition = ExperimentDefinition(EXPRUN_REPO, 'ipcmb_development')
IPCMBDevelopmentDefinition.experiment_run_class = ExperimentRun
IPCMBDevelopmentDefinition.algorithm_run_class = AlgorithmRun
IPCMBDevelopmentDefinition.algorithm_run_datapoint_class = AlgorithmRunDatapoint
IPCMBDevelopmentDefinition.exds_definition = None
IPCMBDevelopmentDefinition.save_algorithm_run_datapoints = True
IPCMBDevelopmentDefinition.algorithm_run_log__stdout = True
IPCMBDevelopmentDefinition.algorithm_run_log__file = True
IPCMBDevelopmentDefinition.algorithm_run_configuration = {
    'label': Template('ipcmb_development_${algorithm_run_index}'),
    'algorithm': algorithm_IPCMB
}
IPCMBDevelopmentDefinition.algorithm_run_parameters = [
    { 
        # The ID of the variable 'R' 
        'target': 3,     

        # Builder that provides a custom CI test, which only relies on the
        # d-separation criterion within a known Bayesian network. No samples
        # are required and no statistics are calculated.
        'ci_test_builder': lambda dm, params: development_conditionally_independent,

        # We did not provide a real experimental dataset (exds_definition is
        # None in the experiment definition above), therefore we instruct
        # IPC-MB to run CI tests and explore the topology of the Bayesian
        # network using the following variables.
        'all_variables': list(range(len(bayesian_network.variable_nodes)))
    },    
]

IPCMBDevelopment = IPCMBDevelopmentDefinition.create_experiment_run()



if __name__ == '__main__':
    # Always erase the experiment run folder, during development.
    IPCMBDevelopmentDefinition.unlock_folder()
    IPCMBDevelopmentDefinition.delete_folder()
    IPCMBDevelopment.run()

