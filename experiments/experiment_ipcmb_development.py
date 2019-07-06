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



from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ExperimentalDataset import ExperimentalDataset
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource

EXDS_REPO = EXPERIMENTS_ROOT / 'exds_repository'

SurveyDatasetDefinition = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_survey_1e4_fullX')
SurveyDatasetDefinition.exds_class = ExperimentalDataset
SurveyDatasetDefinition.source = SampledBayesianNetworkDatasetSource
SurveyDatasetDefinition.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'survey.bif',
    'sample_count': 10000,
    'random_seed': 128,
}



from mbff.experiment.ExperimentDefinition import ExperimentDefinition
from mbff.experiment.ExperimentRun import ExperimentRun
from mbff.experiment.AlgorithmRun import AlgorithmRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
from mbff.algorithms.mb.ipcmb import algorithm_IPCMB

# DEVELOPMENT ONLY:
# Load the source Bayesian network from the BIF file, in order to pass its
# 'conditionally_independent' method to the IPC-MB algorithm as a CI test.
import mbff.utilities.functions as util
bayesian_network = util.read_bif_file(SurveyDatasetDefinition.source_configuration['sourcepath'])
bayesian_network.finalize()

def development_conditionally_independent(X, Y, Z):
    result = bayesian_network.conditionally_independent(X, Y, Z)
    print('CI test {} vs {} given {} is {}'.format(X, Y, Z, result))
    return result


print(str(development_conditionally_independent))

EXPRUN_REPO = EXPERIMENTS_ROOT / 'exprun_repository'

IPCMBDevelopmentDefinition = ExperimentDefinition(EXPRUN_REPO, 'ipcmb_development')
IPCMBDevelopmentDefinition.experiment_run_class = ExperimentRun
IPCMBDevelopmentDefinition.algorithm_run_class = AlgorithmRun
IPCMBDevelopmentDefinition.algorithm_run_datapoint_class = AlgorithmRunDatapoint
IPCMBDevelopmentDefinition.exds_definition = SurveyDatasetDefinition
IPCMBDevelopmentDefinition.save_algorithm_run_datapoints = True
IPCMBDevelopmentDefinition.algorithm_run_log__stdout = True
IPCMBDevelopmentDefinition.algorithm_run_log__file = True
IPCMBDevelopmentDefinition.algorithm_run_configuration = {
    'label': Template('ipcmb_development_${algorithm_run_index}'),
    'algorithm': algorithm_IPCMB
}
IPCMBDevelopmentDefinition.algorithm_run_parameters = [
    { 
        'target': 3,     # The ID of the variable 'R' 
        'ci_test_builder': lambda dm, params: development_conditionally_independent
    },    
]

IPCMBDevelopment = IPCMBDevelopmentDefinition.create_experiment_run()



if __name__ == '__main__':
    # Always erase the experiment run folder, during development.
    IPCMBDevelopmentDefinition.unlock_folder()
    IPCMBDevelopmentDefinition.delete_folder()
    IPCMBDevelopment.run()

