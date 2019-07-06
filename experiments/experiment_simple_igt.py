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

SurveyDatasetDefinition = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_survey_1e4')
SurveyDatasetDefinition.exds_class = ExperimentalDataset
SurveyDatasetDefinition.source = SampledBayesianNetworkDatasetSource
SurveyDatasetDefinition.source_configuration = {
        'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'survey.bif',
        'sample_count': 10000,
        'random_seed': 128,
        'objectives': ['T']
        }



from mbff.experiment.ExperimentDefinition import ExperimentDefinition
from mbff.experiment.ExperimentRun import ExperimentRun
from mbff.experiment.AlgorithmRun import AlgorithmRun
from mbff.experiment.AlgorithmRunDatapoint import AlgorithmRunDatapoint
from mbff.algorithms.basic.IGt import algorithm_IGt

EXPRUN_REPO = EXPERIMENTS_ROOT / 'exprun_repository'

SimpleExperimentDefinition = ExperimentDefinition(EXPRUN_REPO, 'simple_igt')
SimpleExperimentDefinition.experiment_run_class = ExperimentRun
SimpleExperimentDefinition.algorithm_run_class = AlgorithmRun
SimpleExperimentDefinition.algorithm_run_datapoint_class = AlgorithmRunDatapoint
SimpleExperimentDefinition.exds_definition = SurveyDatasetDefinition
SimpleExperimentDefinition.save_algorithm_run_datapoints = True
SimpleExperimentDefinition.algorithm_run_log__stdout = True
SimpleExperimentDefinition.algorithm_run_log__file = True
SimpleExperimentDefinition.algorithm_run_configuration = {
        'label': Template('simple_igt_${algorithm_run_index}__Q=${Q}'),
        'algorithm': algorithm_IGt
        }
SimpleExperimentDefinition.algorithm_run_parameters = [
        { 'Q': 1, 'objective_index': 0 },
        { 'Q': 2, 'objective_index': 0 },
        { 'Q': 3, 'objective_index': 0 },
        { 'Q': 4, 'objective_index': 0 }
        ]

SimpleExperiment = SimpleExperimentDefinition.create_experiment_run()



if __name__ == '__main__':
    SimpleExperiment.run()
