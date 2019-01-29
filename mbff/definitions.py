from experimental_dataset import *
from experimental_pipeline import *
import collections

from definition_utilities import *
import definitions_oevexp
from definitions_oevexp import *

# Ensure industry list files and exds list files are stored in industry_list/ and exds_list/.
util.autoListFolders.extend(['industry_list', 'exds_list'])

## Experimental dataset definitions

ExperimentalDatasets = {}

add_to_definitions(ExperimentalDatasets, [
    ExperimentalDatasetDefinition('tiny', 'I3302022', 0.30, (0.1, 0.9)),
    ExperimentalDatasetDefinition('sampling_tester', 'I3302022', 0.30, (0.1, 0.9)),
    ExperimentalDatasetDefinition('industry_I65100', 'I65100', 0.30, (0.1, 0.9)),
    ])



# industry_list_files = ['tier1', 'tier2']
# add_to_definitions(ExperimentalDatasets,
    # create_exds_definitions_from_industry_list_files(industry_list_files)
    # )

# ExDs tags
#ExperimentalDatasets['industry_I3302019'].tags.append('gamma_test')     # Selected for preliminary gamma tests
ExperimentalDatasets['industry_I65100'].tags.append('c2fsa')


## Experiment definitions

Experiments = {}

default_parameters = collections.OrderedDict()
default_parameters['algorithm'] = ['NL', 'RND', 'KS', 'IG']
default_parameters['Q'] = range(1, 169 + 1)     # ExDs 'tiny' has 169 features.
default_parameters['target'] = [0]
default_parameters['K'] = [5]

dummy = ExperimentDefinition(
    'dummy',
    ExperimentalDatasets['tiny'],
    default_parameters,
    {
        'ks_iteration_cache' : FULL_USE,
        'ks_feature_db' : FULL_USE,
        'ks_parallelism': 0,
        'run_parallelism': 0,
    })

sampling_tester_params = collections.OrderedDict()
sampling_tester_params['algorithm'] = ['NL', 'RND', 'KS', 'IG']
sampling_tester_params['Q'] = range(1, 169 + 1)     # ExDs 'tiny' has 169 features.
sampling_tester_params['target'] = [0, 1]
sampling_tester_params['K'] = [1, 2, 3, 4]

sampling_tester = ExperimentDefinition(
    'sampling_tester',
    ExperimentalDatasets['sampling_tester'],
    sampling_tester_params,
    {
        'ks_iteration_cache' : FULL_USE,
        'ks_feature_db' : FULL_USE,
        'ks_parallelism': 0,
        'run_parallelism': 0,
    })

## An experiment analyzing the behavior of KS with In-iteration parallelism
## enabled.
ks_ip_analysis_params = collections.OrderedDict()
ks_ip_analysis_params['algorithm'] = ['KS']
ks_ip_analysis_params['Q'] = [1]
ks_ip_analysis_params['target'] = [0]
ks_ip_analysis_params['K'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ks_ip_analysis_ip_off = ExperimentDefinition(
        'ks_ip_analysis_ip_off',
        ExperimentalDatasets['tiny'],
        ks_ip_analysis_params,
        {
            'run_parallelism' : 0,
            'algrun_saving' : 'save_full_and_samples',
            'algrun_stdout' : 'sys.stdout',
            'ks_iteration_cache' : DISABLED,
            'ks_feature_db' : FULL_USE,
            'ks_parallelism' : 0,
            'ks_debug' : True
        })

ks_ip_analysis_ip_on = ExperimentDefinition(
        'ks_ip_analysis_ip_on',
        ExperimentalDatasets['tiny'],
        ks_ip_analysis_params,
        {
            'run_parallelism' : 0,
            'algrun_saving' : 'save_full_and_samples',
            'algrun_stdout' : 'sys.stdout',
            'ks_iteration_cache' : DISABLED,
            'ks_feature_db' : COMPARE_ONLY,
            'ks_parallelism' : 1,
            'ks_debug' : True,
            'plots': {
                'iteration_time' : {
                    'title' : '',
                    'legend' : ['Unoptimized', 'with IP, {} processes'.format(2)],
                    'axisY' : [0, 1]
                }
            }
        })
add_to_definitions(Experiments, [ks_ip_analysis_ip_on])
add_to_definitions(Experiments, [ks_ip_analysis_ip_off])

## Define a simple experiment which selects features using IG, for the given
## values of Q.
def define_igt_experiment(dataset_name, target, maxQ, name='igt'):
    parameters = collections.OrderedDict()
    parameters['target'] = [target]
    parameters['algorithm'] = ['IG']
    parameters['Q'] = list(range(1, maxQ + 1))
    return ExperimentDefinition(
            '{}_{}'.format(name, 0),
            ExperimentalDatasets[dataset_name],
            parameters,
            {
                'run_parallelism': 0
            })

## Define a simple experiment which selects features using KS, for the given 
## values of Q and K.
def define_ks_experiment(dataset_name, target, maxQ, K_values, name='ks'):
    parameters = collections.OrderedDict()
    parameters['target'] = [target]
    parameters['algorithm'] = ['KS']
    parameters['K'] = list(K_values)
    parameters['Q'] = list(range(1, maxQ + 1))
    return ExperimentDefinition(
            '{}_{}'.format(name, 0),
            ExperimentalDatasets[dataset_name],
            parameters,
            {
                'ks_iteration_cache': FULL_USE,
                'ks_feature_db': FULL_USE,
                'ks_parallelism': 0,
                'run_parallelism': 0
            })



## Add experiment definitions for the Optimization Evaluation Experiment (OEvExp).
definitions_oevexp.dataset = ExperimentalDatasets['industry_I65100']
add_to_definitions(Experiments, [
    dummy,
    sampling_tester,
    define_oevexp_reference_experiment(0),
    define_oevexp_reference_experiment(1),
    define_oevexp_reference_experiment(2),
    define_oevexp_reference_experiment(3),
    define_oevexp_reference_experiment(4),
    define_oevexp_reference_experiment(5),
    define_oevexp_reference_experiment(6),
    define_oevexp_ks_parallelism_validation(0, 7),
    define_oevexp_ks_parallelism_validation(1, 7),
    define_oevexp_ks_parallelism_validation(2, 7),
    define_oevexp_ks_parallelism_validation(3, 7),
    define_oevexp_ks_parallelism_validation(4, 7),
    define_oevexp_ks_parallelism_validation(5, 7),
    define_oevexp_ks_parallelism_validation(6, 7),
    define_oevexp_ksic_validation(0),
    define_oevexp_ksic_validation(1),
    define_oevexp_ksic_validation(2),
    define_oevexp_ksic_validation(3),
    define_oevexp_ksic_validation(4),
    define_oevexp_ksic_validation(5),
    define_oevexp_ksic_validation(6),
    define_oevexp_ksfdb_validation(0),
    define_oevexp_ksfdb_validation(1),
    define_oevexp_ksfdb_validation(2),
    define_oevexp_ksfdb_validation(3),
    define_oevexp_ksfdb_validation(4),
    define_oevexp_ksfdb_validation(5),
    define_oevexp_ksfdb_validation(6),
    define_oevexp_gd_validation(0),
    define_oevexp_gd_validation(1),
    define_oevexp_gd_validation(2),
    define_oevexp_gd_validation(3),
    define_oevexp_gd_validation(4),
    define_oevexp_gd_validation(5),
    define_oevexp_gd_validation(6),
    define_igt_experiment('industry_I65100', 0, 2645),
    define_ks_experiment('industry_I65100', 0, 2645, range(16, 21))
    ])

