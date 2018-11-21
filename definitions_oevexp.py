import collections

from definition_utilities import *


dataset = None

# Optimization Evaluation Experiments (OEvExp)

oevexp_reference_parameters = collections.OrderedDict()
oevexp_reference_parameters['algorithm'] = ['KS']
oevexp_reference_parameters['Q'] = [1]
oevexp_reference_parameters['target'] = [0]
oevexp_reference_parameters['K'] = [5]

def define_oevexp_reference_experiment(target):
    global dataset
    parameters = collections.OrderedDict()
    parameters['algorithm'] = ['KS']
    parameters['Q'] = [1]
    parameters['target'] = [target]
    parameters['K'] = [5]

    oevexp = ExperimentDefinition(
        'oevexp_reference_{}'.format(target), 
        dataset,
        parameters,
        {
            'ks_iteration_cache': DISABLED,
            'ks_feature_db': FULL_USE,
            'ks_parallelism': 0,
            'run_parallelism': 0,
        })
    return oevexp


def define_oevexp_ks_parallelism_validation(target, ks_parallelism):
    global dataset
    parameters = collections.OrderedDict()
    parameters['algorithm'] = ['KS']
    parameters['Q'] = [1]
    parameters['target'] = [target]
    parameters['K'] = [5]

    oevexp = ExperimentDefinition(
        'oevexp_ks_parallelism_{}_P{}'.format(target, ks_parallelism), 
        dataset,
        parameters,
        {
            'ks_iteration_cache': DISABLED,
            'ks_feature_db': COMPARE_ONLY,
            'ks_parallelism': ks_parallelism,
            'run_parallelism': 0,
            'plots': {
                'iteration_time' : {
                    'title' : '',
                    'legend' : ['Unoptimized', 'with IP, {} processes'.format(ks_parallelism)]
                }
            }
        })
    return oevexp


def define_oevexp_ksfdb_validation(target):
    global dataset
    parameters = collections.OrderedDict()
    parameters['algorithm'] = ['KS']
    parameters['Q'] = [1]
    parameters['target'] = [target]
    parameters['K'] = [5]

    oevexp = ExperimentDefinition(
        'oevexp_ksfdb_validation_{}'.format(target), 
        dataset,
        parameters,
        {
            'ks_iteration_cache': DISABLED,
            'ks_feature_db': COMPARE_ONLY,
            'ks_parallelism': 0,
            'run_parallelism': 0,
        })
    return oevexp

def define_oevexp_ksic_validation(target):
    global dataset
    parameters = collections.OrderedDict()
    parameters['algorithm'] = ['KS']
    parameters['Q'] = [1]
    parameters['target'] = [target]
    parameters['K'] = [5]

    oevexp = ExperimentDefinition(
        'oevexp_ksic_validation_{}'.format(target), 
        dataset,
        parameters,
        {
            'ks_iteration_cache': FULL_USE,
            'ks_feature_db': COMPARE_ONLY,
            'ks_parallelism': 0,
            'run_parallelism': 0,
            'plots': {
                'iteration_time' : {
                    'title' : '',
                    'legend' : ['Unoptimized', 'with IC']
                }
            }
        })
    return oevexp

def define_oevexp_gd_validation(target):
    global dataset
    parameters = collections.OrderedDict()
    parameters['algorithm'] = ['KS']
    parameters['Q'] = [1]
    parameters['target'] = [target]
    parameters['K'] = [5]

    oevexp = ExperimentDefinition(
        'oevexp_gd_validation_{}'.format(target), 
        dataset,
        parameters,
        {
            'ks_iteration_cache': FULL_USE,
            'ks_feature_db': COMPARE_ONLY,
            'ks_parallelism': 0,
            'run_parallelism': 0,
        })
    return oevexp

## Also define a simple experiment using only IGt, used to compare
## the results of OEvExp with IGt.
def define_oevexp_igt_vs_ks(target, maxQ):
    global dataset
    parameters = collections.OrderedDict()
    parameters['algorithm'] = ['KS', 'IG']
    parameters['Q'] = range(1, maxQ + 1)
    parameters['target'] = [target]
    parameters['K'] = [5]

    oevexp_igt = ExperimentDefinition(
        'oevexp_igt_vs_ks_{}'.format(target),
        dataset,
        parameters,
        {
            'ks_iteration_cache': DISABLED,
            'ks_feature_db': FULL_USE,
            'ks_parallelism': 0,
            'run_parallelism': 0
        })
    return oevexp_igt

