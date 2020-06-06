import sys
import os
import pickle
from pathlib import Path


# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd())
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))


import mbff.math.DSeparationCITest
import mbff.math.G_test__with_AD_tree
import mbff.math.G_test__unoptimized
import mbff.math.G_test__with_dcMI
import mbff.math.DoFCalculators

import mbff.utilities.functions as util


################################################################################
# Create AlgorithmRun parameters

# Create all AlgorithmRun parameters
def create_algrun_parameters(experimental_setup):
    default_parameters = create_default_parameters(experimental_setup)

    parameters_dsep = create_algrun_parameters__dsep(experimental_setup, default_parameters)
    parameters_unoptimized = create_algrun_parameters__unoptimized(experimental_setup, default_parameters)
    parameters_adtree_static = create_algrun_parameters__adtree_static(experimental_setup, default_parameters)
    parameters_dcmi = create_algrun_parameters__dcmi(experimental_setup, default_parameters)

    parameters_list = [] \
        + parameters_dsep \
        + parameters_unoptimized \
        + parameters_adtree_static \
        + parameters_dcmi

    for index, parameters in enumerate(parameters_list):
        parameters['index'] = index
        ID_format = parameters['ID']
        parameters['ID'] = ID_format.format(**parameters)

    return parameters_list



def create_default_parameters(experimental_setup):
    bn_sourcepath = experimental_setup.ExDsDef.source_configuration['sourcepath']
    bayesian_network = prepare_bayesian_network(bn_sourcepath)

    default_parameters = {
        'omega': experimental_setup.Omega,
        'source_bayesian_network': bayesian_network,
        'algorithm_debug': 1,
        'ci_test_debug': 0,
        'ci_test_significance': experimental_setup.CITest_Significance,
    }

    return default_parameters



def prepare_bayesian_network(sourcepath, force_rebuild=False):
    bayesian_network = None

    # BIF files might be large, so we read them from source and then we pickle
    # them to files. If a pickle-file is found, read it instead of the
    # requested BIF file.
    cachefile = sourcepath.with_suffix('.pickle')
    if cachefile.exists() and force_rebuild is False:
        with cachefile.open('rb') as f:
            bayesian_network = pickle.load(f)
        return bayesian_network

    bayesian_network = util.read_bif_file(sourcepath)
    bayesian_network.finalize()
    with cachefile.open('wb') as f:
        pickle.dump(bayesian_network, f)
    return bayesian_network


# Create AlgorithmRun parameters using the D-separation CI test
def create_algrun_parameters__dsep(experimental_setup, default_parameters):
    bayesian_network = default_parameters['source_bayesian_network']
    target_count = len(bayesian_network)
    citrrepo = experimental_setup.Paths.CITestResultRepository
    dsep = mbff.math.DSeparationCITest.DSeparationCITest
    exds_name = experimental_setup.ExDsDef.name

    parameters_list = list()
    for target in range(target_count):
        citr_filename = 'ci_test_results_{}_T{}_dsep.pickle'.format(exds_name, target)
        parameters = {
            'target': target,
            'ci_test_class': dsep,
            'ci_test_results_path__save': citrrepo / citr_filename,
            'tags': ['dsep', 'fast', 'no_dependencies'],
            'ID': 'run_{index}_T{target}__dsep',
        }
        parameters.update(default_parameters)
        parameters_list.append(parameters)

    return parameters_list


# Create AlgorithmRun parameters using the unoptimized G-test
def create_algrun_parameters__unoptimized(experimental_setup, default_parameters):
    bayesian_network = default_parameters['source_bayesian_network']
    target_count = len(bayesian_network)
    citrrepo = experimental_setup.Paths.CITestResultRepository
    g_test__unoptimized = mbff.math.G_test__unoptimized.G_test
    dof__structural = mbff.math.DoFCalculators.StructuralDoF
    exds_name = experimental_setup.ExDsDef.name

    parameters_list = list()
    for target in range(target_count):
        citr_filename = 'ci_test_results_{}_T{}_unoptimized.pickle'.format(exds_name, target)
        parameters = {
            'target': target,
            'ci_test_class': g_test__unoptimized,
            'ci_test_dof_calculator_class': dof__structural,
            'ci_test_results_path__save': citrrepo / citr_filename,
            'tags': ['unoptimized', 'slow', 'no_dependencies'],
            'ID': 'run_{index}_T{target}__unoptimized',
        }
        parameters.update(default_parameters)
        parameters_list.append(parameters)

    return parameters_list


# Create AlgorithmRun parameters using the G-test optimized with an AD-tree @LLT=0
def create_algrun_parameters__adtree_static(experimental_setup, default_parameters):
    bayesian_network = default_parameters['source_bayesian_network']
    target_count = len(bayesian_network)
    citrrepo = experimental_setup.Paths.CITestResultRepository
    g_test__adtree = mbff.math.G_test__with_AD_tree.G_test
    dof__structural = mbff.math.DoFCalculators.StructuralDoF
    exds_name = experimental_setup.ExDsDef.name

    parameters_list = list()
    for target in range(target_count):
        for llt in experimental_setup.AllowedLLTArgument:
            citr_filename = 'ci_test_results_{}_T{}_ADtree_LLT{}.pickle'.format(exds_name, target, llt)
            parameters = {
                'target': target,
                'ci_test_class': g_test__adtree,
                'ci_test_dof_calculator_class': dof__structural,
                'ci_test_ad_tree_leaf_list_threshold': experimental_setup.calculate_absolute_LLT_from_llt_argument(llt),
                'ci_test_ad_tree_llt_argument': llt,
                'ci_test_ad_tree_path__load': experimental_setup.get_ADTree_path_for_llt_argument(llt),
                'ci_test_results_path__save': citrrepo / citr_filename,
                'tags': ['adtree-static', 'adtree-llt{}'.format(llt), 'fast', 'has_dependencies'],
                'ID': 'run_{index}_T{target}__@LLT={ci_test_ad_tree_llt_argument}',
            }
            parameters.update(default_parameters)
            parameters_list.append(parameters)

    return parameters_list


# Create AlgorithmRun parameters using the G-test optimized with dcMI
def create_algrun_parameters__dcmi(experimental_setup, default_parameters):
    bayesian_network = default_parameters['source_bayesian_network']
    target_count = len(bayesian_network)
    citrrepo = experimental_setup.Paths.CITestResultRepository
    g_test__dcmi = mbff.math.G_test__with_dcMI.G_test
    dof__structural_cached = mbff.math.DoFCalculators.CachedStructuralDoF
    exds_name = experimental_setup.ExDsDef.name

    jht_filename = 'jht_{}.pickle'.format(experimental_setup.ExDsDef.name)
    jht_path = experimental_setup.Paths.JHTRepository / jht_filename

    dof_cache_filename = 'dofcache_{}.pickle'.format(experimental_setup.ExDsDef.name)
    dof_cache_path = experimental_setup.Paths.DoFCacheRepository / dof_cache_filename

    parameters_list = list()
    for target in range(target_count):
        citr_filename = 'ci_test_results_{}_T{}_dcMI.pickle'.format(exds_name, target)
        parameters = {
            'target': target,
            'ci_test_class': g_test__dcmi,
            'ci_test_dof_calculator_class': dof__structural_cached,
            'ci_test_jht_path__load': jht_path,
            'ci_test_jht_path__save': jht_path,
            'ci_test_dof_calculator_cache_path__load': dof_cache_path,
            'ci_test_dof_calculator_cache_path__save': dof_cache_path,
            'ci_test_results_path__save': citrrepo / citr_filename,
            'tags': ['dcmi', 'fast', 'no_dependencies'],
            'ID': 'run_{index}_T{target}__dcMI',
        }

        parameters.update(default_parameters)
        parameters_list.append(parameters)

    return parameters_list
