import sys
import os
import pickle
from pathlib import Path


# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd())
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))

CITest_Significance = 0.95
LLT = 0


class ExperimentalPathSet:

    def __init__(self, root):
        self.Root = root
        self.ExDsRepository = self.Root / 'exds_repository'
        self.ExpRunRepository = self.Root / 'exprun_repository'
        self.BIFRepository = self.Root / 'bif_repository'



class ExperimentalSetup:

    def __init__(self):
        self.ExperimentDef = None
        self.ExDsDef = None
        self.Paths = None
        self.Omega = None
        self.CITest_Significance = None
        self.LLT = None
        self.ADTree = None
        self.AlgorithmRunParameters = None
        self.Arguments = None


    def update_paths(self):
        self.Paths.Experiment = self.ExperimentDef.path
        self.Paths.ADTreeRepository = self.Experiment / 'adtrees'
        self.Paths.JHTRepository = self.Experiment / 'jht'
        self.Paths.CITestResultRepository = self.Experiment / 'ci_test_results'
        self.Paths.DoFCacheRepository = self.Experiment / 'dof_cache'

        self.Paths.ADTreeRepository.mkdir(parents=True, exist_ok=True)
        self.Paths.JHTRepository.mkdir(parents=True, exist_ok=True)
        self.Paths.CITestResultRepository.mkdir(parents=True, exist_ok=True)
        self.Paths.DoFCacheRepository.mkdir(parents=True, exist_ok=True)

        adtree_filename = 'adtree_{}_llt{}.pickle'.format(self.ExDsDef.name, self.LLT)
        self.Paths.ADTree = self.Paths.ADTreeRepository / adtree_filename


    def preload_ADTree(self):
        with self.Paths.ADTree.open('rb') as f:
            self.ADTree = pickle.load(f)
        self.set_preloaded_ADTree_to_relevant_algrun_parameters()


    def filter_algruns_by_tag(self, tag):
        self.AlgorithmRunParameters = [p for p in self.AlgorithmRunParameters if tag in p['tags']]


    def is_tag_present_in_any_algrun(self, tag):
        for parameters in self.AlgorithmRunParameters:
            if tag in parameters['tags']:
                return True
        return False


    def set_preloaded_ADTree_to_relevant_algrun_parameters(self):
        for parameters in self.AlgorithmRunParameters:
            if 'adtree' in parameters['tags']:
                parameters['ci_test_ad_tree_preloaded'] = self.ADTree
                del parameters['ci_test_ad_tree_path__load']


################################################################################
# Command-line interface
if __name__ == '__main__':
    import mbff.utilities.experiment as utilcli
    import definitions
    import algrun_parameters
    import commands

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--preload-adtree', action='store_true')
    argparser.add_argument('--algrun-tag', type=str, default=None, nargs='?')

    object_subparsers = argparser.add_subparsers(dest='object')
    object_subparsers.required = True

    utilcli.configure_objects_subparser__exp_def(object_subparsers)
    utilcli.configure_objects_subparser__exds_def(object_subparsers)
    utilcli.configure_objects_subparser__exds(object_subparsers)
    utilcli.configure_objects_subparser__exp(object_subparsers)
    utilcli.configure_objects_subparser__algruns(object_subparsers)
    utilcli.configure_objects_subparser__algrun_datapoints(object_subparsers)

    commands.configure_objects_subparser__adtree(object_subparsers)

    arguments = argparser.parse_args()

    experimental_setup = ExperimentalSetup()
    experimental_setup.Paths = ExperimentalPathSet(EXPERIMENTS_ROOT)
    experimental_setup.ExDsDef = definitions.exds_definition(experimental_setup)
    experimental_setup.ExperimentDef = definitions.exds_definition(experimental_setup)
    experimental_setup.Arguments = arguments
    experimental_setup.update_paths()
    experimental_setup.AlgorithmRunParameters = algrun_parameters.create_algrun_parameters(experimental_setup)

    if arguments.algrun_tag is not None:
        experimental_setup.select_algruns_by_tag(arguments.algrun_tag)

    if arguments.preload_adtree is True:
        if experimental_setup.is_tag_present_in_any_algrun('adtree'):
            experimental_setup.preload_ADTree()


    command_handled = utilcli.handle_command(experimental_setup)
    if command_handled is False:
        commands.handle_command(experimental_setup)
