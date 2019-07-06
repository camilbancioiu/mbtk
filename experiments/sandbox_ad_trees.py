import sys
import os
from pathlib import Path
import pickle
import gc

# Assume that the 'experiments' folder, which contains this file, is directly
# near the 'mbff' package.
EXPERIMENTS_ROOT = Path(os.getcwd())
MBFF_PATH = EXPERIMENTS_ROOT.parents[0]
sys.path.insert(0, str(MBFF_PATH))



from mbff.dataset.ExperimentalDatasetDefinition import ExperimentalDatasetDefinition
from mbff.dataset.ExperimentalDataset import ExperimentalDataset
from mbff.dataset.sources.SampledBayesianNetworkDatasetSource import SampledBayesianNetworkDatasetSource

EXDS_REPO = EXPERIMENTS_ROOT / 'exds_repository'

ExDsDefinition_ALARM = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_alarm_1e6')
ExDsDefinition_ALARM.exds_class = ExperimentalDataset
ExDsDefinition_ALARM.source = SampledBayesianNetworkDatasetSource
ExDsDefinition_ALARM.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'alarm.bif',
    'sample_count': int(1e6),
    'random_seed': 128,
}

ExDsDefinition_ALARM_med = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_alarm_1e5')
ExDsDefinition_ALARM_med.exds_class = ExperimentalDataset
ExDsDefinition_ALARM_med.source = SampledBayesianNetworkDatasetSource
ExDsDefinition_ALARM_med.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'alarm.bif',
    'sample_count': int(1e5),
    'random_seed': 128,
}

ExDsDefinition_ALARM_med2 = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_alarm_8e4')
ExDsDefinition_ALARM_med2.exds_class = ExperimentalDataset
ExDsDefinition_ALARM_med2.source = SampledBayesianNetworkDatasetSource
ExDsDefinition_ALARM_med2.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'alarm.bif',
    'sample_count': int(8e4),
    'random_seed': 128,
}

ExDsDefinition_ALARM_small = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_alarm_1e4')
ExDsDefinition_ALARM_small.exds_class = ExperimentalDataset
ExDsDefinition_ALARM_small.source = SampledBayesianNetworkDatasetSource
ExDsDefinition_ALARM_small.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'alarm.bif',
    'sample_count': int(1e4),
    'random_seed': 128,
}

ExDsDefinition_ALARM_tiny = ExperimentalDatasetDefinition(EXDS_REPO, 'synthetic_alarm_2e3')
ExDsDefinition_ALARM_tiny.exds_class = ExperimentalDataset
ExDsDefinition_ALARM_tiny.source = SampledBayesianNetworkDatasetSource
ExDsDefinition_ALARM_tiny.source_configuration = {
    'sourcepath': EXPERIMENTS_ROOT / 'bif_repository' / 'alarm.bif',
    'sample_count': int(2e3),
    'random_seed': 128,
}

adtree_folder = Path('adtrees')
adtree_folder.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    exdsDef = ExDsDefinition_ALARM_med2

    exds = exdsDef.create_exds()
    if exdsDef.exds_ready():
        print('Loading dataset {}'.format(exdsDef.name))
        exds.load()
    else:
        print('Building dataset {}'.format(exdsDef.name))
        exds.build()

    from mbff.structures.ADTree import ADTree

    for LLT in [16384, 8192, 4096, 2048]:
        adtree_file = adtree_folder / 'adtree_{}_llt{}.pickle'.format(exdsDef.name, LLT)
        if adtree_file.exists():
            print('AD-tree @ LLT={} already exists'.format(LLT))
        else:
            print("Begin building AD-tree @ LLT={} for {}".format(LLT, exdsDef.name))
            AD_Tree = ADTree(exds.matrix.X, exds.matrix.get_values_per_column('X'), LLT, debug=True, debug_to_stdout=True)
            print("AD-tree @ LLT={} completed in {:.2f}s".format(LLT, AD_Tree.duration))
            print("AD-tree @ LLT={} contains {} ADNodes and {} VaryNodes".format(
                LLT, AD_Tree.ad_node_count, AD_Tree.vary_node_count))

            import humanize
            import pympler.asizeof

            AD_Tree.size = pympler.asizeof.asizeof(AD_Tree)
            print("AD-tree @ LLT={} size: {}".format(LLT, humanize.naturalsize(AD_Tree.size)))

            with adtree_file.open('wb') as output:
                pickle.dump(AD_Tree, output)

            AD_Tree = None
            gc.collect()
