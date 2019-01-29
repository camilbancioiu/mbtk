import os
import operator
from pathlib import Path
import pickle
import utilities as util
import numpy

EXDSSTATS_FROM_EXDS = 'E'
EXDSSTATS_FROM_SAVED_STATS = 'S'

class ExperimentalDatasetStats():

    def __init__(self, exds_definition=None, exds=None):
        self.exds = None
        self.exds_definition = None
        self.loaded_from = None
        self.stats = None
        if not exds_definition == None:
            self.from_saved_stats(exds_definition)
        if not exds == None:
            self.from_exds(exds)

    def __getitem__(self, key):
        return self.stats.get(key, '?')

    def __contains__(self, key):
        return self.stats.__contains__[key]

    def __iter__(self):
        return self.stats.__iter__()

    def __str__(self):
        if self.stats == None:
            return ''

        output = "Dataset split at {0} into {1} training documents and {2} testing documents."\
               .format(self['train_rows_proportion'], self['train_docs'], self['test_docs'])

        output += "\nVocabulary size: Train {0} / Full {1}, Topics: {2}\n"\
               .format(self['train_vocab_size'], self['full_vocab_size'], self['topics_count'] )
        output += 'Topic list: {}\n'.format(self['topic_list'])
        output += 'Topic use proportions: {}\n'.format(self['topic_use_proportions'])
        return output

    def to_csv_dict(self):
        return {k: v for k, v in self.stats.items() if k in ExperimentalDatasetStats.csv_keys()}

    def from_exds(self, exds):
        self.exds = exds
        self.exds_definition = self.exds.definition
        self.load_from_exds()
        self.loaded_from = EXDSSTATS_FROM_EXDS

    def from_saved_stats(self, exds_definition):
        self.exds = None
        self.exds_definition = exds_definition
        self.load_from_saved_stats()
        self.loaded_from = EXDSSTATS_FROM_SAVED_STATS

    def load_from_exds(self):
        stats = {}
        stats['name'] = self.exds_definition.name
        stats['train_rows_proportion'] = self.exds.train_rows_proportion
        stats['train_docs'] = len(self.exds.train_rows)
        stats['test_docs'] = len(self.exds.test_rows)
        stats['total_docs'] = stats['train_docs'] + stats['test_docs']
        stats['train_vocab_size'] = self.exds.Xtrain.get_shape()[1]
        stats['full_vocab_size'] = self.exds.X.get_shape()[1]
        stats['topics_count'] = self.exds.Ytrain.get_shape()[1]
        stats['topic_use_proportions'] = self.get_topic_use_proportions()
        stats['topic_list'] = list(self.exds.topics)
        self.stats = stats
        return stats

    def load_from_saved_stats(self):
        if not self.exds_definition.folder_exists():
            raise ExperimentalDatasetStatsError(str(self.exds_definition), 'ExDs folder does not exist.')
        folder = self.exds_definition.folder + '/analysis'
        with open(folder + '/stats.pickle', 'rb') as f:
            stats = pickle.load(f)
        self.stats = stats
        return stats

    def save(self):
        self.delete_saved_stats()
        if not self.loaded_from == EXDSSTATS_FROM_EXDS:
            raise ExperimentalDatasetStatsError(str(self.exds_definition), 'Cannot save stats loaded directly from files.')
        if self.exds_definition.folder_is_locked():
            raise ExperimentalDatasetStatsError(str(self.exds_definition), 'ExDs folder is locked.')
        folder = self.exds_definition.folder + '/analysis'
        path = Path('./' + folder)
        path.mkdir(parents=True, exist_ok=True)
        with open(folder + '/stats.pickle', 'wb') as f:
            pickle.dump(self.stats, f)

    def delete_saved_stats(self):
        if self.exds_definition.folder_is_locked():
            raise ExperimentalDatasetStatsError(str(self.exds_definition), 'ExDs folder is locked.')
        folder = self.exds_definition.folder + '/analysis'
        try:
            os.remove(folder + '/stats.pickle')
            os.remove(folder + '/co_stats.pickle')
        except:
            pass
 
    def get_topic_use_proportions(self):
        topic_use_proportions = {}
        columns = self.exds.get_Y_count()
        for i in range(columns):
            Yi = self.exds.get_Y_column(i)
            topic_use_proportions[i] = round(numpy.sum(Yi) / len(Yi), 2)
        return topic_use_proportions
    def csv_keys():
        return ['name', 'total_docs', 'train_docs', 'test_docs',
            'full_vocab_size', 'train_vocab_size', 'topics_count']

class ExperimentalDatasetStatsError(Exception):
    def __init__(self, exds_definition, message):
        self.exds_definition = exds_definition
        self.message = message
