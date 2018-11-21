import pickle
import multiprocessing


## Feature Database

class KSFeatureDatabase:
    def __init__(self):
        self.lock = multiprocessing.Lock()
        self.feature_db = {}
        self.feature_db_filename = ''

    def set_filename(self, filename):
        self.feature_db_filename = filename

    def load(self):
        try:
            with open(self.feature_db_filename, 'rb') as f:
                self.feature_db = pickle.load(f)
        except FileNotFoundError:
            self.feature_db = {}

    def save(self):
        with open(self.feature_db_filename, 'wb') as f:
            pickle.dump(self.feature_db, f)

    def store(self, key, feature, metadata=None):
        with self.lock:
            self.feature_db[key] = (feature, metadata)
            self.save()

    def get(self, key):
        with self.lock:
            (feature, metadata) = self.feature_db[key]
        return feature

KSFDB = KSFeatureDatabase()

def get_ks_feature_db():
    return KSFDB

