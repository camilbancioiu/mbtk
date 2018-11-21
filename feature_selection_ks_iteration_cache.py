import multiprocessing
import pickle


## Candidate Markov blanket cache

class KSIterationCache:
    def __init__(self):
        self.cmbs = {}
        self.deltas = {}
        self.feature_uses = {}
        self.iteration_stats = {}
        self.iteration_hits = 0
        self.iteration_misses = 0
        self.iteration_key = None
        self.filename = ''

    def reset(self):
        self.cmbs = {}
        self.deltas = {}
        self.feature_uses = {}
        self.iteration_stats = {}

    def set_iteration_key(self, key):
        self.iteration_key = key

    def set_stats_filename(self, filename):
        self.filename = filename

    def update(self, candidates):
        self.feature_uses = {}
        self.reset_iteration_stats()
        for c in candidates:
            self.update_cmb(c)
            self.update_feature_usage(c)
            self.add_to_iteration_stats(c)
        self.update_iteration_stats()

    def get_cmb(self, feature):
        return self.cmbs[feature]

    def get_delta(self, feature):
        return self.deltas[feature]

    def update_cmb(self, c):
        (feature, mb_delta, cmb, cached_mb_delta, cached_cmb, hit) = c
        self.cmbs[feature] = cmb
        self.deltas[feature] = mb_delta

    def update_feature_usage(self, c):
        (feature, mb_delta, cmb, cached_mb_delta, cached_cmb, hit) = c
        for cmb_feature in cmb:
            try:
                self.feature_uses[cmb_feature].append(feature)
            except KeyError:
                self.feature_uses[cmb_feature] = [feature]

    def remove_feature(self, feature):
        try:
            for feature_usage in self.feature_uses[feature]:
                del self.cmbs[feature_usage]
                del self.deltas[feature_usage]
            del self.feature_uses[feature]
        except KeyError:
            pass
        del self.cmbs[feature]

    def save_stats(self):
        pickle.dump(self.iteration_stats, open(self.filename, 'wb'))

    def reset_iteration_stats(self):
        self.iteration_hits = 0
        self.iteration_misses = 0

    def add_to_iteration_stats(self, c):
        (feature, mb_delta, cmb, cached_mb_delta, cached_cmb, hit) = c
        if hit:
            self.iteration_hits += 1
        else:
            self.iteration_misses += 1

    def update_iteration_stats(self):
        self.iteration_stats[self.iteration_key] = (self.iteration_hits, self.iteration_misses)

KSIC = KSIterationCache()

def get_ks_iteration_cache():
    return KSIC
