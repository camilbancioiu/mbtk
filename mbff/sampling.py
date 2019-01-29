import itertools as Iter
import functools as F
import utilities as Util
import operator  as Op
import pickle

class Samples():
    def __init__(self, experiment, load=True):
        self.experiment = experiment
        if load:
            self.original_samples = self.load_samples(self.experiment)
        else:
            self.original_samples = []
        self.accessors = {}
        self.working_samples = list(self.original_samples)

    def __iter__(self):
        return self.working_samples.__iter__()

    def __len__(self):
        return list(self.working_samples).__len__()

    def load_samples(self, experiment):
        filename = '{}/samples/algrun-samples.pickle'.format(experiment.definition.folder)
        samples = []
        with open(filename, 'rb') as f:
            samples = pickle.load(f)
        return samples

    def set_default_accessors(self, default_properties):
        for default_property in default_properties:
            self.accessors[default_property] = Op.attrgetter(default_property)
        return self

    def list(self):
        return list(self.working_samples)

    def filter(self, propertyName, value):
        accessor = self.accessors[propertyName]
        predicate = lambda sample: accessor(sample) == value
        self.working_samples = list(filter(predicate, self.working_samples))
        return self

    def filterCustom(self, predicate):
        self.working_samples = list(filter(predicate, self.working_samples))
        return self

    def property(self, propertyName):
        accessor = self.accessors[propertyName]
        return list(map(accessor, self.working_samples))

    def values(self, propertyName):
        accessor = self.accessors[propertyName]
        return frozenset(map(accessor, self.working_samples))

    def max(self, propertyName):
        return max(self.property(propertyName))

    def maxSample(self, propertyName):
        accessor = self.accessors[propertyName]
        return max(self.working_samples, key=accessor)

    def argmax(self, argPropertyName, propertyName):
        argAccessor = self.accessors[argPropertyName]
        accessor = self.accessors[propertyName]
        return argAccessor(max(self.working_samples, key=accessor))

    def average(self, propertyName):
        return sum(self.property(propertyName)) / len(self)

    def sort(self, propertyName, reverse=False):
        accessor = self.accessors[propertyName]
        self.working_samples = sorted(self.working_samples, key=accessor, reverse=reverse)
        return self

    def reset(self):
        self.working_samples = list(self.original_samples)
        return self

    def new(self):
        # Make sure the reinstantiation takes into account that it
        # may happen in a subclass of Samples.
        samplesClass = self.__class__
        newSamples = samplesClass(self.experiment, False)
        newSamples.original_samples = list(self.working_samples)
        newSamples.reset()
        return newSamples

class KS_IGt_Samples(Samples):
    def __init__(self, experiment, load=True):
        super().__init__(experiment, load)
        self.default_properties = ["target", "algorithm", "Q", "K", "TP", "FP", "TN", "FN", "duration", "result", "accuracy"]
        self.set_default_accessors(self.default_properties)
