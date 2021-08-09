class SetCache:

    def __init__(self):
        self.cache = dict()


    def add(self, cset, *elements):
        self.cache[frozenset(elements)] = cset


    def get(self, *elements):
        return self.cache[frozenset(elements)]


    def contains(self, *elements):
        return frozenset(elements) in self.cache
