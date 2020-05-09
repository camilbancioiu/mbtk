import sys


class MultiFileWriter:

    def __init__(self, files):
        self.files = files


    def write(self, string):
        for f in self.files:
            f.write(string)


    def flush(self):
        for f in self.files:
            f.flush()


    def close(self):
        self.flush()
        for f in self.files:
            if f is not sys.__stdout__:
                f.close()
