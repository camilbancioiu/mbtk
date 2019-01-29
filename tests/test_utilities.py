import os
import shutil
from pathlib import Path

TestFilesRootFolder = 'testfiles/tmp'

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def ensure_tmp_subfolder(subfolder):
    path = Path(TestFilesRootFolder + '/' + subfolder)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_empty_tmp_subfolder(subfolder):
    try:
        shutil.rmtree(TestFilesRootFolder + '/' + subfolder)
    except FileNotFoundError:
        pass
    path = Path(TestFilesRootFolder + '/' + subfolder)
    path.mkdir(parents=True, exist_ok=True)
    return path
