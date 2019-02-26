import unittest
import os
import shutil
from pathlib import Path

class TestBase(unittest.TestCase):
    TestFilesRootFolder = 'testfiles/tmp'

    def ensure_tmp_subfolder(self, subfolder):
        path = Path(TestBase.TestFilesRootFolder + '/' + subfolder)
        path.mkdir(parents=True, exist_ok=True)
        return path


    def ensure_empty_tmp_subfolder(self, subfolder):
        try:
            shutil.rmtree(TestBase.TestFilesRootFolder + '/' + subfolder)
        except FileNotFoundError:
            pass
        path = Path(TestBase.TestFilesRootFolder + '/' + subfolder)
        path.mkdir(parents=True, exist_ok=True)
        return path
