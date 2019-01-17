import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import subprocess
from definitions import (ExperimentalDatasets,
                         ExperimentalDatasetDefinition,
                         get_from_definitions,
                         add_to_definitions)


class ExDsTestCase(unittest.TestCase):
    def setUp(self):
        ExperimentalDatasets = {}
        add_to_definitions(ExperimentalDatasets, [
            ExperimentalDatasetDefinition('test_exds', 'I3302022', 0.30, (0.1, 0.9)),
        ])

    def test_exds_list_all(self):
        process = subprocess.run(['./exds.py', 'list', 'all'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        stdout = process.stdout.decode('utf-8')
        stderr = process.stderr.decode('utf-8')
        # self.assertEqual(len(stdout.splitlines()), 2)
        print("--------asdfasdfasdf-----------")
        print(stdout)

if __name__ == '__main__':
    t = ExDsTestCase()
    t.setUp()
    t.test_exds_list_all()
