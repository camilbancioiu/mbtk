from pathlib import Path

from mbff_tests.TestBase import TestBase

import mbff.utilities.functions as util

class TestBIFReading(TestBase):

    def test_reading_bif_file(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
        bn = util.read_bif_file(survey_bif)

        self.assertEqual('survey', bn.name)
        self.assertEqual(6, len(bn.variables))
        self.assertListEqual(
                ["age", "education", "occupation", "sex", "transportation", "unknown"],
                sorted([variable.properties['label'] for variable in bn.variables.values()]))
        self.assertListEqual(['A', 'S'], bn.variables['E'].probdist.conditioning_set)
