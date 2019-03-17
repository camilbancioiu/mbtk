from pathlib import Path
from pprint import pprint

import mbff.utilities.functions as util
from mbff_tests.TestBase import TestBase


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


    def test_building_pomegranate_probability_distributions(self):
        survey_bif = Path('testfiles', 'bif_files', 'survey.bif')
        bn = util.read_bif_file(survey_bif)
        distributions = bn.create_pomegranate_probability_distributions()

        A = distributions['A']
        E = distributions['E']
        print('==========')
        print(E)
        print('==========')
        pprint(E.keymap)
        print(E.__class__.__name__)
        pprint(dir(E))
        self.assertEqual(12, len(E.keymap.keys()))
        E.bake(list(E.keymap.keys()))
        print('==========')
        print(E.parents)
        print(E.sample())
        print(E.values)


