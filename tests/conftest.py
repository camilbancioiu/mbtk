pytest_plugins = ['mbff_tests.fixtures_survey',
                  'mbff_tests.fixtures_alarm',
                  'mbff_tests.fixtures_lc']


def pytest_configure(config):
    # register the 'slow' marker
    config.addinivalue_line(
        "markers", "slow: designates slow tests"
    )
