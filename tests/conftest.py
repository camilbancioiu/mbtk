pytest_plugins = ['tests.fixtures_survey',
                  'tests.fixtures_alarm',
                  'tests.fixtures_lc']


def pytest_configure(config):
    # register the 'slow' marker
    config.addinivalue_line(
        "markers", "slow: designates slow tests"
    )
