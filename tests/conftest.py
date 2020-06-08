pytest_plugins = ['tests.fixtures_survey',
                  'tests.fixtures_alarm',
                  'tests.fixtures_lc',
                  'tests.fixtures_small',
                  'tests.fixtures_pathfinder',
                  'tests.fixtures_andes']


def pytest_configure(config):
    # register the 'slow' marker
    config.addinivalue_line(
        "markers", "slow: designates slow tests"
    )
    # register the 'demo' marker
    config.addinivalue_line(
        "markers", "demo: demo tests"
    )
    config.addinivalue_line(
        "markers", "demo_alarm: demo tests on alarm"
    )
    config.addinivalue_line(
        "markers", "demo_pathfinder: demo tests on pathfinder"
    )
    config.addinivalue_line(
        "markers", "demo_andes: demo tests on andes"
    )
