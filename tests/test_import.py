from importlib.metadata import version

import jeam


def test_import_and_version():
    assert jeam.__version__ == version("jeam")
