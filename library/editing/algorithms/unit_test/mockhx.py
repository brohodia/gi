from operator import itemgetter
from algorithms.unit_test.parameter_tables_util import ParameterTable
from algorithms.unit_test.utils import Util
import os
import json


class DummyDecorator(object):
    def __call__(self, func):
        pass


class dummy:
    pass


def print_nothing(self):
    print("nothing")


def init_mockhx():
    hxmock = Hxdummy()
    hxmock.task = print_nothing
    return hxmock


class Hxdummy:
    """Dummy class for loading parameter tables when executing rating algorithms as part of pytest unit and functional tests.
    Each python file that is being tested needs to include the following conditional import for this to execute, replacing the standard "import hx":
        try:
            import hx
            test_params = hx.params
            import_type = 'standard'
        except:
            from algorithms.unit_test.mockhx import *
            hx = init_mockhx()
            import_type = 'mock'
    """

    def __init__(self):
        # Init params - empty class
        self.params = Hxparams()

        # NOTE: added in based on AIGRM PAC
        self.errors = Errors()

        # NOTE: required to run through code
        self.Hxd = None
        self.rating = None

        # Open parameter table schema from parameter tables folder
        curr_dir = Util.get_abs_path()
        folder = os.path.join(
            curr_dir,
            "..",
            "..",
            "parameter_tables",
        )
        json_def_path = os.path.join(folder, "parameter_tables_schema.json")
        paramdef = Util.load_json(json_def_path)

        # Iterate through schema, load from csv folded using schema structure for each parameter table
        for key, val in paramdef.items():
            setattr(self.params, key, ParameterTable.from_csv(f"{folder}/{key}.csv", schema=val).df())

    # Dummy to handle decorator function
    def task(func):
        def wrapper():
            return func()

        return wrapper


class Hxparams:
    def __int__(self):
        print("created")


class Errors:
    def __init__(self):
        pass

    def fatal(self, errMsg: str):
        raise Exception(errMsg)

    def validation(self, errMsg: str):
        print("Validation error: " + errMsg)