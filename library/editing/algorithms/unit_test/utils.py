from itertools import compress
import pandas as pd
import numpy as np
from numpy import log as ln
import math
import re
import collections
import os, json, time, errno, copy

# define object class to store object data
class C:
    def __iter__(self):
        return CIterator(self)

    def __setattr__(self, name, value):
        if type(value) == dict:
            self.__dict__[name] = C()
            for item in value.items():
                setattr(self.__dict__[name], item[0], Util.dict2obj(item[1]))
        elif type(value) == list:
            self.__dict__[name] = Util.dict2obj(value)
        else:
            self.__dict__[name] = value

    def append(self, value):
        pass


class CIterator:
    def __init__(self, c):
        self._c = c
        self._index = 0

    def __next__(self):
        if self._index < self._c.__dict__.__len__():
            self._index += 1
            return list(self._c.__dict__.items())[self._index - 1]
        raise StopIteration


class CList(list):
    def append(self, obj):
        super(CList, self).append(Util.dict2obj(obj))


class Util:
    """just helper functions"""

    def __init__(self):
        pass

    @staticmethod
    def zero_if_none(var):
        return 0 if var is None else var

    @staticmethod
    def none_or_zero(var):
        return var == None or var == 0

    @staticmethod
    def interpolate_2d_lookup(d_in: pd.DataFrame, x_in, y_in):

        # assumes dataframe looks like:
        # y_axis_name,    x_axis_value_1,    x_axis_value_2,      ...
        # y_axis_value_1       value              value          value
        # y_axis_value_2       value              value          value
        # ...                  value              value          value

        # convert the headers of the data frame (the x axis) to floats. All letters and special characters are removed - unless there is
        # an underscore inbetween two digits that will be interpreted as a decimal. This is to get around the strict naming rules in Hx
        x_axis = [
            float(re.sub("[^0-9.]*", "", re.sub("(?<=[0-9])_(?=[0-9])", ".", v))) for v in d_in.columns.tolist()[1:]
        ]
        x_axis_match_index_lower = len(list(compress(x_axis, [item <= x_in for item in x_axis]))) - 1
        x_axis_match_index_upper = x_axis_match_index_lower + 1
        x_axis_lower_value = x_axis[x_axis_match_index_lower]
        x_axis_upper_value = x_axis[x_axis_match_index_upper]

        y_axis_name = d_in.columns.tolist()[0]
        y_axis = d_in[y_axis_name].tolist()
        y_axis_match_index_lower = len(list(compress(y_axis, [item <= y_in for item in y_axis]))) - 1
        y_axis_match_index_upper = y_axis_match_index_lower + 1
        y_axis_lower_value = y_axis[y_axis_match_index_lower]
        y_axis_upper_value = y_axis[y_axis_match_index_upper]

        x_low_y_low_val = d_in.iloc[y_axis_match_index_lower, x_axis_match_index_lower + 1]
        x_low_y_up_val = d_in.iloc[y_axis_match_index_upper, x_axis_match_index_lower + 1]
        x_up_y_low_val = d_in.iloc[y_axis_match_index_lower, x_axis_match_index_upper + 1]
        x_up_y_up_val = d_in.iloc[y_axis_match_index_upper, x_axis_match_index_upper + 1]

        # to excel workbook was exploding when x axis lower was 0
        # ln(0) = -inf, number--inf = number + infinity, number / infinity -> 0
        if x_axis_lower_value == 0:
            section_x = 0
        else:
            section_x = (ln(x_in) - ln(x_axis_lower_value)) / (ln(x_axis_upper_value) - ln(x_axis_lower_value))

        # same will happen if y axis lower is 0
        if y_axis_lower_value == 0:
            section_y = 0
        else:
            section_y = (ln(y_in) - ln(y_axis_lower_value)) / (ln(y_axis_upper_value) - ln(y_axis_lower_value))

        return (
            x_low_y_low_val
            + (x_up_y_low_val - x_low_y_low_val) * section_x
            + (
                (x_low_y_up_val + (x_up_y_up_val - x_low_y_up_val) * section_x)
                - (x_low_y_low_val + (x_up_y_low_val - x_low_y_low_val) * section_x)
            )
            * section_y
        )

    @staticmethod
    def get_dict_types(d_in):
        return {k: type(v) for k, v in d_in.inputs_dict.items()}

    @staticmethod
    def getattr2(obj, attr1: str, attr2: str):
        return getattr(getattr(obj, attr1), attr2)

    @classmethod
    def dict2obj(Util, d):

        # if variable is a list then recurse over each item in list
        if isinstance(d, list):
            d = CList([Util.dict2obj(x) for x in d])

        # dont do anything if the variable is not a dictionary
        if not isinstance(d, dict):
            # return Util.type_converter(d)
            return d

        # create object and store dictionary values in it
        obj = C()

        for k in d:
            obj.__dict__[k] = Util.dict2obj(d[k])

        return obj

    @classmethod
    def obj2dict(Util, o):

        # if variable passed in is list then recursively call function on each element
        if isinstance(o, list):
            o = [Util.obj2dict(x) for x in o]

        # if the object has attributes - shove each one in a dictionary
        dic = dict()
        if hasattr(o, "__dict__"):
            for k, v in o.__dict__.items():
                dic[k] = Util.obj2dict(v)
            return dic
        else:
            return o

    @staticmethod
    def string_to_var_name(str_in: str) -> str:
        return re.sub("[^0-9a-zA-Z_]*", "", re.sub("[ ]+", "_", str_in.lower()))

    # converts data to json serializeable types - also some hacks to make specific ints floats
    # basically just converts types and values into something that hx will accept
    @staticmethod
    def type_converter(v: ..., k: str = ""):
        if type(v) == np.int64 or type(v) == int:
            if (
                "premium" in k
                or "modifier" in k
                or ("tria" in k and "class" not in k)
                or "weight" in k
                or "bound" in k
                or v == 0
            ):
                return float(v)
            else:
                return int(v)
        elif type(v) == np.float64 or type(v) == float:
            return float(0.0) if math.isnan(v) else float(v)
        elif type(v) == np.int64 or type(v) == int:
            return int(0) if math.isnan(v) else int(v)
        elif type(v) == pd.Timestamp:
            return v.strftime("%Y-%m-%d")
        elif type(v) == dict:
            return {key: Util.type_converter(value, key) for key, value in v.items()}
        elif type(v) == float:
            return float(0.0) if math.isnan(v) else float(v)
        elif type(v) == str or type(v) == bool:
            return v
        elif v is None:
            return ""
        else:
            return {key: Util.type_converter(value, key) for key, value in v.__dict__.items()}

    @staticmethod
    def iter_dict(d_in, func):
        if isinstance(d_in, collections.Mapping):
            return {k: Util.iter_dict(v, func) for k, v in d_in.items()}
        elif isinstance(d_in, list):
            return [Util.iter_dict(v, func) for k, v in enumerate(d_in)]
        else:
            return func(d_in)

    @staticmethod
    def get_abs_path(rel_file_path=None):
        """Gets absolute path from a path relative to the current file's location. If no relative path provided, returns absolute path to directory in which calling code file exists."""
        dirname = os.path.dirname(__file__)
        if rel_file_path is None:
            return dirname
        else:
            filename = os.path.join(dirname, rel_file_path)
            return filename

    class DataMergeError(Exception):
        """Error class for json object data merge exceptions."""

        pass

    @staticmethod
    def data_merge(a, b):
        """deep merges object b into object a and return merged result
        NOTE: tuples and arbitrary objects are not handled"""
        key = None
        # ## debug output
        # sys.stderr.write("DEBUG: %s to %s\n" %(b,a))
        try:
            if a is None or isinstance(a, str) or isinstance(a, str) or isinstance(a, int) or isinstance(a, float):
                # border case for first run or if a is a primitive
                a = b
            elif isinstance(a, list):
                # lists can be only appended
                if isinstance(b, list):
                    # merge lists
                    a.extend(b)
                else:
                    # append to list
                    a.append(b)
            elif isinstance(a, dict):
                # dicts must be merged
                if isinstance(b, dict):
                    for key in b:
                        if key in a:
                            a[key] = Util.data_merge(a[key], b[key])
                        else:
                            a[key] = b[key]
                else:
                    raise Util.DataMergeError('Cannot merge non-dict "%s" into dict "%s"' % (b, a))
            else:
                raise Util.DataMergeError('NOT IMPLEMENTED "%s" into "%s"' % (b, a))
        except TypeError as e:
            raise Util.DataMergeError('TypeError "%s" in key "%s" when merging "%s" into "%s"' % (e, key, b, a))
        return a

    @staticmethod
    def load_json(file_path):
        """Read json from file and create json object."""
        with open(file_path, "r") as json_file:
            data = json.loads(json_file.read())
            return data

    @staticmethod
    def dump_json(jsonobj, dump_path):
        """Outputs json to file."""
        with open(dump_path, "w") as json_file:
            json.dump(jsonobj, json_file, sort_keys=False, indent=4, cls=NumpyEncoder)

    @staticmethod
    def dump_string(str, dump_path):
        """Outputs json to file."""
        with open(dump_path, "w") as str_file:
            str_file.write(str)

    @staticmethod
    def dump_pyobj(pyobj, dump_path):
        """Converts an hxd-like python object to json and outputs to file."""
        Util.dump_json((pyobj), dump_path)

    @staticmethod
    def silentremove(filename):
        """Delete's a file if it exists, does nothing if it does not. Construction avoide unnecessary checking of existence of directory"""
        try:
            os.remove(filename)
        except OSError as e:  # this would be "except OSError, e:" before Python 2.6
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                raise  # re-raise exception if a different error occurred

    @staticmethod
    def generate_timestamp():
        """Generates timestamp for use in filenames."""
        return time.strftime("%Y%m%d-%H%M")


class UnitTest:
    @staticmethod
    def execute_ave_bulk_test(test_name, test_description, data_set_index, actual, expected, out_file_path):
        test_passed = True

        log = {}
        log["data_set_index"] = data_set_index
        log["test_name"] = test_name
        log["test_description"] = test_description

        for path, expected_val in expected.items():
            is_equal = True

            try:
                actual_val = eval(f"actual.{path}")
                is_equal = Compare.is_equal(actual_val, expected_val)
                msg = ""
            except Exception:
                msg = f"Unable to resolve path on output: {path}"
                actual_val = np.nan
                is_equal = False

            if is_equal == False:
                test_passed = False

            log["schema_path"] = path
            log["actual"] = actual_val
            log["expected"] = expected_val
            log["passed"] = is_equal
            log["msg"] = msg
            UnitTest.append_dictionary_file(out_file_path, log)

        return test_passed

    ##populates the json output
    @staticmethod
    def append_dictionary_file(file_path, dictionary):
        out_s = json.dumps(dictionary)
        if os.path.exists(file_path) == False:
            with open(file_path, mode="w") as file:
                file.write("[{}]".format(json.dumps(dictionary)))
        else:
            with open(file_path, mode="r+") as file:
                file.seek(0, 2)
                position = file.tell() - 1
                file.seek(position)
                file.write(",{}]".format(json.dumps(dictionary)))

    @staticmethod
    def get_test_policy_and_expected_outputs(ave_test_policy, compare_paths):
        """Returns expected outputs from comparison paths and clears test policy for subsequent testing"""
        # Get expected values from path and clear to avoid expected values being compared with themselves
        expected_outputs = {}

        # Actual values should default to null
        for curr_path in compare_paths:
            try:
                # Try to read property value
                prop = eval(f"ave_test_policy.{curr_path}")
                expected_val = copy.copy(prop)

                # And remove from tree
                exec(f"del ave_test_policy.{curr_path}")
            except:
                # If value not found, set to nan - which is ignored by the compare function
                expected_val = np.nan

            # Add to dictionary of expected values
            expected_outputs[curr_path] = expected_val

        return ave_test_policy, expected_outputs

    def get_full_compare_paths_for_list_and_attributes(
        ave_test_policy, comparison_base_list_path, child_attributes_list
    ):
        """Based on selected test policy, list node path and list of child attributes, returns full list of fully qualified paths."""

        # Evaluate to get the list object to iterate over
        comparison_base_list = eval(f"ave_test_policy.{comparison_base_list_path}")

        # Compile into a list of fully qualified paths
        compare_paths = []

        # Iterate over list node to get full list over all items
        for curr_sub_index in range(0, len(comparison_base_list)):
            this_path = f"{comparison_base_list_path}[{curr_sub_index}]."
            # Add this path as a prefix to the list of child attributes and add to list of paths for comparison
            compare_paths = compare_paths + [this_path + x for x in child_attributes_list]

        return compare_paths


class Compare:
    """Library with methods for comparing values"""

    @staticmethod
    def is_float(val):
        return isinstance(val, (float, np.float_, np.float16, np.float32, np.float64))

    @staticmethod
    def is_nan(val):
        if isinstance(val, (str, bool)):
            return False
        else:
            return math.isnan(float(val))

    @staticmethod
    def is_equal(actual_value, expected_value):
        # if np.isnan(expected_value):
        # if pd.isnull(expected_value) or pd.isna(expected_value):
        if Compare.is_nan(expected_value):
            return True
        elif Compare.is_float(expected_value) or Compare.is_float(actual_value):
            return math.isclose(actual_value, expected_value, rel_tol=0.01)
        elif expected_value == actual_value:
            return True
        else:
            return False


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)