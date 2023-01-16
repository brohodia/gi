import re
import pandas as pd
import math
from datetime import datetime
from libraries.cm_testing.algorithms.DotDict import DotDict
from typing import List, Union, Any, Optional
from dataclasses import dataclass

# Tries to import HX package
try:
    import hx

    test_params = hx.params
    import_type = "standard"

except:
    # If that fails then return mockHX or error
    from algorithms.unit_test.mockhx import *

    try:
        hx = init_mockhx()
        import_type = "mock"
    except:
        print(f"Failed to initialise mock {__file__}")


def convert_single_layer_list_to_dataframe(list_obj, keys=[], rename_dict={}):
    """Converts a single layer list i.e. where you just want elements from one hxd list object
    To a dataframe
    And keeps required keys and renames them
    """
    df = pd.DataFrame([dict(list_el) for list_el in list_obj])

    if keys:
        df = df[keys]

    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df


def convert_nested_list_to_dataframe(
    parent_list,
    child_list: str,
    # keys: list = [],
    parent_keys: list = [],
    child_keys: list = [],
    rename_dict={},
):
    """Converts a nested list to a dataframe e.g. claims experience tables
    This is where you want items from the parent list and the child list
    Keeps required keys and renames them
    """

    # TODO: bugs when the parent and child list have the same name
    # TODO: workaround - would be to implement - child_keys,parent_keys
    # TODO: if the child key also exists in the parent key then append with {child_list}_{child_key}
    temp_list = []
    for parent_el in parent_list:
        for child_el in getattr(parent_el, child_list):

            parent_dict = dict(parent_el)
            parent_dict_subset = {key: value for key, value in parent_dict.items() if key in parent_keys}
            # del parent_dict[child_list]

            # temp_parent_dict = parent_dict.copy()

            child_dict = dict(child_el)
            child_dict_subset = {key: value for key, value in child_dict.items() if key in child_keys}

            temp_child_dict = child_dict_subset.copy()

            # Prefix as {child_list}_{child_key} if the key exists in parent_list
            for common_key in set(parent_keys).intersection(child_keys):
                temp_child_dict.pop(common_key)
                temp_child_dict[f"{child_list}_{common_key}"] = child_dict[common_key]

            # Append together
            parent_dict_subset.update(temp_child_dict)
            temp_list.append(parent_dict_subset)

    df = pd.DataFrame.from_dict(temp_list)

    # if keys:
    #     df = df[keys]

    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df


def set_dataframe_to_hxd(hxd_object, child: str, df: pd.DataFrame) -> None:
    """Pushes a dtaframe to HXD
    Intermediate rate walk scenarios require a DotDict object to be passed
    So that we can continue to use the dot notation thorughout code
    """

    temp_dict = df.to_dict("records")
    setattr(hxd_object, child, [DotDict(record) for record in temp_dict])


def set_hxd_values_from_dict(hxd_object, input_dict: dict):
    """Sets values to HXD given a dictionary and checks if the keys exists before assignment"""

    for key, value in input_dict.items():
        # When testing in VSCode some nodes / attributes might not be present
        # Hence hasattr will always be FALSE
        # Handled this via try/except but could equally define a parameter as import_type = mock/hx
        try:
            setattr(hxd_object, key, value)
        except:
            if hasattr(hxd_object, key):
                setattr(hxd_object, key, value)


def divide(numerator: float, denom: float):
    """safe divide that checks for non-zero denomninator"""

    if numerator is None or denom is None:
        return None
    elif denom != 0:
        return numerator / denom
    else:
        return None


def set_mult_values(
    values_to_set: List[str],
    obj_to_set,
    obj_to_fetch,
) -> None:
    """set mulitple values for object from an object"""

    for val in values_to_set:
        setattr(obj_to_set, val, getattr(obj_to_fetch, val, None))


def output_list(list_obj, output_name):
    """Helper function to output the rating results to csv
    Allows you to interrogate whether the calcs are working as expected
    Could be made generic to other HX raters
    Put in utilities common module
    """

    df = pd.DataFrame([dict(list_el) for list_el in list_obj])
    df.to_csv(f"library/editing/test_cases/{output_name}.csv", index=False)


def get_df_name(df):
    """Gets the name of the parameter table given a dataframe
    Note that df.name isn't populated so we can't use this attribute
    """

    # NOTE: when running in VSCode `dir()` returns the dunder methods
    keys = [key for key in dir(hx.params) if not key.startswith("__")]

    for key in keys:
        if getattr(hx.params, key).equals(df):
            return key


def lookup_func(lu_table, lu_id, lu_column, lu_value, default_return_value=0):
    """
    Function to standardise using lookup tables
    and catch errors when levels aren't in lookup table
    """

    # TODO: put this in common modules
    df_name = get_df_name(lu_table)

    lu_table_subset = lu_table[lu_table[lu_id] == lu_value][lu_column]

    # Either return the subset or a default value and validation error
    if len(lu_table_subset) == 0:

        # TODO: add this to either hxd.validations() or a list object - chat with Rob on this
        # hx.errors.validation(f"The level '{lu_value}' doesn't exist in the lookup table '{df_name}' and column '{lu_id}'")
        return default_return_value
    else:
        return lu_table_subset.iloc[0]


## All in a common module
def lookup_rels(rels_input_dict: dict) -> dict:
    """Gets the lookups given the provided model,dictionary containing arguments to the function and assigns back to HXD
    # Example dictionary
    # rels_input_dict = {
    #     "attHullHullArea1": {"lu_table": hull_area_lookup,"lu_id":"Area","lu_column": "Att Hull","lu_value": aircraft_grp.attHullHullArea},
    # }
    """

    rels_dict_output = {}
    for new_field, lookup_params in rels_input_dict.items():
        rels_dict_output[new_field] = lookup_func(**lookup_params)

    return rels_dict_output


def lookup_and_assign_rels(rels_input_dict: dict, hxd_object=None) -> dict:
    """Wrapper to lookup the relativities and then assign to HXD"""

    temp_dict = lookup_rels(rels_input_dict=rels_input_dict)

    if hxd_object is not None:
        set_hxd_values_from_dict(hxd_object=hxd_object, input_dict=temp_dict)

    # Return dictionaries in case values don't exist in HXD or further manipulation requried
    return temp_dict


def diff_month(d1, d2):
    """Calculates the months between 2 dates"""
    return (d1.year - d2.year) * 12 + d1.month - d2.month


# NOTE: assume that first column is always the index
# TODO: take more detailed code from HX's documentation
def lookup_as_dict(df, index_col):
    df_indexed = df.set_index(index_col)
    return df_indexed.to_dict("index")


# NOTE: already in SRM and tested
def agg_list(parent_obj, list_obj, keys):
    """Sums up keys from a nested layer to a higher layer"""

    # NOTE: doesn't allow for AIG share - loadings module does
    for key in keys:
        setattr(parent_obj, key, sum([getattr(el.ratingResult, key) for el in list_obj]))


def get_percentage(numerator: float, denom: float):
    """Get percentage checking for nones"""

    val = divide(numerator=numerator, denom=denom)

    if val is None:
        return None
    else:
        return val - 1


def safe_subtract(val1: float, val2: float):
    """Subtract fields allowing for nones"""

    if val1 is None or val2 is None:
        return None
    else:
        return val1 - val2


def camel_to_snake(string: str):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()


def get_target_source(target_str, source_str, key: str) -> tuple:
    """Helper to extract required keys from structures"""
    target = getattr(target_str, key)
    source = getattr(source_str, key)

    return target, source
