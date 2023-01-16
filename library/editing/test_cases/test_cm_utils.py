import numpy as np
import pandas as pd
import pytest
from algorithms.unit_test.utils import Util
import libraries.cm_testing.algorithms.utils as cm_test
from libraries.cm_testing.algorithms.DotDict import DotDict
from typing import List, Union, Any

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


@pytest.fixture(scope="module", name="hxd")
def load_hxd() -> DotDict:
    raw = Util.load_json("library/editing/test_cases/test_data.json")

    hxd = DotDict(raw)
    yield hxd


keys = [
    "groupId",
    "operator",
    "aircraftTypeSummary",
    "aircraftUse",
    "ageGroup",
    "operatorCountry",
    "aircraftCt",
    "selectedPerCraftExpDep",
    "selectedPerCraftAvgSeats",
    "selectedPerCraftUsedAv",
]

rename_dict = {"groupId": "groupId"}


@pytest.fixture(scope="module", name="run_data")
def run_data(hxd):
    """Runs through the module with mockHX object"""
    scen_inputs = hxd.insuredInterest
    nes_inputs = hxd.lossHistory


def test_convert_single_layer_list_to_dataframe(hxd, run_data):
    in_input = hxd.insuredInterest
    test_df = cm_test.convert_single_layer_list_to_dataframe(
        list_obj=in_input.aircraftGroupAirlines, keys=keys, rename_dict=rename_dict
    )

    assert type(test_df) == pd.DataFrame


agg_exp_keys = ["lossHistoryYear"]
claim_segment_keys = [
    "claimSegmentCd",
    "indemnityIncurredAmt",
    "aircraftCt",
    "cdf",
    "trend",
    "indemnityIncurredAmtAttrUlt",
    "lossCostAttr",
]


def test_convert_nested_list_to_dataframe(hxd, run_data):
    nested_input = hxd.lossHistory
    test_df = cm_test.convert_nested_list_to_dataframe(
        parent_list=nested_input.aggExperience,
        child_list="claimSegment",
        parent_keys=agg_exp_keys,
        child_keys=claim_segment_keys,
    )

    assert type(test_df) == pd.DataFrame


def test_set_mult_values():
    class testclass1:
        def __init__(self):
            self.name = "Alice"
            self.age = 25
            self.height = 150

    class testclass2:
        def __init__(self):
            self.name = "Bob"
            self.age = 30

    obj1 = testclass1()
    obj2 = testclass2()
    values_to_set = ["name", "age"]

    cm_test.set_mult_values(values_to_set, obj1, obj2)
    assert obj1.name == "Bob"
    assert obj1.age == 30
    assert obj1.height == 150


def test_camel_to_snake() -> None:
    snake_case = cm_test.camel_to_snake("dummyStringValue")
    assert snake_case == "dummy_string_value"


def test_agg_list() -> None:
    test_dict = DotDict(
        {
            "ratingResult": {},
            "policyStructure": [{"ratingResult": {"test_value": 1}}, {"ratingResult": {"test_value": 2}}],
        }
    )

    cm_test.agg_list(parent_obj=test_dict.ratingResult, list_obj=test_dict.policyStructure, keys=["test_value"])

    assert test_dict.ratingResult.test_value == 3


def test_get_target_source() -> None:
    """Checks that we can extract the same attribute from 2 different object"""
    # Setup test data
    target_str = Util.dict2obj({"test_val": 1})
    source_str = Util.dict2obj({"test_val": 2})
    key = "test_val"

    # Run function
    output = cm_test.get_target_source(target_str=target_str, source_str=source_str, key=key)

    # Check
    assert output == (1, 2)


def test_divide() -> None:
    assert cm_test.divide(4, 2) == 2
    assert cm_test.divide(4, None) == None
    assert cm_test.divide(None, None) == None
    assert cm_test.divide(4, 0) == None


def test_get_percentage() -> None:
    assert cm_test.get_percentage(2, 1) == 1
    assert cm_test.get_percentage(4, None) == None
    assert cm_test.get_percentage(None, None) == None
    assert cm_test.get_percentage(4, 0) == None


def test_safe_subtract() -> None:
    assert cm_test.safe_subtract(2, 1) == 1
    assert cm_test.safe_subtract(4, None) == None
    assert cm_test.safe_subtract(None, 2) == None


def test_set_dataframe_to_hxd() -> None:

    # Setup test data
    hxd = Util.dict2obj({"parent": {"child_list": []}})
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    # Run function
    cm_test.set_dataframe_to_hxd(hxd_object=hxd.parent, child="child_list", df=df)

    # Expected values
    expected_vals = Util.dict2obj({"parent": {"child_list": [{"x": 1, "y": 4}, {"x": 2, "y": 5}, {"x": 3, "y": 6}]}})

    # Check - convert to DotDict - otherwise it is check that the object IDs are identical
    assert DotDict(hxd) == DotDict(expected_vals)


def test_output_list():
    # Sample list and output name
    test_list = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    output_name = "test_output"

    # Call the function
    cm_test.output_list(test_list, output_name)

    # Assert that the CSV file was created and has the correct contents
    df = pd.read_csv(f"library/editing/test_cases/{output_name}.csv")
    assert df.equals(pd.DataFrame(test_list))
