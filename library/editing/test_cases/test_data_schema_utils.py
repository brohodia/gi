import libraries.cm_testing.data_schema.utils as ds_util

# TODO: add in checks to thousands_format to check that it is an integer
def test_thousands_format() -> None:
    assert ds_util.thousands_format(2) == {"thousandSeparated": True, "mantissa": 2}


def test_percent_format() -> None:
    assert ds_util.percent_format(2) == {"output": "percent", "mantissa": 2}


def test_no_comma() -> None:
    assert ds_util.no_comma(2) == {"thousandSeparated": False, "mantissa": 2}
