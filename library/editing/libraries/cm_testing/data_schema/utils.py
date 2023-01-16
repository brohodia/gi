from typing import Callable

try:
    import hx_data_schema as hx
except:
    print("Failed to import HX Data Schema")


def thousands_format(mantissa: int = 0) -> dict:
    return {"thousandSeparated": True, "mantissa": mantissa}


def percent_format(mantissa: int = 0) -> dict:
    return {"output": "percent", "mantissa": mantissa}


def no_comma(mantissa: int = 0) -> dict:
    return {"thousandSeparated": False, "mantissa": mantissa}


def read_only() -> dict:
    return {"read_only": {"read_only": True}}
