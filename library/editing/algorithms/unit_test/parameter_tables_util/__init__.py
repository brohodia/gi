import csv
import datetime
import math
import os
from typing import Any, Dict, List


def model_path(relative_path_in_model: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", relative_path_in_model)


class ParameterTable:
    _schema_type_to_python_type = {
        "str": str,
        "int": int,
        "bool": bool,
        "date": datetime.date,
        "float": float,
    }

    def __init__(self, data: List[Dict[str, Any]], schema: Dict[str, str]):
        self.data = data
        self.schema = schema

    @staticmethod
    def from_csv(path: str, schema: Dict[str, str]) -> "ParameterTable":
        for column, column_type in schema.items():
            if column_type not in ParameterTable._schema_type_to_python_type:
                raise ValueError(
                    f"Schema column '{column}' is of invalid type '{column_type}'\n"
                    f"Allowed types: {list(ParameterTable._schema_type_to_python_type)}"
                )

        column_names: List[str]
        data = []

        try:
            with open(model_path(path), newline="", encoding="utf-8-sig") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",", quotechar='"')

                for row_index, row in enumerate(csv_reader):
                    row_data = {}
                    if row_index == 0:
                        if set(schema.keys()) != set(row):
                            raise ValueError(
                                f"Parameter table '{path}' csv file does not contain expected column names\n"
                                f"Expected columns: {sorted(list(schema.keys()))}\n"
                                f"Observed columns: {sorted(row)}"
                            )
                        column_names = row
                        continue

                    if len(column_names) != len(row):
                        raise ValueError(
                            f"Parameter table '{path}' row {row_index + 1} does not contain expected number of values\n"
                            f"Expected number: {len(column_names)}\n"
                            f"Observed number: {len(row)}"
                        )

                    for col_name, raw_value in zip(column_names, row):
                        col_type = ParameterTable._schema_type_to_python_type[schema[col_name]]

                        if raw_value == "" and col_type != str:
                            raise ValueError(
                                f"Parameter table 'table_name' row {row_index + 1}: column '{col_name}' contains no data"
                            )

                        try:
                            if col_type == datetime.date:
                                row_data[col_name] = datetime.date.fromisoformat(raw_value)
                            elif col_type == bool:
                                if raw_value.upper() == "TRUE":
                                    row_data[col_name] = True
                                elif raw_value.upper() == "FALSE":
                                    row_data[col_name] = False
                                else:
                                    raise ValueError("needs to be 'true' or 'false'")
                            else:
                                new_value = col_type(raw_value)
                                if isinstance(new_value, float) and not math.isfinite(new_value):
                                    raise ValueError("value not finite")
                                row_data[col_name] = new_value
                        except ValueError as e:
                            raise ValueError(
                                f"Parameter table '{path}' row {row_index + 1}: column '{col_name}' ('{raw_value}') is invalid: {e}"
                            )

                    data.append(row_data)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot read parameter table file '{path}': {e}")
        return ParameterTable(data, schema)

    def column(self, column: str) -> List[Any]:
        if column not in self.schema:
            raise KeyError(f"Column '{column}' not found. Available columns: {list(self.schema)}")
        return [row[column] for row in self.data]

    def columns(self, *columns: str) -> List[Any]:
        for column in columns:
            if column not in self.schema:
                raise KeyError(f"Column '{column}' not found. Available columns: {list(self.schema)}")
        return [[row[column] for column in columns] for row in self.data]

    def df(self):
        try:
            import pandas
        except ImportError:
            raise ImportError("Install a version of pandas to access parameter table as Pandas data frame")

        dataframe = pandas.DataFrame(self.data)
        for column_name, column_type in self.schema.items():
            if column_type == "date":
                dataframe[column_name] = dataframe[column_name].astype("datetime64[D]")
        return dataframe