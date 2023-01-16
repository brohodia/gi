import hx_data_schema as hx


@hx.data_schema
def data_schema():
    return hx.Structure(
        children={
            "a": hx.Float(mode="input", default=1, view={"label": "Input a"}),
            "b": hx.Float(mode="input", default=2, view={"label": "Input b"}),
            "c": hx.Float(mode="output", view={"label": "Output c"}),
        }
    )
