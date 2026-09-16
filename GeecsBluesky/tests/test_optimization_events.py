"""Optimization columns survive event-model and Tiled SQL storage."""

import math

import pytest

from geecs_bluesky.optimization_events import optimization_column, optimization_name


@pytest.mark.parametrize(
    "name",
    [
        "U_EMQTripletBipolar:Current_Limit.Ch1",
        "cam.charge",
        "a/b",
        "a%2E",
        "a~2E",
        "a--b",
        "a+b=(c)",
        "énergie",
        "a_b",
        "a.b",
    ],
)
def test_column_round_trip(name):
    column = optimization_column("output", name)
    assert optimization_name(column.split(":", 1)[1]) == name
    assert not any(char in column for char in ".%/+-=()")


def test_columns_archive_through_tiled_sql(tmp_path):
    pytest.importorskip("tiled.server.app")
    from bluesky import RunEngine, plans
    from bluesky.callbacks.tiled_writer import TiledWriter
    from tiled.catalog import in_memory
    from tiled.config import Authentication
    from tiled.client import from_context
    from tiled.client.context import Context
    from tiled.server.app import build_app
    from geecs_bluesky.plans.optimize import OptimizationRecord

    name = "U_EMQTripletBipolar:Current_Limit.Ch1"
    columns = [optimization_column(role, name) for role in ("measured", "best_move")]
    record = OptimizationRecord(columns)
    record.values[columns[0]] = 1.4
    tree = in_memory(writable_storage=[f"sqlite:///{tmp_path / 'tables.db'}"])
    app = build_app(tree, authentication=Authentication(single_user_api_key="test"))
    with Context.from_app(app, api_key="test") as context:
        client = from_context(context)
        engine = RunEngine()
        engine.subscribe(TiledWriter(client))  # No safe wrapper hiding a failure.
        (uid,) = engine(plans.count([record], num=2))
        table = client[uid]["primary"].base["internal"].read()
        assert list(table[columns[0]]) == [1.4, 1.4]
        assert all(math.isnan(value) for value in table[columns[1]])
        assert client[uid].metadata["stop"]["exit_status"] == "success"
