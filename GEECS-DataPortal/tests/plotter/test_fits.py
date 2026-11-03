"""GEECSplotter's Fit menu: per Y list, the curve dashed in its colour, the numbers listed.

The fake Scan001 has three finite U_ICT charge shots, (1, 10), (1, 12) and
(2, 30), against the jet: the order-1 line through them is c1 = 19,
c0 = -8 (through the x = 1 mean, 11, and 30). Residuals -1, 1, 0 give
s² = 2 / (3 - 2) = 2; x̄ = 4/3, Sxx = 2/3, so se(c1) = sqrt(s²/Sxx) = √3
and se(c0) = sqrt(s²(1/3 + x̄²/Sxx)) = √6.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from geecs_data_utils.data.curve_fit import FitModel

from geecs_portal.plotter import figures
from geecs_portal.plotter.routes.api import plus_minus
from geecs_portal.plotter.settings import PlotterSettings, SettingsError, form_value
from tests.plotter.conftest import SCAN_VARIABLE, post

LINE = {"day": "2026-10-02", "scan": 1, "x": SCAN_VARIABLE, "y": "U_ICT charge"}


def figure(client: TestClient, **params: object) -> dict:
    response = client.get("/api/figure", params={**LINE, **params})
    assert response.status_code == 200, response.text
    return response.json()


def test_every_menu_entry_is_a_data_utils_model() -> None:
    assert set(figures.FITS) - {"none"} <= set(get_args(FitModel))


class TestFigures:
    SERIES = {"x": [1.0, 1.0, 2.0, 2.0], "a": [10.0, 12.0, math.nan, 30.0]}

    def test_fit_columns_fits_the_drawn_points(self) -> None:
        (fit,) = figures.fit_columns(
            self.SERIES, ["a"], ["polynomial"], x="x", orders=[1]
        )
        assert [p.value for p in fit.parameters] == pytest.approx([-8.0, 19.0])
        assert [p.stderr for p in fit.parameters] == pytest.approx(
            [math.sqrt(6), math.sqrt(3)]
        )

    def test_no_fit_and_an_off_list_are_none(self) -> None:
        assert figures.fit_columns(
            self.SERIES, ["a", None, "a"], ["none", "gaussian", None], x="x"
        ) == [None, None, None]

    def test_the_curve_is_dashed_on_its_slots_axis_in_its_colour(self) -> None:
        series = {**self.SERIES, "b": [1.0, 2.0, 3.0, 4.0]}
        ys = ["a", None, "b"]
        fits = figures.fit_columns(
            series, ys, [None, None, "polynomial"], x="x", orders=[2, 2, 1]
        )
        fig = figures.shots_figure(series, ys, x="x", fits=fits)
        points_a, points_b, curve = fig.data
        assert curve.mode == "lines" and curve.line.dash == "dash"
        assert curve.yaxis == "y3" == points_b.yaxis
        assert curve.line.color == figures.NOTEBOOK_PALETTE.trace[2]
        assert curve.name == "b: Polynomial fit"
        assert (curve.x[0], curve.x[-1]) == (1.0, 2.0)
        assert len(curve.x) == figures.FIT_POINTS
        # b = 1, 2 at x = 1 and 3, 4 at x = 2: the line through the means.
        assert (curve.y[0], curve.y[-1]) == pytest.approx((1.5, 3.5))

    def test_a_fit_that_could_not_be_made_draws_nothing(self) -> None:
        fits = figures.fit_columns(self.SERIES, ["a"], ["gaussian"], x="x")
        assert fits[0].error
        fig = figures.shots_figure(self.SERIES, ["a"], x="x", fits=fits)
        assert len(fig.data) == 1


class TestApi:
    def test_the_fit_is_listed_with_its_errors(self, client: TestClient) -> None:
        body = figure(client, yfit="polynomial", yorder=1)
        (fit,) = body["fits"]
        assert (fit["slot"], fit["column"], fit["label"]) == (
            1,
            "U_ICT charge",
            "Polynomial",
        )
        assert fit["formula"] == "c0 + c1·x"
        assert fit["error"] is None and fit["points"] == 3
        c0, c1 = fit["terms"]
        assert (c0["name"], c0["value"]) == ("c0", pytest.approx(-8.0))
        assert c1["stderr"] == pytest.approx(math.sqrt(3))
        assert (c0["text"], c1["text"]) == ("-8.0 ± 2.4", "19.0 ± 1.7")
        # r2: SSR 2 over SST = 1 + 1 + 361 - 3·(52/3 - 11)²… worked by numpy.
        assert 0 < fit["r2"] < 1

    def test_the_curve_is_drawn_in_the_themed_colour(self, client: TestClient) -> None:
        body = figure(client, yfit="polynomial", yorder=1)
        points, curve = body["figure"]["data"]
        assert curve["line"] == {
            "color": "$tok:--trace-1",
            "dash": "dash",
            "width": 2,
        }
        assert curve["y"][0] == pytest.approx(11.0)

    def test_no_fit_is_no_block_and_no_curve(self, client: TestClient) -> None:
        body = figure(client)
        assert body["fits"] == [] and len(body["figure"]["data"]) == 1
        assert "fit" not in body["code"]

    def test_a_fit_that_cannot_be_made_says_why(self, client: TestClient) -> None:
        (fit,) = figure(client, yfit="gaussian")["fits"]
        assert "4 distinct x" in fit["error"] and fit["terms"] == []

    def test_a_fit_this_copy_lacks_is_reported_not_fitted(
        self, client: TestClient
    ) -> None:
        body = figure(client, yfit="lorentzian")
        (fit,) = body["fits"]
        assert "no fit called 'lorentzian'" in fit["error"]
        assert len(body["figure"]["data"]) == 1

    def test_a_fit_on_y2_is_slot_two(self, client: TestClient) -> None:
        body = figure(client, y2="U_ICT charge", y2fit="polynomial", y2order=1)
        (fit,) = body["fits"]
        assert fit["slot"] == 2
        assert body["figure"]["data"][-1]["line"]["color"] == "$tok:--trace-2"

    def test_a_fit_on_a_list_that_is_off_is_ignored(self, client: TestClient) -> None:
        assert figure(client, y2fit="polynomial")["fits"] == []

    def test_an_order_out_of_range_is_refused(self, client: TestClient) -> None:
        response = client.get("/api/figure", params={**LINE, "yorder": 11})
        assert response.status_code == 422

    def test_the_code_rebuilds_the_fit(self, client: TestClient, share: Path) -> None:
        body = figure(client, yfit="polynomial", yorder=1)
        code = body["code"].replace(
            "experiment='Testexp'", f"experiment='Testexp', base_path={str(share)!r}"
        )
        assert "orders=[1]" in code and "fits=fits" in code
        assert "filter(None, fits)" in code  # no None printed for an unfitted list
        namespace: dict = {}
        exec(
            code.replace("fig.show()\n", "").replace("    print(fit)", "    pass"),
            namespace,
        )
        rebuilt = figures.page_figure(namespace["fig"])["data"][1]["y"]
        assert rebuilt == pytest.approx(body["figure"]["data"][1]["y"])
        assert namespace["fits"][0].value("c1") == pytest.approx(19.0)


def test_a_shifted_polynomial_says_so_in_its_formula() -> None:
    from geecs_data_utils.data.curve_fit import fit_curve

    from geecs_portal.plotter.routes.api import fit_result

    t = [1.7e9 + k for k in range(10)]
    fit = fit_curve(t, [k * k for k in range(10)], "polynomial", order=2)
    shown = fit_result(1, "y", "polynomial", fit)
    assert shown.formula == "c0 + c1·(x − 1700000004.5) + c2·(x − 1700000004.5)^2"


@pytest.mark.parametrize(
    ("value", "stderr", "text"),
    [
        (22.9901, 0.4463, "22.99 ± 0.45"),
        (2517.2, 662.6, "2520 ± 660"),
        (-8.0, math.sqrt(6), "-8.0 ± 2.4"),
        (1.5, math.nan, "1.5"),
        (1.5, 0.0, "1.5"),
        (math.nan, 1.0, "undefined"),
        (3.2e-9, 1.1e-10, "3.20e-09 ± 1.1e-10"),
        # Review of #1175: an unconstrained term keeps its own digits...
        (38.95, 3217.0, "39 ± 3200"),
        (-0.4, 2.0e4, "-0.4 ± 20000"),
        # ...and a large value keeps every digit its error makes meaningful.
        (1234567.0, 5.0, "1234567.0 ± 5.0"),
        (1.2345678e12, 3.3e9, "1.2346e+12 ± 3.3e+09"),
        # An error at float noise (a constant column) shows no noise digits.
        (6.0000000000000044, 6.6e-15, "6.00000000000000e+00 ± 6.6e-15"),
    ],
)
def test_plus_minus(value: float, stderr: float, text: str) -> None:
    assert plus_minus(value, stderr) == text


class TestSettings:
    def test_the_fit_defaults_to_none(self) -> None:
        settings = PlotterSettings()
        assert (settings.y_fit, settings.y2_fit, settings.y3_fit) == ("none",) * 3
        assert settings.y_fit_order == 2

    def test_a_fit_pick_saves_into_the_set(
        self, client: TestClient, settings_dir: Path
    ) -> None:
        post(client, scan=1, yfit="polynomial", yorder="3")
        saved = client.get("/api/settings/default").json()
        assert (saved["y_fit"], saved["y_fit_order"]) == ("polynomial", 3)

    @pytest.mark.parametrize(
        ("field", "value"), [("yorder", "11"), ("yorder", "-1"), ("yfit", "../x")]
    )
    def test_a_bad_fit_field_is_refused(self, field: str, value: str) -> None:
        with pytest.raises(SettingsError):
            form_value(field, value)

    def test_a_newer_copys_fit_name_is_kept(self) -> None:
        assert form_value("y2fit", "lorentzian") == "lorentzian"


class TestPage:
    def test_each_y_list_has_its_fit_menu(self, client: TestClient) -> None:
        html = post(client, scan=1, ycount="2", y2fit="gaussian")
        assert 'name="yfit"' in html and 'name="y2fit"' in html
        assert '<option value="gaussian" selected>Gaussian</option>' in html
        assert "y2fit=gaussian" in html  # the figure URL carries it
        assert 'name="yorder"' not in html  # only a polynomial has an order

    def test_a_polynomial_shows_its_order(self, client: TestClient) -> None:
        html = post(client, scan=1, yfit="polynomial", yorder="4")
        assert 'name="yorder"' in html and 'value="4"' in html
        assert "yorder=4" in html

    def test_a_fit_this_copy_lacks_stays_picked(self, client: TestClient) -> None:
        html = post(client, scan=1, yfit="lorentzian")
        assert "lorentzian (not in this version)" in html

    def test_each_fit_menu_is_above_its_list(self, client: TestClient) -> None:
        # Under a 10-row list the menu fell past the bottom of the rail, where
        # the author never saw it.
        html = post(client, scan=1, ycount="2")
        for key in ("y", "y2"):
            assert html.index(f'name="{key}fit"') < html.index(
                f'id="{key}" name="{key}"'
            )
