import numpy as np
import pytest

from app.config import AxisRange, PlotArea
from app.coordinate import pixel_to_data_point, pixel_to_data_points


@pytest.fixture
def setup_ranges():
    plot_area = PlotArea(left=100, top=50, right=500, bottom=450)
    x_range = AxisRange(min=0, max=1000)
    y_range = AxisRange(min=0, max=10)
    return plot_area, x_range, y_range


def test_corner_and_center_mapping(setup_ranges):
    plot_area, x_range, y_range = setup_ranges

    left_bottom = pixel_to_data_point(100, 450, plot_area, x_range, y_range)
    assert left_bottom["x"] == pytest.approx(0)
    assert left_bottom["y"] == pytest.approx(0)

    right_top = pixel_to_data_point(500, 50, plot_area, x_range, y_range)
    assert right_top["x"] == pytest.approx(1000)
    assert right_top["y"] == pytest.approx(10)

    center = pixel_to_data_point(300, 250, plot_area, x_range, y_range)
    assert center["x"] == pytest.approx(500)
    assert center["y"] == pytest.approx(5)


def test_y_axis_reversed(setup_ranges):
    plot_area, x_range, y_range = setup_ranges
    p_top = pixel_to_data_point(200, 50, plot_area, x_range, y_range)
    p_bottom = pixel_to_data_point(200, 450, plot_area, x_range, y_range)
    assert p_top["y"] > p_bottom["y"]


def test_batch_and_empty(setup_ranges):
    plot_area, x_range, y_range = setup_ranges
    pts = [(100, 450), (300, 250), (500, 50)]
    out = pixel_to_data_points(pts, plot_area, x_range, y_range)
    assert len(out) == 3

    arr = np.array(pts, dtype=float)
    out2 = pixel_to_data_points(arr, plot_area, x_range, y_range)
    assert len(out2) == 3

    assert pixel_to_data_points([], plot_area, x_range, y_range) == []


def test_invalid_inputs_raise(setup_ranges):
    _, x_range, y_range = setup_ranges
    bad_plot_area = PlotArea(left=0, top=0, right=1, bottom=1)
    with pytest.raises(ValueError):
        pixel_to_data_points([(1,)], bad_plot_area, x_range, y_range)

    with pytest.raises(ValueError):
        pixel_to_data_points([(1, "a")], bad_plot_area, x_range, y_range)

    class RawPlot:
        left, top, right, bottom = 1, 1, 1, 2

    with pytest.raises(ValueError):
        pixel_to_data_point(1, 1, RawPlot(), x_range, y_range)
