import numpy as np

from app.config import AxisRange
from app.coordinate import pixel_to_data


def test_coordinate_mapping_basic():
    pts = np.array([[0, 0], [9, 9]], dtype=float)
    out = pixel_to_data(pts, width=10, height=10, x_range=AxisRange(min=0, max=1), y_range=AxisRange(min=0, max=1))
    assert np.allclose(out[0], [0, 1])
    assert np.allclose(out[1], [1, 0])


def test_coordinate_todo_placeholder():
    # TODO: add more rigorous coordinate mapping tests for non-linear axes.
    assert True
