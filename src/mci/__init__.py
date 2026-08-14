"""mci — materials-curve-intel baseline.

Single-curve, single-plot chart data extraction:
structure detection -> tick OCR -> coordinate mapping -> curve extraction -> export.

The pipeline is deliberately modular so that the classical-CV baseline modules can be
replaced one-by-one by learned models (YOLOv8 detector, U-Net segmentation) without
changing the rest of the system. See README.md "Roadmap to multi-curve".
"""

__version__ = "0.1.0"
