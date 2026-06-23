"""Temporal (clip-level) safety-behaviour modelling.

This package adds *motion-aware* perception on top of the per-frame YOLOv8
classifier. The dataset is clip-level labelled (no bounding boxes), so the
natural upgrade is video/temporal classification rather than bbox detection.

- `model.py`   : the temporal models (embedding+head, and a 3D-conv video net)
- `dataset.py` : a torch Dataset that decodes K frames per clip from the mp4s
"""
