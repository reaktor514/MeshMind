"""Vendored TripoSR (https://github.com/VAST-AI-Research/TripoSR), MIT licensed.

Patched for MeshMind:
- ``models/isosurface.py`` uses ``skimage.measure.marching_cubes`` instead of
  the optional ``torchmcubes`` C++ extension.
- ``utils.py`` makes ``rembg`` and ``imageio`` optional/lazy imports and
  rewrites ``tsr.<...>`` class paths to ``meshmind._tsr.<...>``.
"""
