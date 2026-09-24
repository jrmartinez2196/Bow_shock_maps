"""Shared pytest fixtures for the bowshockmaps test suite."""

import matplotlib

# Use a non-interactive backend so importing the visualization module
# (which builds matplotlib widgets at instantiation time) never tries
# to open a window during tests.
matplotlib.use("Agg")
