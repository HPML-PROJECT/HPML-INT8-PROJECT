"""Plots for triton-v3: reuses the WS1 plotting code (baseline-benchmark/plot_results.py) so there is one copy.
bench_rmsnorm_int8.py imports make_plots from here."""
import importlib.util
import os

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baseline-benchmark", "plot_results.py")
_spec = importlib.util.spec_from_file_location("_ws1_plot_results", _src)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_plots = _mod.make_plots
