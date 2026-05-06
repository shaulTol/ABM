"""Shared style configuration for all paper figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt

def setup_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })

SSA_COLOR = '#2166ac'
MI_COLOR = '#b2182b'
SCATTER_COLOR = '#404040'
DIAGONAL_COLOR = '#999999'
TEXT_WIDTH = 5.5
