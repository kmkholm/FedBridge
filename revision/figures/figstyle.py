"""Shared matplotlib style for the V2 figures (matches the v1 figures: serif text, muted palette)."""
import matplotlib as mpl

C = dict(blue="#4C72B0", red="#C44E52", green="#55A868", purple="#8172B2",
         orange="#DD8452", grey="#8C8C8C", dark="#2B2B2B", gold="#C9A227")


def setup():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
        "axes.titlelocation": "left", "axes.labelsize": 9, "legend.fontsize": 7.5,
        "legend.frameon": False, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
