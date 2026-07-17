#!/usr/bin/env python3

import os
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


def find_spectrum(argument):
    candidates = []
    if argument:
        candidates.append(argument)
    candidates.extend([
        "out/cylStringSpectrum.dat",
        "./cylStringSpectrum.dat",
        "../out/cylStringSpectrum.dat",
    ])
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find cylStringSpectrum.dat; pass its path to "
        "`jaxi cylStringSpectrum <file>`."
    )


path = find_spectrum(sys.argv[1] if len(sys.argv) > 1 else None)
table = np.loadtxt(path)
if table.ndim != 2 or table.shape[1] < 3:
    raise ValueError(f"{path} does not have the expected bin/k/power columns")

# New files carry the Jaxions-compatible RMS momentum in column 7.  Keep the
# nominal bin centre fallback so older spectrum files remain readable.
k = table[:, 6] if table.shape[1] >= 7 else table[:, 1]
power = table[:, 2]
valid = np.isfinite(k) & np.isfinite(power) & (k > 0) & (power > 0)
k = k[valid]
power = power[valid]
if len(k) < 4:
    raise ValueError(f"{path} contains too few positive finite spectrum bins")

pg.setConfigOptions(antialias=True)
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
window = pg.GraphicsLayoutWidget(title="Cylindrical moving-string spectrum")
window.resize(1050, 760)
window.setWindowTitle("jaxions cylindrical string spectrum")

power_plot = window.addPlot(title="Power spectrum")
power_plot.showGrid(x=True, y=True, alpha=0.25)
power_plot.setLogMode(x=True, y=True)
power_plot.setLabel("bottom", "k/k0")
power_plot.setLabel("left", "P(k)")
power_curve = power_plot.plot(k, power, pen=pg.mkPen("w", width=2), name="P(k)")
reference_curve = power_plot.plot(
    [], [], pen=pg.mkPen((255, 180, 50), width=2, style=QtCore.Qt.DashLine),
    name="1/k",
)
power_plot.addLegend()

window.nextRow()
compensated_plot = window.addPlot(title="Compensated spectrum", xLink=power_plot)
compensated_plot.showGrid(x=True, y=True, alpha=0.25)
compensated_plot.setLogMode(x=True, y=False)
compensated_plot.setLabel("bottom", "k/k0")
compensated_plot.setLabel("left", "k P(k)")
compensated_plot.plot(k, k*power, pen=pg.mkPen((80, 190, 255), width=2))
plateau_curve = compensated_plot.plot(
    [], [], pen=pg.mkPen((255, 180, 50), width=2, style=QtCore.Qt.DashLine)
)

initial_min = max(4.0, float(k[0]))
initial_max = min(float(k[-1]), max(initial_min*2.0, float(k[-1])/4.0))
fit_region = pg.LinearRegionItem(
    # Non-curve graphics items use ViewBox coordinates; x is log10 here.
    values=(np.log10(initial_min), np.log10(initial_max)),
    orientation=pg.LinearRegionItem.Vertical,
    brush=pg.mkBrush(100, 100, 160, 45),
    movable=True,
)
fit_region.setZValue(-10)
compensated_plot.addItem(fit_region)

fit_label = pg.LabelItem(justify="left")
window.addItem(fit_label, row=2, col=0)


def update_fit():
    log_lower, log_upper = sorted(fit_region.getRegion())
    lower, upper = 10.0**log_lower, 10.0**log_upper
    selected = (k >= lower) & (k <= upper)
    count = int(np.count_nonzero(selected))
    if count < 2:
        fit_label.setText("Select at least two positive bins.", color="#ff7777")
        reference_curve.setData([], [])
        plateau_curve.setData([], [])
        return

    slope, intercept = np.polyfit(np.log(k[selected]), np.log(power[selected]), 1)
    normalization = np.exp(np.mean(np.log(power[selected]) + np.log(k[selected])))
    reference_curve.setData(k, normalization/k)
    plateau_curve.setData([k[0], k[-1]], [normalization, normalization])
    fit_label.setText(
        f"file: {path}    fit: {lower:.3g} < k/k0 < {upper:.3g}    "
        f"bins: {count}    slope: {slope:.6f}    deviation from -1: {slope + 1.0:+.6f}",
        color="#ffffff",
    )
    power_plot.setTitle(f"Power spectrum — fitted slope {slope:.4f}")


fit_region.sigRegionChanged.connect(update_fit)
update_fit()
window.show()

if __name__ == "__main__":
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec_()
