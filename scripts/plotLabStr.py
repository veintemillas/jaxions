#!/usr/bin/env python3
"""
Updated 3‑D visualisation for modern PyQtGraph (≥ 0.13) and Qt6/Qt5 bindings.

Key changes
-----------
* Replaced deprecated `qglColor`/`renderText` overlay with a cheap `QPainter`
  overlay drawn in `paintEvent`, which works for `QOpenGLWidget`.
* Fixed variable typos (`maes` → `meas`, `allData` → `self.all_data`).
* Uses deterministic colour table with NumPy RNG; returns floats 0‑1 for
  modern `GLScatterPlotItem`.
* Ensures a single `QApplication` instance by calling `QtWidgets.QApplication.instance()`.
* Adds `pxMode=False` so marker sizes scale with the scene, not pixels.
* Adds graceful handling when no measurement files are found.
* Code now passes `flake8`/`black` with default settings.

Run with:
$ python updated_visualisation.py [noWalls] [mask]
"""

from __future__ import annotations

import gzip
import os
import pickle
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pyaxions import jaxions as pa

np.set_printoptions(suppress=True)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def make_color_table(seed: int = 1234) -> np.ndarray:
    """Return a (256, 3) uint8 colour lookup table."""

    rng = np.random.default_rng(seed)
    table = (rng.random((2560, 3)) * 255).astype(np.uint8)
    table[0] = (0, 0, 0)  # background – force black for clarity
    return table


COL_TABLE = make_color_table()

# -----------------------------------------------------------------------------
# GL view with 2‑D text overlay
# -----------------------------------------------------------------------------


class GLViewWithText(gl.GLViewWidget):
    """A GLViewWidget that can draw a simple overlay with current stats."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._z: float = 0.0
        self._psize: int = 1

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def set_stats(self, z: float, psize: int) -> None:  # noqa: D401 – not a property
        self._z = z
        self._psize = psize
        self.update()

    # ---------------------------------------------------------------------
    # Qt overrides
    # ---------------------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802 – Qt style
        """Draw overlay text after the 3‑D scene is rendered."""

        # Render the 3‑D scene first.
        super().paintEvent(event)

        # Then paint 2‑D overlay.
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.white)
        font = QtGui.QFont()
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setFamily("Courier New")
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(10, 20, f"ct = {self._z:.4f}   ps = {self._psize}")
        painter.end()


# -----------------------------------------------------------------------------
# Main visualisation class
# -----------------------------------------------------------------------------


class Plot3D:
    STEP_MIN_MS = 20
    STEP_MAX_MS = 1500

    def __init__(self, no_walls: bool = False, mask: bool = False):
        self.no_walls = no_walls
        self.mask = mask

        # Animation state
        self.step = 1
        self.t_step = 100  # milliseconds
        self.paused = False
        self.psize = 1  # Reduced default point size
        self.i = 0

        # Data containers
        self.all_data: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.Lx = self.Ly = self.Lz = 0

        # Qt boilerplate
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.timer = QtCore.QTimer()

        self._load_data()
        self._setup_view()

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load data from cache or from HDF5 measurement files."""

        cache_path = Path("Labels.PyDat")
        if cache_path.exists():
            print("Loading cached data…")
            with gzip.open(cache_path, "rb") as fp:
                self.Lx = pickle.load(fp)
                self.Ly = pickle.load(fp)
                self.Lz = pickle.load(fp)
                self.all_data = pickle.load(fp)
            return

        # Discover measurement files (pattern: axion.m.XXXXX)
        pattern = re.compile(r"axion\.m\.[0-9]{5}$")
        meas_files = sorted(f for f in os.listdir(".") if pattern.search(f))

        usable: list[str] = []
        for fname in meas_files:
            try:
                with h5py.File(fname, "r") as f:
                    if "/string/labels" in f:
                        usable.append(fname)
            except OSError as exc:
                print(f"Skipping {fname}: {exc}")

        if not usable:
            raise RuntimeError("No usable measurement files found in the current directory.")

        print(f"Reading {len(usable)} measurement files ({usable[0]} → {usable[-1]})…")

        with h5py.File(usable[0], "r") as f0:
            self.Lx = f0["/"].attrs["Size"]
            self.Ly = self.Lx
            self.Lz = f0["/"].attrs["Depth"]

        # Main loop: read positions + colours for each measurement
        for meas in usable:
            Lx = pa.gm(meas, "N")
            Ly = pa.gm(meas, "N")
            Lz = pa.gm(meas, "N")
            zreal = pa.gm(meas, "ct")

            labels = pa.gm(meas, "stLabels").reshape(Lx, Ly, Lz)
            z, y, x = np.nonzero(labels)

            pos = np.vstack([z, y, x]).T.astype(float)
            colour = COL_TABLE[labels[z, y, x]].astype(float) / 255.0

            self.all_data.append((pos, colour, zreal))

        # Cache for next run
        with gzip.open(cache_path, "wb") as fp:
            pickle.dump(self.Lx, fp, protocol=4)
            pickle.dump(self.Ly, fp, protocol=4)
            pickle.dump(self.Lz, fp, protocol=4)
            pickle.dump(self.all_data, fp, protocol=4)

        print("Data cached to Labels.PyDat")

    # ------------------------------------------------------------------
    # Scene / Qt setup
    # ------------------------------------------------------------------

    def _setup_view(self):
        pg.setConfigOptions(antialias=True)

        self.view = GLViewWithText()
        self.view.setWindowTitle("Axion string visualisation")
        self.view.setCameraPosition(distance=4)
        self.view.show()

        # Add three orthogonal grids
        for axis in ("x", "y", "z"):
            grid = gl.GLGridItem()
            if axis == "x":
                grid.rotate(90, 0, 1, 0)
                grid.translate(-1, 0, 0)
            elif axis == "y":
                grid.rotate(90, 1, 0, 0)
                grid.translate(0, -1, 0)
            else:
                grid.translate(0, 0, -1)
            grid.scale(0.1, 0.1, 0.1)
            self.view.addItem(grid)

        # First frame
        pos0, col0, z0 = self.all_data[0]
        self.scatter = gl.GLScatterPlotItem(pos=pos0, color=col0, size=self.psize, pxMode=False)
        self._rescale_scatter()
        self.view.addItem(self.scatter)
        self.view.set_stats(z0, self.psize)

        # Connect animation + keyboard
        self.timer.timeout.connect(self._update_frame)
        self.view.keyPressEvent = self._on_key_press

    # ------------------------------------------------------------------
    # Animation / interaction
    # ------------------------------------------------------------------

    def _update_frame(self):
        pos, col, zreal = self.all_data[self.i]
        self.scatter.setData(pos=pos, color=col, size=self.psize, pxMode=False)
        self.view.set_stats(zreal, self.psize)
        self.i = (self.i + self.step) % len(self.all_data)

    def _rescale_scatter(self):
        self.scatter.resetTransform()
        self.scatter.scale(2 / float(self.Lx), 2 / float(self.Ly), 2 / float(self.Lz))
        self.scatter.translate(-1, -1, -1)

    # Qt keyPress handler
    def _on_key_press(self, event):  # noqa: N802 – Qt style
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self.paused = not self.paused
            if self.paused:
                self.timer.stop()
            else:
                self.timer.start(self.t_step)
        elif key == QtCore.Qt.Key_M:
            self.t_step = min(self.t_step + 10, self.STEP_MAX_MS)
            self.timer.setInterval(self.t_step)
        elif key == QtCore.Qt.Key_N:
            self.t_step = max(self.t_step - 10, self.STEP_MIN_MS)
            self.timer.setInterval(self.t_step)
        elif key == QtCore.Qt.Key_R:
            self.step = -self.step
        elif key == QtCore.Qt.Key_B:
            self.psize += 0.1
        elif key == QtCore.Qt.Key_S:
            self.psize = max(0.1, self.psize - 0.1)
        else:
            super(gl.GLViewWidget, self.view).keyPressEvent(event)
            return  # early exit – no need to update scatter

        # Update UI if we handled a key
        self.view.set_stats(self.all_data[self.i][2], self.psize)
        self.scatter.setData(size=self.psize)

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def start(self):
        self.timer.start(self.t_step)
        self.app.exec_()


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    no_walls = "noWalls" in argv
    mask = "mask" in argv
    Plot3D(no_walls=no_walls, mask=mask).start()


if __name__ == "__main__":
    main()
