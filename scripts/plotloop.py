import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyaxions import jaxions as pa


def split_periodic_jumps_3d(coords, max_gap=2.0):
    """Split a loop into continuous pieces, avoiding fake lines across PBC jumps."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return [coords]

    jumps = np.where(np.linalg.norm(np.diff(coords, axis=0), axis=1) > max_gap)[0] + 1
    return [seg for seg in np.split(coords, jumps) if len(seg) > 1]


class GLViewWithText(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.text = ""

    def updateText(self, text):
        self.text = text
        self.update()

    def paintGL(self, *args, **kwds):
        super().paintGL(*args, **kwds)
        # self.qglColor(QtCore.Qt.white)
        # self.renderText(0, 0, 1.5, self.text)


class Loop3DViewer:
    def __init__(self, pa, mf, it0=0, max_gap=2.0, show_labels=False):
        self.pa = pa
        self.mf = mf
        self.it = it0
        self.max_gap = max_gap
        self.show_labels = show_labels

        self.tStep = 150
        self.step = 1
        self.pause = False

        self.line_width = 2
        self.point_size = 2

        self.items = []
        self.scatter = None

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.timer = QtCore.QTimer()

        pg.setConfigOptions(antialias=True)

        self.view = GLViewWithText()
        self.view.show()
        self.view.setWindowTitle("String loops")
        self.view.setCameraPosition(distance=3.5)

        self.add_grid()

        self.baseKeyPress = self.view.keyPressEvent
        self.view.keyPressEvent = self.keyPressEvent

        self.load_frame(self.it)

    def add_grid(self):
        for axis in range(3):
            g = gl.GLGridItem()
            g.scale(0.1, 0.1, 0.1)

            if axis == 0:
                g.rotate(90, 0, 1, 0)
                g.translate(-1, 0, 0)
            elif axis == 1:
                g.rotate(90, 1, 0, 0)
                g.translate(0, -1, 0)
            else:
                g.translate(0, 0, -1)

            self.view.addItem(g)

    def clear_items(self):
        for item in self.items:
            self.view.removeItem(item)
        self.items = []

        if self.scatter is not None:
            self.view.removeItem(self.scatter)
            self.scatter = None

    def normalize_pos(self, pos, N):
        """
        Convert simulation coordinates [0,N] to OpenGL box [-1,1].
        Input/output order is x,y,z.
        """
        return 2.0 * pos / float(N) - 1.0

    def load_frame(self, it):
        self.clear_items()

        f = self.mf[it]
        N = self.pa.gm(f, "N")
        loops_ok = self.pa.gm(f, "stloops?")

        if not loops_ok:
            self.view.updateText(f"it={it}: no stloops")
            return

        d = self.pa.gm(f, "stloops")
        ztime = self.pa.gm(f, "ct") # if self.pa.gm(f, "z?") else np.nan

        nloops = len(d["labels"])

        # Draw loop coordinates
        for i in range(nloops):
            coords = np.asarray(d["coords"][i], dtype=float)  # assumed x,y,z

            # One color per loop, RGBA in [0,1]
            hue = i / max(1, nloops)
            color = pg.mkColor(pg.intColor(i, hues=max(8, nloops)))
            rgba = np.array(color.getRgbF())

            for seg in split_periodic_jumps_3d(coords, self.max_gap):
                pos = self.normalize_pos(seg, N)
                item = gl.GLLinePlotItem(
                    pos=pos,
                    color=rgba,
                    width=self.line_width,
                    mode="line_strip",
                    antialias=True,
                )
                self.view.addItem(item)
                self.items.append(item)

        # Optional: overlay labelled voxels
        if self.show_labels and self.pa.gm(f, "stLabels?"):
            ld = np.reshape(self.pa.gm(f, "stLabels"), (N, N, N))

            z, y, x = np.nonzero(ld)
            pos = np.vstack([x, y, z]).T.astype(float)
            pos = self.normalize_pos(pos, N)

            colors = np.zeros((len(pos), 4))
            colors[:, :3] = 1.0
            colors[:, 3] = 0.08

            self.scatter = gl.GLScatterPlotItem(
                pos=pos,
                color=colors,
                size=self.point_size,
                pxMode=True,
            )
            self.view.addItem(self.scatter)

        self.view.updateText(
            f"file={it}/{len(self.mf)-1}  ct={ztime:.4g}  loops={nloops}  width={self.line_width}"
        )

    def update(self):
        self.it = (self.it + self.step) % len(self.mf)
        self.load_frame(self.it)

    def start(self):
        self.timer.timeout.connect(self.update)
        self.timer.start(self.tStep)

        # In Jupyter, run `%gui qt` before creating the viewer.
        if not hasattr(QtCore, "PYQT_VERSION") or QtWidgets.QApplication.instance() is None:
            self.app.exec_()

    def keyPressEvent(self, event):
        key = event.key()
        self.baseKeyPress(event)

        if key == QtCore.Qt.Key_Space:
            self.pause = not self.pause
            self.timer.stop() if self.pause else self.timer.start(self.tStep)

        elif key == QtCore.Qt.Key_M:
            self.tStep = min(self.tStep + 10, 1500)
            self.timer.setInterval(self.tStep)

        elif key == QtCore.Qt.Key_N:
            self.tStep = max(self.tStep - 10, 20)
            self.timer.setInterval(self.tStep)

        elif key == QtCore.Qt.Key_R:
            self.step = -self.step

        elif key == QtCore.Qt.Key_B:
            self.line_width += 1
            self.load_frame(self.it)

        elif key == QtCore.Qt.Key_S:
            self.line_width = max(1, self.line_width - 1)
            self.load_frame(self.it)

        elif key == QtCore.Qt.Key_Right:
            self.it = (self.it + 1) % len(self.mf)
            self.load_frame(self.it)

        elif key == QtCore.Qt.Key_Left:
            self.it = (self.it - 1) % len(self.mf)
            self.load_frame(self.it)


mf = pa.fm()
print(mf[0],mf[-1])

viewer = Loop3DViewer(
    pa,
    mf,
    it0=0,
    max_gap=2.0,
    show_labels=False,   # True if you also want labelled voxels
)

viewer.start()