#!/usr/bin/python3

import argparse
import os
import sys

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = os.path.join(os.environ.get("TMPDIR", "/tmp"), "jaxions-matplotlib")
    os.makedirs(mpl_cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_cache

import numpy as np

from pyaxions.tunelog import read_tune_file, select_run


def _axis_label(run):
    bits = []
    for key in ("field", "device", "lattice", "tuner_lattice", "mpi_ranks", "omp_threads", "Ng"):
        if key in run.header:
            bits.append(f"{key}={run.header[key]}")
    return "  ".join(bits)


def _set_block_axis(ax, values, label):
    vals = sorted(set(values))
    ax.set_xlabel(label)
    if vals and min(vals) > 0:
        ax.set_xscale("log", base=2)
        ax.set_xticks(vals)
        ax.set_xticklabels([str(v) for v in vals], rotation=45, ha="right")


def plot_tune_run(run, output=None, show=False, include_predicted=True):
    import matplotlib.pyplot as plt

    points = run.valid_points()
    if not include_predicted:
        points = [p for p in points if not p.predicted]

    if not points:
        raise ValueError("selected tune block has no valid timing points")

    zvals = sorted(set(p.bz for p in points))
    ncols = min(3, len(zvals))
    nrows = int(np.ceil(len(zvals) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.4 * nrows),
                             squeeze=False, constrained_layout=True)

    times_ms = np.array([p.time_ns for p in points], dtype=float) / 1.0e6
    vmin = float(np.min(times_ms))
    vmax = float(np.max(times_ms))
    best = run.best_point()
    scatter = None

    for ax, bz in zip(axes.flat, zvals):
        layer = [p for p in points if p.bz == bz]
        xs = np.array([p.bx for p in layer])
        ys = np.array([p.by for p in layer])
        cs = np.array([p.time_ns for p in layer], dtype=float) / 1.0e6
        markers = np.array([p.predicted for p in layer])

        real = ~markers
        if np.any(real):
            scatter = ax.scatter(xs[real], ys[real], c=cs[real], vmin=vmin, vmax=vmax,
                                 s=70, cmap="viridis", edgecolors="black", linewidths=0.35,
                                 label="coarse")
        if np.any(markers):
            scatter = ax.scatter(xs[markers], ys[markers], c=cs[markers], vmin=vmin, vmax=vmax,
                                 s=90, cmap="viridis", marker="D", edgecolors="black",
                                 linewidths=0.45, label="interp/neighbour")

        if best is not None and best.bz == bz:
            ax.scatter([best.bx], [best.by], marker="*", s=260,
                       facecolors="none", edgecolors="red", linewidths=1.8,
                       label="best")

        ax.set_title(f"bz = {bz}")
        ax.set_ylabel("by")
        if xs.size:
            _set_block_axis(ax, xs, "bx")
        if ys.size and np.min(ys) > 0:
            yvals = sorted(set(int(y) for y in ys))
            ax.set_yscale("log", base=2)
            ax.set_yticks(yvals)
            ax.set_yticklabels([str(v) for v in yvals])
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes.flat[len(zvals):]:
        ax.axis("off")

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.92)
        cbar.set_label("time [ms]")

    title = "Jaxions propagator tuner"
    if best is not None:
        title += f"  best={best.bx}x{best.by}x{best.bz}  {best.time_ns/1.0e6:.3f} ms"
    fig.suptitle(title, fontsize=13)

    subtitle = _axis_label(run)
    if subtitle:
        fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    if output:
        fig.savefig(output, dpi=180)
        print(f"wrote {output}")
    if show or not output:
        plt.show()


def print_summary(run, index_label):
    best = run.best_point()
    print(f"selected run: {index_label}")
    for key in ("field", "device", "lattice", "tuner_lattice", "mpi_ranks", "omp_threads", "Ng"):
        if key in run.header:
            print(f"{key}: {run.header[key]}")
    print(f"points: {len(run.points)} total, {len(run.valid_points())} valid")
    if best is not None:
        print(f"best point from rows: {best.bx} {best.by} {best.bz} "
              f"threads={best.threads} time_ms={best.time_ns/1.0e6:.6f}")
    if run.conclusion:
        print("conclusion:")
        for key in ("stop", "evals", "best_block", "best_threads", "best_time_ns", "cache_written"):
            if key in run.conclusion:
                print(f"  {key}: {run.conclusion[key]}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot Jaxions adaptive propagator tuner timings from tune.txt")
    parser.add_argument("tune_file", nargs="?", default="tune.txt",
                        help="tune.txt file to read")
    parser.add_argument("-r", "--run", default="last",
                        help="run to plot: last, best, or integer index")
    parser.add_argument("-o", "--output", default=None,
                        help="output image path, e.g. tuneplot.png")
    parser.add_argument("--show", action="store_true",
                        help="show the matplotlib window after saving")
    parser.add_argument("--coarse-only", action="store_true",
                        help="hide interpolated/neighbour points")
    parser.add_argument("--summary", action="store_true",
                        help="only print a text summary, no plot")
    args = parser.parse_args(argv)

    if not os.path.exists(args.tune_file):
        raise FileNotFoundError(args.tune_file)

    runs = read_tune_file(args.tune_file)
    run = select_run(runs, args.run)
    print_summary(run, args.run)

    if not args.summary:
        plot_tune_run(run, output=args.output, show=args.show,
                      include_predicted=not args.coarse_only)


if __name__ == "__main__":
    main(sys.argv[1:])
