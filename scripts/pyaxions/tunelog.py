#!/usr/bin/python3

from dataclasses import dataclass, field
import math


@dataclass
class TunePoint:
    eval_id: int
    level: int
    region: int
    source: str
    ix: int
    iy: int
    iz: int
    bx: int
    by: int
    bz: int
    threads: int
    time_ns: int
    valid: bool
    predicted: bool


@dataclass
class TuneRun:
    header: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)
    batches: list = field(default_factory=list)
    points: list = field(default_factory=list)
    conclusion: dict = field(default_factory=dict)

    def valid_points(self):
        return [p for p in self.points if p.valid and math.isfinite(p.time_ns)]

    def best_point(self):
        pts = self.valid_points()
        if not pts:
            return None
        return min(pts, key=lambda p: p.time_ns)


def _parse_int_list(text):
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_header(run, line):
    body = line[2:].strip()

    if body.startswith("batch "):
        run.batches.append(body)
        return

    if body.startswith("conclusion_stop "):
        run.conclusion["stop"] = body.split(None, 1)[1]
        return
    if body.startswith("conclusion_evals "):
        run.conclusion["evals"] = int(body.split()[1])
        return
    if body.startswith("conclusion_best_index "):
        run.conclusion["best_index"] = tuple(int(x) for x in body.split()[1:4])
        return
    if body.startswith("conclusion_best_block "):
        parts = body.split()
        run.conclusion["best_block"] = tuple(int(x) for x in parts[1:4])
        for part in parts[4:]:
            if part.startswith("threads="):
                run.conclusion["best_threads"] = int(part.split("=", 1)[1])
        return
    if body.startswith("conclusion_best_time_ns "):
        run.conclusion["best_time_ns"] = int(body.split()[1])
        return
    if body.startswith("conclusion_cache_written "):
        parts = body.split(None, 2)
        run.conclusion["cache_written"] = bool(int(parts[1]))
        if len(parts) > 2 and parts[2].startswith("path="):
            run.conclusion["cache_path"] = parts[2][5:]
        return

    for axis in ("x", "y", "z"):
        key = f"{axis}_candidates "
        if body.startswith(key):
            vals = body.split("values=", 1)
            run.candidates[axis] = _parse_int_list(vals[1] if len(vals) == 2 else "")
            return

    parts = body.split(None, 1)
    if len(parts) == 2:
        run.header[parts[0]] = parts[1]
    elif parts:
        run.header[parts[0]] = ""


def read_tune_file(path):
    runs = []
    current = None

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line == "# adaptive tune":
                current = TuneRun()
                runs.append(current)
                continue

            if current is None:
                continue

            if line.startswith("# "):
                _parse_header(current, line)
                continue
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 14:
                continue

            try:
                current.points.append(TunePoint(
                    eval_id=int(parts[0]),
                    level=int(parts[1]),
                    region=int(parts[2]),
                    source=parts[3],
                    ix=int(parts[4]),
                    iy=int(parts[5]),
                    iz=int(parts[6]),
                    bx=int(parts[7]),
                    by=int(parts[8]),
                    bz=int(parts[9]),
                    threads=int(parts[10]),
                    time_ns=int(parts[11]),
                    valid=bool(int(parts[12])),
                    predicted=bool(int(parts[13])),
                ))
            except ValueError:
                continue

    return runs


def select_run(runs, which="last"):
    if not runs:
        raise ValueError("no adaptive tune blocks found")

    if which == "last":
        return runs[-1]
    if which == "best":
        usable = [r for r in runs if r.best_point() is not None]
        if not usable:
            raise ValueError("no tune block has valid timing points")
        return min(usable, key=lambda r: r.best_point().time_ns)

    idx = int(which)
    if idx < 0:
        idx += len(runs)
    return runs[idx]
