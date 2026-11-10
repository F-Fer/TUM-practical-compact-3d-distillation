#!/usr/bin/env python3
"""Index every scored run on this machine into one CSV (knowledge/runs.csv), so "has this been run, and what did it get?"
is a grep instead of a search through run folders.

Usage: .venv/bin/python scripts/index_runs.py [--out knowledge/runs.csv]

Scans, relative to the repository root:
  results/pba/runs/**, results/pba/runs6/**   local copies of PBA runs (configs + metric JSONs; not in git since 2026-11-10)
  workspace/tto/**                            all TTO runs on scratch (PBA, Antonin's track BA, DéjàView TTO); auto-deleted
                                              after 30 days untouched, so copy metrics of runs worth keeping to results/
  workspace/evaluation/<model>                feed-forward runs (DA3, baselines, DéjàView with K steps)
  workspace/views/<model>/n<N>                DA3 with N input views
A run is any folder with a metric_results/ subfolder. The same (model, run) found in several places is listed once, from
the copy with the most metric files (ties: results/ first).

Columns:
  model, run        model folder (DA3-SMALL, dvlt_K4, ...) and the run's path below it (groups like runs6/eth3d_all
                    are prefixed)
  kind              pba (has pba_config.json), track (Antonin's run_tto_ba.py: per-scene `ba`/`track_s` keys),
                    feedforward (workspace/evaluation), views, other
  set               dev (10/2/3/4/3 scenes on HiRoom/7-Scenes/ScanNet++/DTU-64/ETH3D), full (30/7/20/13/11), held-out
                    (20/5/17/9/8), mini (5/1/2/2/3, "subset" in REPORT3), drift (5/0/0/2/3), tiny (2/0/0/0/1), else the counts
  auc03_*, auc30    pose AUC in %, per dataset (mean over the run's scenes) and the mean over the datasets present
  f1_*, f1_sets     recon_unposed F1 in % per dataset and the mean over the datasets present (f1_sets says which;
                    the compute plot uses the 4 sets hiroom+7scenes+scannetpp+eth3d)
  tto_s             mean TTO seconds per scene (opt_s; first scene of each dataset replaced by the dataset's median,
                    as in plot_compute.py). For feed-forward runs: median forward seconds from timings.jsonl
  overrides         pba_config.json values that differ from the current DEFAULTS of run_tto_pba.py (bookkeeping keys
                    left out). Options added after a run are absent from its config and not shown
  date              modification date of the run's config or metric files (approximate: copies may be newer)
Numbers depend on GPU (RTX 5090 on tum2, 3070 for session 5), code snapshot and seed; see knowledge/evaluation.md.
"""
import argparse
import ast
import csv
import datetime
import json
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE = ["hiroom", "7scenes", "scannetpp", "dtu64", "eth3d"]
RECON = ["hiroom", "7scenes", "scannetpp", "eth3d"]
SETS = {
    (10, 2, 3, 4, 3): "dev",
    (30, 7, 20, 13, 11): "full",
    (20, 5, 17, 9, 8): "held-out",
    (5, 1, 2, 2, 3): "mini",
    (5, 0, 0, 2, 3): "drift",
    (2, 0, 0, 0, 1): "tiny",
}
SKIP_KEYS = {"datasets", "max_scenes", "scenes", "save", "first_scene", "init_from", "track_every", "wandb"}
MODEL_RE = re.compile(r"^(DA3-|dvlt|VGGT|Pi3|pilot_)")


def load(path):
    try:
        with open(path) as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(r, dict):
        r.pop("mean", None)
        r = {k: v for k, v in r.items() if isinstance(v, dict)}
    return r or None


def defaults():
    """DEFAULTS of run_tto_pba.py, read from the source (importing it needs torch and cv2/libGL)."""
    src = os.path.join(REPO, "scripts", "run_tto_pba.py")
    try:
        tree = ast.parse(open(src).read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULTS" for t in node.targets):
                return ast.literal_eval(node.value)
    except (OSError, SyntaxError, ValueError) as e:
        print(f"[index_runs] can't read DEFAULTS from {src} ({e}); overrides will list every key", file=sys.stderr)
    return None


def fmt(x):
    return "" if x is None else f"{x:.1f}"


def find_runs():
    roots = [("results", "results/pba/runs"), ("results", "results/pba/runs6"), ("workspace", "workspace/tto"),
             ("workspace", "workspace/evaluation"), ("workspace", "workspace/views")]
    for src, root in roots:
        top = os.path.join(REPO, root)
        if not os.path.isdir(top):
            continue
        for dirpath, dirnames, _ in os.walk(top, followlinks=False):
            if "metric_results" in dirnames:
                rel = os.path.relpath(dirpath, top).split(os.sep)
                dirnames[:] = [d for d in dirnames if d != "model_results" and d != "metric_results"]
                yield src, root, dirpath, rel
            dirnames[:] = [d for d in dirnames if d not in ("model_results", "exports", "queue", "snap", "feats")]


def describe(src, root, dirpath, rel, dflt):
    idx = next((i for i, p in enumerate(rel) if MODEL_RE.match(p)), None)
    if idx is None:
        return None
    model = rel[idx]
    group = [p for p in rel[:idx]]
    rest = rel[idx + 1:]
    if root.endswith("runs6"):
        group = ["runs6"] + group
    kind = "other"
    if root.endswith("evaluation"):
        kind, rest = "feedforward", rest or ["eval"]
    elif root.endswith("views"):
        kind = "views"
    run = "/".join(group + rest) or "."
    mdir = os.path.join(dirpath, "metric_results")
    files = os.listdir(mdir)
    cfg = None
    cfg_path = os.path.join(dirpath, "pba_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except ValueError:
            cfg = None
        if kind == "other":
            kind = "pba"
    row = {"model": model, "run": run, "kind": kind, "path": os.path.relpath(dirpath, REPO), "n_files": len(files)}
    counts, a3, a30, opt = [], {}, [], []
    for d in POSE:
        r = load(os.path.join(mdir, f"{d}_pba.json")) or load(os.path.join(mdir, f"{d}_pose.json"))
        if not r:
            counts.append(0)
            continue
        counts.append(len(r))
        vals = list(r.values())
        if all("auc03" in v for v in vals):
            a3[d] = 100 * statistics.mean(v["auc03"] for v in vals)
            a30.append(100 * statistics.mean(v.get("auc30", 0) for v in vals))
        if kind == "other" and any("track_s" in v or "ba" in v for v in vals):
            row["kind"] = kind = "track"
        o = [v["opt_s"] for v in vals if isinstance(v.get("opt_s"), (int, float))]
        if len(o) > 1:
            o[0] = sorted(o[1:])[len(o[1:]) // 2]
        opt += o
    if not any(counts):
        return None
    row["set"] = SETS.get(tuple(counts), "/".join(map(str, counts)))
    for d in POSE:
        row[f"auc03_{d}"] = fmt(a3.get(d))
    row["auc03"] = fmt(statistics.mean(a3.values())) if a3 else ""
    row["auc30"] = fmt(statistics.mean(a30)) if a30 else ""
    f1 = {}
    for d in RECON:
        r = load(os.path.join(mdir, f"{d}_recon_unposed.json"))
        if r and all("fscore" in v for v in r.values()):
            f1[d] = 100 * statistics.mean(v["fscore"] for v in r.values())
        row[f"f1_{d}"] = fmt(f1.get(d))
    row["f1"] = fmt(statistics.mean(f1.values())) if f1 else ""
    row["f1_sets"] = "+".join(f1)
    if kind == "feedforward":
        t = []
        tf = os.path.join(dirpath, "timings.jsonl")
        if os.path.exists(tf):
            for line in open(tf):
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                if not x.get("posed") and "forward_s" in x:
                    t.append(x["forward_s"])
        row["tto_s"] = f"{statistics.median(t):.2f}" if t else ""
    else:
        row["tto_s"] = f"{statistics.mean(opt):.2f}" if opt else ""
    ov = []
    if cfg:
        for k, v in cfg.items():
            if k in SKIP_KEYS:
                continue
            if dflt is not None and k in dflt and dflt[k] == v:
                continue
            ov.append(f"{k}={v}")
    row["overrides"] = " ".join(ov)
    stamp_src = os.path.join(dirpath, "pba_config.json") if cfg else mdir
    row["date"] = datetime.date.fromtimestamp(os.path.getmtime(stamp_src)).isoformat()
    row["source"] = src
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "knowledge", "runs.csv"))
    args = ap.parse_args()
    dflt = defaults()
    best = {}
    for src, root, dirpath, rel in find_runs():
        row = describe(src, root, dirpath, rel, dflt)
        if row is None:
            continue
        key = (row["model"], row["run"])
        old = best.get(key)
        if old is None or row["n_files"] > old["n_files"] or (row["n_files"] == old["n_files"] and row["source"] == "results" and old["source"] != "results"):
            best[key] = row
    cols = ["model", "run", "kind", "set", "auc03", "auc30", "f1", "f1_sets", "tto_s"] + [f"auc03_{d}" for d in POSE] + \
        [f"f1_{d}" for d in RECON] + ["overrides", "date", "path"]
    rows = sorted(best.values(), key=lambda r: (r["model"], r["run"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[index_runs] {len(rows)} runs -> {os.path.relpath(args.out, REPO)}")


if __name__ == "__main__":
    main()
