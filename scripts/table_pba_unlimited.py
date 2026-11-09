"""Tables of the run_pba_unlimited.sh / run_pba_radial_eth3d.sh runs: every variant against `base` (the recommended
configuration) on the scenes both have, per dataset pose AUC@3 / AUC@30 (run_tto_pba.py's compute_pose, the
evaluator's), plus focal error, PBA time per scene and HiRoom recon F1 (eval_tto.py).

Usage: .venv/bin/python scripts/table_pba_unlimited.py [root=workspace/tto/unlimited] [--md out.md]
"""

import json
import os
import sys

DATASETS = [("hiroom", "HiR"), ("eth3d", "ETH"), ("dtu64", "DTU64")]
ORDER = ["base", "base_s1", "radial", "hires", "hires_match", "allpairs", "allpix"]


def load(path):
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    d.pop("mean", None)
    return d


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    if md:
        args.remove(md)
    root = args[0] if args else "workspace/tto/unlimited"
    out = []
    for model in sorted(os.listdir(root)):
        mdir = os.path.join(root, model)
        if not os.path.isdir(mdir) or not os.path.isdir(os.path.join(mdir, "base")):
            continue
        runs = [r for r in ORDER if os.path.isdir(os.path.join(mdir, r))]
        pba = {r: {ds: load(os.path.join(mdir, r, "metric_results", f"{ds}_pba.json")) for ds, _ in DATASETS} for r in runs}
        rec = {r: load(os.path.join(mdir, r, "metric_results", "hiroom_recon_unposed.json")) for r in runs}
        ref_rec = load(os.path.join("workspace/evaluation", model, "metric_results", "hiroom_recon_unposed.json"))
        ref_rec = ref_rec or load(os.path.join("results/da3_bench", model, "hiroom_recon_unposed.json"))  # committed tum1 run
        dsets = [(ds, s) for ds, s in DATASETS if pba["base"][ds]]
        head = ["run"] + [f"{s} A@3 / A@30 [n]" for _, s in dsets] + ["HiR F1", "focal err %", "s/scene"]
        out += [f"\n### {model}\n", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        base = pba["base"]
        # DA3 without PBA (the init of the base run)
        cells = ["no PBA"]
        for ds, _ in dsets:
            sc = sorted(base[ds])
            cells.append(f"{100 * mean(base[ds][s]['auc03_init'] for s in sc):.1f} / {100 * mean(base[ds][s]['auc30_init'] for s in sc):.1f} [{len(sc)}]")
        sc_rec = [s for s in rec["base"] if s in ref_rec]
        cells += [f"{100 * mean(ref_rec[s]['fscore'] for s in sc_rec):.1f}" if sc_rec else "",
                  f"{100 * mean(base[ds][s]['focal_err_init'] for ds, _ in dsets for s in base[ds]):.1f}", "0"]
        out.append("| " + " | ".join(cells) + " |")
        for r in runs:
            cells = [r]
            focal, t = [], []
            for ds, _ in dsets:
                sc = sorted(set(pba[r][ds]) & set(base[ds]))
                if not sc:
                    cells.append("–")
                    continue
                a3 = 100 * mean(pba[r][ds][s]["auc03"] for s in sc)
                a30 = 100 * mean(pba[r][ds][s]["auc30"] for s in sc)
                ref = "" if r == "base" else f" (base {100 * mean(base[ds][s]['auc03'] for s in sc):.1f})"
                cells.append(f"{a3:.1f} / {a30:.1f} [{len(sc)}]{ref}")
                focal += [pba[r][ds][s]["focal_err"] for s in sc]
                t += [pba[r][ds][s]["opt_s"] for s in sc]
            sc_rec = sorted(set(rec[r]) & set(rec["base"]))
            f1 = f"{100 * mean(rec[r][s]['fscore'] for s in sc_rec):.1f}" if sc_rec else ""
            if sc_rec and r != "base":
                f1 += f" (base {100 * mean(rec['base'][s]['fscore'] for s in sc_rec):.1f})"
            cells += [f1, f"{100 * mean(focal):.1f}", f"{mean(t):.0f}"]
            out.append("| " + " | ".join(cells) + " |")
        out.append("\nPer scene AUC@3: " + "; ".join(
            f"{ds}: " + ", ".join(f"{s.split('/')[1] if '/' in s else s} " + "/".join(
                f"{100 * pba[r][ds][s]['auc03']:.0f}" if s in pba[r][ds] else "–" for r in runs) for s in sorted(base[ds]))
            for ds, _ in dsets) + f"  (order: {', '.join(runs)})")
    text = "\n".join(out)
    print(text)
    if md:
        with open(md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
