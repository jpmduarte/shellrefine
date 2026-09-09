"""
Walks a manifest's run_dirs and reads what's already there (profile/config.json,
profile/summary.json, eval_val/summary.json) — no new logging, per the plan. Writes a
markdown table (quality + efficiency side by side) into experiment.md between named
markers, replacing only that section so hand-written prose elsewhere survives re-runs.

Usage:
python aggregate_results.py --manifest manifests/a1_submitted.csv --section STAGE_A1 \
    --title "Stage A1 -- phase 1 loss x lr"
"""

import argparse
import csv
import json
import os


def eval_summary_path(run_dir: str, kind: str, split: str = "val") -> str:
    if kind == "full_pipeline":
        return os.path.join(run_dir, "phase2", f"eval_{split}", "summary.json")
    # phase1_only and phase2_only both pass run_dir straight to train.py/train_phase2.py
    return os.path.join(run_dir, f"eval_{split}", "summary.json")


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def stage_efficiency(profile_summary: dict, stage: str) -> tuple:
    s = (profile_summary or {}).get(stage, {})
    wall = s.get("wall_s")
    vram = s.get("peak_proc_vram_mb") or s.get("peak_alloc_mb")
    return wall, vram


def collect_row(row: dict) -> dict:
    run_dir = row["run_dir"]
    kind    = row["kind"]

    config  = load_json(os.path.join(run_dir, "profile", "config.json")) or {}
    prof    = load_json(os.path.join(run_dir, "profile", "summary.json")) or {}
    ev      = load_json(eval_summary_path(run_dir, kind))

    out = dict(row)
    out["status"] = "ok" if ev is not None else "MISSING"

    if kind == "phase1_only":
        agg = (ev or {}).get("aggregate", {})
        out["dice"]       = agg.get("dice_mean")
        out["mae"]        = agg.get("mae_mean")
        wall_t, vram_t = stage_efficiency(prof, "phase1_train")
        wall_e, vram_e = stage_efficiency(prof, "phase1_eval")
        out["params_mb"]  = config.get("phase1_train", {}).get("model", {}).get("params_mb")
    else:
        agg = (ev or {}).get("aggregate", {})
        out["dice_coarse"]  = (agg.get("dice_coarse")  or {}).get("mean")
        out["dice_refined"] = (agg.get("dice_refined") or {}).get("mean")
        out["hd95_refined"] = (agg.get("hd95_refined") or {}).get("mean")
        out["asd_refined"]  = (agg.get("asd_refined")  or {}).get("mean")
        stage_key = "phase2_train"
        wall_t, vram_t = stage_efficiency(prof, stage_key)
        wall_e, vram_e = stage_efficiency(prof, "phase2_eval")
        out["params_mb"] = config.get(stage_key, {}).get("model", {}).get("params_mb")

    out["train_wall_s"] = wall_t
    out["train_vram_mb"] = vram_t
    out["eval_wall_s"]  = wall_e
    out["eval_vram_mb"] = vram_e
    return out


def fmt(v, nd=4):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def to_markdown(rows: list[dict], title: str) -> str:
    is_p1 = rows and rows[0]["kind"] == "phase1_only"

    if is_p1:
        cols = ["run_id", "status", "loss_p1", "lr_p1", "bw_floor_p1", "bw_coef_p1",
               "gd_coef_p1", "dice", "mae", "train_wall_s", "train_vram_mb", "params_mb"]
        rows_sorted = sorted(rows, key=lambda r: -(r.get("dice") or -1))
    else:
        cols = ["run_id", "status", "loss_p2", "lr_p2", "bw_floor_p2", "bw_coef_p2",
               "dice_coarse", "dice_refined", "hd95_refined", "asd_refined",
               "train_wall_s", "train_vram_mb", "params_mb"]
        rows_sorted = sorted(rows, key=lambda r: -(r.get("dice_refined") or -1))

    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows_sorted:
        lines.append("| " + " | ".join(fmt(r.get(c), 2 if "vram" in c or "wall" in c or "mb" in c else 4)
                                       for c in cols) + " |")
    return "\n".join(lines)


def update_experiment_md(md_path: str, section: str, content: str) -> None:
    start_marker = f"<!-- {section}_START -->"
    end_marker   = f"<!-- {section}_END -->"
    block = f"{start_marker}\n{content}\n{end_marker}"

    existing = ""
    if os.path.exists(md_path):
        with open(md_path) as f:
            existing = f.read()

    if start_marker in existing and end_marker in existing:
        pre  = existing.split(start_marker)[0]
        post = existing.split(end_marker)[1]
        new_content = pre + block + post
    else:
        sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
        new_content = existing + sep + block + "\n"

    with open(md_path, "w") as f:
        f.write(new_content)
    print(f"Updated {section} in {md_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="the _submitted.csv from submit_grid.py")
    parser.add_argument("--section", required=True, help="marker name, e.g. STAGE_A1")
    parser.add_argument("--title", required=True)
    parser.add_argument("--experiment-md", default="experiment.md")
    args = parser.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    collected = [collect_row(r) for r in rows]
    missing = [r["run_id"] for r in collected if r["status"] == "MISSING"]
    if missing:
        print(f"WARNING: {len(missing)} run(s) have no eval summary yet: {missing}")

    md = to_markdown(collected, args.title)
    print("\n" + md + "\n")
    update_experiment_md(args.experiment_md, args.section, md)


if __name__ == "__main__":
    main()
